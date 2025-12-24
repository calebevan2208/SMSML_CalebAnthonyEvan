#!/usr/bin/env python3
"""Simple inference server with Prometheus metrics.

Usage:
    python Monitoring_dan_Logging/inference.py --model ../artifacts/model_output/model.h5 --port 5001

Endpoints:
    GET /health        -> 200 OK
    POST /predict      -> Accepts JSON payloads and returns predictions
    GET /metrics       -> Prometheus metrics

Accepted payloads for /predict (JSON):
  - {"instances": [[...], [...]]}  -> list of numeric feature lists
  - {"instances": [{"col1": val, "col2": val}, ...]} -> list of dicts (order inferred)
  - Or a plain JSON list [: list] of lists or dicts

Notes: If a scaler file `scaler.pkl` is present next to the model, it will be used to transform input features.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from functools import wraps
from io import BytesIO

import numpy as np
from flask import Flask, jsonify, request, Response

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
except Exception:
    # If prometheus_client is missing, create stubs that raise helpful errors when used
    Counter = Histogram = Gauge = None
    generate_latest = None
    CONTENT_TYPE_LATEST = 'text/plain; version=0.0.4; charset=utf-8'

# App & logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inference_server")
app = Flask(__name__)

# Prometheus metrics
PREDICT_REQUESTS = Counter("predict_requests_total", "Total number of prediction requests") if Counter else None
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Prediction latency in seconds") if Histogram else None
LAST_PREDICTION = Gauge("last_prediction_value", "Last prediction probability returned") if Gauge else None

MODEL = None
SCALER = None
FEATURE_NAMES = None


def require_tf(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if tf is None:
            return jsonify({"error": "TensorFlow is not installed in the environment."}), 500
        return func(*args, **kwargs)

    return wrapper


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics", methods=["GET"])
def metrics():
    if generate_latest is None:
        return Response("prometheus_client not installed", status=500)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
@require_tf
def predict():
    if PREDICT_REQUESTS:
        PREDICT_REQUESTS.inc()

    data = None
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if data is None:
        return jsonify({"error": "Empty JSON payload"}), 400

    # Extract instances
    instances = None
    if isinstance(data, dict) and "instances" in data:
        instances = data["instances"]
    elif isinstance(data, list):
        instances = data
    elif isinstance(data, dict) and "data" in data:
        instances = data["data"]
    else:
        # maybe top-level dict with single sample
        instances = [data]

    # Convert to numpy array
    try:
        X = _parse_instances(instances)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Apply scaler if available
    if SCALER is not None:
        try:
            X = SCALER.transform(X)
        except Exception as e:
            logger.exception("Scaler transform failed")
            return jsonify({"error": f"Scaler transform failed: {e}"}), 500

    # Run prediction
    import time

    t0 = time.time()
    try:
        preds = MODEL.predict(X)
    except Exception as e:
        logger.exception("Model prediction failed")
        return jsonify({"error": f"Model prediction failed: {e}"}), 500
    latency = time.time() - t0

    if PREDICT_LATENCY:
        PREDICT_LATENCY.observe(latency)

    # Convert preds to probabilities and classes
    preds = np.asarray(preds).reshape(-1)
    results = []
    for p in preds:
        cls = int(p >= 0.5)
        results.append({"probability": float(p), "class": cls})
        if LAST_PREDICTION:
            LAST_PREDICTION.set(float(p))

    return jsonify({"predictions": results, "latency_seconds": latency})


def _parse_instances(instances):
    # instances can be list of lists or list of dicts
    if not isinstance(instances, list):
        raise ValueError("'instances' must be a list of samples")

    if len(instances) == 0:
        raise ValueError("Empty instances list")

    # detect if list of dicts
    if isinstance(instances[0], dict):
        # build feature matrix using union of keys; if FEATURE_NAMES known, use that order
        import pandas as pd

        df = pd.DataFrame(instances)
        if FEATURE_NAMES:
            # reindex to FEATURE_NAMES, fill missing with 0
            df = df.reindex(columns=FEATURE_NAMES, fill_value=0)
        return df.to_numpy(dtype=float)

    # otherwise assume list of lists / scalars
    arr = np.asarray(instances, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load(model_path: str | None = None):
    global MODEL, SCALER, FEATURE_NAMES

    if model_path is None:
        # try some defaults
        candidates = [
            os.path.join("..", "artifacts", "model_output", "model.h5"),
            os.path.join("..", "Membangun_model", "artifacts", "model.h5"),
            os.path.join("..", "artifacts", "model.h5"),
            os.path.join("artifacts", "model_output", "model.h5"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if model_path is None:
        logger.error("No model path specified and no default model found")
        raise FileNotFoundError("model file not found")

    logger.info("Loading model from %s", model_path)
    MODEL = load_model(model_path)

    # try to load scaler next to model
    scaler_path = os.path.join(os.path.dirname(model_path), "scaler.pkl")
    if not os.path.exists(scaler_path):
        # also try repository artifact path
        scaler_path = os.path.join("..", "artifacts", "model_output", "scaler.pkl")

    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                SCALER = pickle.load(f)
            logger.info("Loaded scaler from %s", scaler_path)
        except Exception:
            logger.exception("Failed to load scaler at %s", scaler_path)

    # try to infer feature names if a file exists
    features_path = os.path.join(os.path.dirname(model_path), "feature_names.pkl")
    if os.path.exists(features_path):
        try:
            with open(features_path, "rb") as f:
                FEATURE_NAMES = pickle.load(f)
            logger.info("Loaded feature names (%d) from %s", len(FEATURE_NAMES), features_path)
        except Exception:
            logger.exception("Failed to load feature names at %s", features_path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to the Keras model (.h5 or SavedModel dir)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5001, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    try:
        load(args.model)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        sys.exit(1)

    logger.info("Starting inference server on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
