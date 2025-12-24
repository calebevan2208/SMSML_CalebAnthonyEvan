#!/usr/bin/env python3
"""Simple inference server with Prometheus metrics.

Usage:
    python 7.inference.py --model model.h5 --port 5001
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from functools import wraps
from typing import Optional, Union

import numpy as np
from flask import Flask, jsonify, request, Response

# Inisialisasi TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    print("WARNING: TensorFlow not installed. Server will start but predictions will fail.")

# Inisialisasi Prometheus
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    # Dummy classes jika prometheus_client tidak ada
    Counter = Histogram = Gauge = None
    generate_latest = None
    CONTENT_TYPE_LATEST = 'text/plain; version=0.0.4; charset=utf-8'

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("inference_server")
app = Flask(__name__)

# --- Prometheus Metrics Definitions ---
PREDICT_REQUESTS = Counter("predict_requests_total", "Total number of prediction requests") if Counter else None
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Prediction latency in seconds") if Histogram else None
LAST_PREDICTION = Gauge("last_prediction_value", "Last prediction probability returned") if Gauge else None

# --- Global Variables ---
MODEL = None
SCALER = None
FEATURE_NAMES = None

def require_tf(func):
    """Decorator untuk memastikan TensorFlow tersedia sebelum menjalankan fungsi."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if tf is None:
            return jsonify({"error": "TensorFlow is not installed in the environment."}), 500
        return func(*args, **kwargs)
    return wrapper

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/metrics", methods=["GET"])
def metrics():
    """Endpoint untuk scraping Prometheus."""
    if generate_latest is None:
        return Response("prometheus_client not installed", status=500)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
@require_tf
def predict():
    """Endpoint utama untuk inferensi."""
    if PREDICT_REQUESTS:
        PREDICT_REQUESTS.inc()

    t0 = time.time()
    
    # 1. Parsing Input
    try:
        data = request.get_json(force=True)
        if not data:
            raise ValueError("Empty Payload")
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {str(e)}"}), 400

    # 2. Extract Instances
    instances = None
    if isinstance(data, dict):
        if "instances" in data:
            instances = data["instances"]
        elif "data" in data:
            instances = data["data"]
        else:
            instances = [data] # Single dictionary input
    elif isinstance(data, list):
        instances = data
    else:
        return jsonify({"error": "Payload must be a JSON object (dict) or list"}), 400

    # 3. Preprocessing (Convert to NumPy)
    try:
        X = _parse_instances(instances)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # 4. Scaling (Optional)
    if SCALER:
        try:
            X = SCALER.transform(X)
        except Exception as e:
            logger.exception("Scaler transform failed")
            return jsonify({"error": f"Scaler transform failed: {e}"}), 500

    # 5. Prediction
    try:
        preds = MODEL.predict(X, verbose=0)
    except Exception as e:
        logger.exception("Model prediction failed")
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    # 6. Formatting Result
    latency = time.time() - t0
    if PREDICT_LATENCY:
        PREDICT_LATENCY.observe(latency)

    preds = np.asarray(preds).reshape(-1)
    results = []
    
    for p in preds:
        prob = float(p)
        cls = int(prob >= 0.5) # Threshold 0.5
        results.append({"probability": prob, "class": cls})
        
        if LAST_PREDICTION:
            LAST_PREDICTION.set(prob)

    return jsonify({"predictions": results, "latency_seconds": latency})


def _parse_instances(instances):
    """Helper untuk mengubah list/dict menjadi NumPy array."""
    if not isinstance(instances, list):
        raise ValueError("'instances' must be a list")
    
    if len(instances) == 0:
        raise ValueError("Empty instances list")

    # Jika input berupa List of Dicts (misal dari Pandas DF)
    if isinstance(instances[0], dict):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas required for dict inputs but not installed")
            
        df = pd.DataFrame(instances)
        
        # Jika kita punya nama fitur yang disimpan saat training, urutkan kolomnya
        if FEATURE_NAMES:
            # Reindex memastikan urutan kolom sesuai training, isi 0 jika ada yang hilang
            df = df.reindex(columns=FEATURE_NAMES, fill_value=0)
        
        return df.to_numpy(dtype=float)

    # Jika input berupa List of Lists (Raw Values)
    arr = np.asarray(instances, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load(model_path: Optional[str] = None):
    """Memuat Model Keras, Scaler, dan Feature Names."""
    global MODEL, SCALER, FEATURE_NAMES

    # --- 1. Cari Lokasi Model ---
    if model_path is None:
        # Daftar lokasi kemungkinan file model berada
        candidates = [
            "model.h5",                                         # Di folder yang sama (Docker root)
            "model_output/model.h5",                            # Di folder artifacts (Artifact upload)
            os.path.join("artifacts", "model_output", "model.h5"), 
            os.path.join("..", "artifacts", "model_output", "model.h5"),
            "/app/model.h5"                                     # Lokasi umum di Docker
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    if model_path is None or not os.path.exists(model_path):
        logger.error("Model file not found. Checked paths: %s", candidates if model_path is None else model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from: {os.path.abspath(model_path)}")
    try:
        MODEL = load_model(model_path)
    except Exception as e:
        logger.fatal(f"Failed to load Keras model: {e}")
        sys.exit(1)

    # --- 2. Cari Lokasi Scaler ---
    # Coba cari di folder yang sama dengan model
    base_dir = os.path.dirname(model_path)
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    
    # Jika tidak ada, coba cari di path alternatif
    if not os.path.exists(scaler_path):
        scaler_path = "scaler.pkl" 

    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                SCALER = pickle.load(f)
            logger.info(f"Loaded scaler from: {os.path.abspath(scaler_path)}")
        except Exception:
            logger.warning(f"Found scaler at {scaler_path} but failed to load it.")
    else:
        logger.warning("Scaler not found. Running without input scaling.")

    # --- 3. Cari Feature Names ---
    features_path = os.path.join(base_dir, "feature_names.pkl")
    if os.path.exists(features_path):
        try:
            with open(features_path, "rb") as f:
                FEATURE_NAMES = pickle.load(f)
            logger.info(f"Loaded {len(FEATURE_NAMES)} feature names.")
        except Exception:
            logger.warning("Failed to load feature names.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to the .h5 model file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", default=5001, type=int, help="Port to bind")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    # Load model sebelum server jalan
    try:
        load(args.model)
    except Exception as e:
        logger.critical(f"Server startup failed: {e}")
        sys.exit(1)

    logger.info(f"Starting inference server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
