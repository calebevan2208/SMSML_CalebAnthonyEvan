#!/usr/bin/env python3
"""
Inference Gateway Server with Prometheus Metrics.
Bertugas sebagai perantara (proxy) ke Model Serving Endpoint.
"""

import argparse
import logging
import os
import time
import requests # Library untuk request ke Model Serving
import numpy as np
from flask import Flask, jsonify, request, Response

# Inisialisasi Prometheus
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    Counter = Histogram = Gauge = None
    generate_latest = None
    CONTENT_TYPE_LATEST = 'text/plain; version=0.0.4; charset=utf-8'

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("inference_gateway")
app = Flask(__name__)

# --- Konfigurasi Endpoint Model ---
# Ganti URL ini sesuai alamat serving model (MLflow/Docker)
# Jika jalan di Docker Compose, gunakan nama service, misal: http://model-service:5000/invocations
# Jika jalan manual di local, biasanya: http://127.0.0.1:5000/invocations
MODEL_SERVING_URL = os.environ.get("MODEL_ENDPOINT", "http://127.0.0.1:5000/invocations")

# --- Prometheus Metrics ---
PREDICT_REQUESTS = Counter("predict_requests_total", "Total prediction requests forwarded") if Counter else None
PREDICT_LATENCY = Histogram("predict_latency_seconds", "End-to-end latency including model serving") if Histogram else None
LAST_PREDICTION = Gauge("last_prediction_value", "Last prediction probability returned") if Gauge else None

@app.route("/health", methods=["GET"])
def health():
    """Health check gateway dan koneksi ke model."""
    try:
        # Opsional: Ping model server untuk memastikan dia hidup
        # endpoint /ping biasanya ada di TFServing/MLflow
        ping_url = MODEL_SERVING_URL.replace("/invocations", "/ping") 
        # Cek sederhana: kalau gateway nyala, return ok.
        # Kalau mau lebih canggih, bisa request.get(ping_url)
        return jsonify({"status": "ok", "gateway": "running", "target_model": MODEL_SERVING_URL})
    except Exception as e:
        return jsonify({"status": "warning", "details": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    if generate_latest is None:
        return Response("prometheus_client not installed", status=500)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Menerima request dari user, meneruskan ke Model Serving, 
    dan mengembalikan hasil.
    """
    if PREDICT_REQUESTS:
        PREDICT_REQUESTS.inc()

    t0 = time.time()
    
    # 1. Terima Data dari User
    try:
        payload = request.get_json(force=True)
        if not payload:
            raise ValueError("Empty Payload")
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    # 2. Forward Request ke Model Serving Endpoint
    try:
        # Kita kirim mentah-mentah (pass-through) payloadnya ke model server
        # Pastikan format payload sesuai dengan apa yang diharapkan MLflow/Model (biasanya format "instances" atau "dataframe_split")
        response = requests.post(
            MODEL_SERVING_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Cek jika Model Server error
        if response.status_code != 200:
            logger.error(f"Model Serving Error ({response.status_code}): {response.text}")
            return jsonify({
                "error": "Error from upstream model server", 
                "details": response.text
            }), response.status_code

        # Ambil hasil prediksi
        result_data = response.json()
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to Model Serving at {MODEL_SERVING_URL}")
        return jsonify({"error": "Model Serving Unreachable"}), 503
    except Exception as e:
        logger.exception("Unexpected error during forwarding")
        return jsonify({"error": str(e)}), 500

    # 3. Format Hasil & Update Metrics
    # Asumsi: MLflow serving mengembalikan list prediksi, misal [0.8, 0.1, ...]
    # Atau format {"predictions": [...]}
    
    latency = time.time() - t0
    if PREDICT_LATENCY:
        PREDICT_LATENCY.observe(latency)

    # Parsing sederhana untuk metric Prometheus (ambil probabilitas pertama untuk gauge)
    try:
        # Sesuaikan parsing ini dengan format output model serving Anda
        # Contoh output MLflow: [0.823] atau {"predictions": [0.823]}
        predictions = result_data if isinstance(result_data, list) else result_data.get("predictions", [])
        
        if predictions:
            first_val = predictions[0]
            # Handle jika outputnya list of list [[0.8]]
            if isinstance(first_val, list):
                first_val = first_val[0]
            
            prob = float(first_val)
            if LAST_PREDICTION:
                LAST_PREDICTION.set(prob)
    except Exception:
        pass # Jangan biarkan error parsing metric membatalkan response ke user

    return jsonify({
        "gateway_latency": latency,
        "model_response": result_data
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", default=5001, type=int, help="Port to bind (Gateway Port)")
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:5000/invocations", help="URL of the actual Model Serving")
    args = parser.parse_args()

    # Update URL global jika argumen diberikan
    global MODEL_SERVING_URL
    MODEL_SERVING_URL = args.model_endpoint

    logger.info(f"🚀 Gateway running on http://{args.host}:{args.port}")
    logger.info(f"🔗 Connected to Model Service at: {MODEL_SERVING_URL}")
    
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
