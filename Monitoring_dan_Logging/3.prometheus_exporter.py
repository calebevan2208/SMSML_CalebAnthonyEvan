"""
3.prometheus_exporter.py
------------------------
Script untuk mengekspos metrics sistem dan aplikasi ke Prometheus.
Menggunakan library `prometheus_client`.
"""

from prometheus_client import start_http_server, Summary, Counter, Gauge
import random
import time
import psutil

# --- METRICS DEFINITION ---
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions made')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')

@REQUEST_TIME.time()
def process_request(t):
    """A dummy function that takes some time."""
    time.sleep(t)

def update_system_metrics():
    """Update system metrics (CPU, Memory)."""
    MEMORY_USAGE.set(psutil.virtual_memory().used)
    CPU_USAGE.set(psutil.cpu_percent())

if __name__ == '__main__':
    # Start up the server to expose the metrics.
    start_http_server(8000)
    print("Prometheus Exporter running on port 8000")
    
    # Generate some dummy metrics
    while True:
        process_request(random.random())
        PREDICTION_COUNTER.inc()
        update_system_metrics()
        time.sleep(1)
