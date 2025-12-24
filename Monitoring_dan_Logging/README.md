# Monitoring dan Logging (Prometheus + Grafana)

Folder ini berisi konfigurasi sederhana untuk memonitor aplikasi (contoh metrics dari `3.prometheus_exporter.py`) menggunakan Prometheus dan Grafana.

## Komponen
- `2.prometheus.yml` — konfigurasi Prometheus (scrape Prometheus sendiri dan exporter pada port 8000).
- `3.prometheus_exporter.py` — contoh exporter Python (menggunakan `prometheus_client` dan `psutil`).
- `Dockerfile.exporter` & `requirements.txt` — build image untuk menjalankan exporter.
- `docker-compose.yml` — menerima 3 service: `prometheus`, `grafana`, `exporter`.
- `grafana/` — provisioning untuk datasource (Prometheus) dan dashboard sederhana.

## Cara cepat (Docker Compose)
Pastikan Docker dan docker-compose terpasang, lalu jalankan di folder ini:

```bash
docker compose up --build -d
```

Service :
- Prometheus: `http://localhost:9090` (UI & query)
- Grafana: `http://localhost:3000` (default user/pass: `admin`/`admin`)
- Exporter: `http://localhost:8000/metrics` (exposed metrics untuk Prometheus)

Grafana akan secara otomatis memuat datasource dan dashboard yang ada di `grafana/dashboards`.

## Menjalankan exporter secara lokal (tanpa Docker)
1. Buat virtualenv (opsional)
2. Install deps:

```bash
python -m pip install -r requirements.txt
python 3.prometheus_exporter.py
```

## Menjalankan inference server (lokal)
Dokumentasi dan contoh inference server untuk model yang sudah dilatih. Server juga mengekspos `/metrics` untuk Prometheus.

Contoh menjalankan server:

```bash
python 7.inference.py --model ../artifacts/model_output/model.h5 --port 5001
```

Endpoints:
- `GET /health` — healthcheck
- `POST /predict` — kirim JSON payload; contoh: `{ "instances": [[...], [...]] }`
- `GET /metrics` — Prometheus metrics yang dapat di-scrape oleh Prometheus

Catatan:
- Jika ada `scaler.pkl` di folder model, server akan memuatnya dan menerapkan transformasi sebelum prediksi.
- Untuk menambahkanInference server ke `docker-compose.yml`, tambahkan service yang menjalankan `python 7.inference.py` dan expose port yang sesuai.

## Catatan & rekomendasi
- Dashboard yang disertakan bersifat minimal; tambahkan panels sesuai kebutuhan produksi.
- Untuk alerting: tambahkan `alerting` rules di `2.prometheus.yml` dan konfigurasi Alertmanager.
- Jika Anda ingin menggunakan server MLflow / system-specific metrics, tambahkan job pada `2.prometheus.yml`.

Jika mau, saya bisa:
- Menambahkan contoh rule alerting dan Alertmanager config ✅
- Menambahkan simple Grafana dashboard yang lebih lengkap (panels & thresholds) ✅
- Menjalankan `docker compose up` di environment untuk smoke-test (perlu izin/akses Docker) ⚠️

