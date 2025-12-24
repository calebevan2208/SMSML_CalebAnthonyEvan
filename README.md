# SMSML_CalebAnthony

Project ini adalah implementasi end-to-end Machine Learning pipeline untuk prediksi churn nasabah kartu kredit. Ini mencakup data ingestion, preprocessing, EDA, model training (TensorFlow/Keras), experiment tracking (MLflow), monitoring (Prometheus + Grafana), dan CI with GitHub Actions.

---

## 📂 Struktur Project (ringkas)
- `Eksperimen_SML_CalebAnthony/` — Eksperimen, EDA, dan preprocessing
- `Membangun_model/` — Training, tuning, dan model artifacts
- `Monitoring_dan_Logging/` — Prometheus exporter, Docker Compose, Grafana provisioning
- `Workflow-CI/` & `.github/workflows/` — CI workflows and MLProject entry points
- `mlruns/` & `artifacts/` — MLflow runs & logged artifacts

---

## ✅ Recent additions (Dec 2025)
- Added sample datasets:
  - `Eksperimen_SML_CalebAnthonyEvan/preprocessing/churn_raw/data.csv`
  - `Eksperimen_SML_CalebAnthonyEvan/preprocessing/churn_preprocessing/clean_data.csv`
- Added many visualization images (learning curve, ROC, confusion matrix) and Grafana/Prometheus screenshots under `Monitoring_dan_Logging/`.
- Monitoring stack and provisioning (Prometheus + Grafana + exporter) with `docker-compose` and dashboard JSONs.
- CI: lightweight GitHub Actions workflow at `.github/workflows/ci.yml` with optional `workflow_dispatch` for manual runs.

---

## 🚀 Quick start
1. Install Python deps:

```bash
python -m pip install -r Membangun_model/requirements.txt
```

2. Run training (baseline):

```bash
python Membangun_model/modelling.py
```

3. Run monitoring stack (Docker Compose):

```bash
docker compose -f Monitoring_dan_Logging/docker-compose.yml up --build -d
# Grafana: http://localhost:3000 (default admin:admin — change after first login)
```

4. Run exporter locally:

```bash
python -m pip install -r Monitoring_dan_Logging/requirements.txt
python Monitoring_dan_Logging/3.prometheus_exporter.py
```

---

## 💡 Notes & recommendations
- Large binary/data files (PNGs, CSVs) were added to the repo recently. To avoid repository bloat, consider enabling Git LFS:

```bash
git lfs install
git lfs track "*.png" "*.csv"
# Commit .gitattributes
# Optionally rewrite history: git lfs migrate import --include="*.png,*.csv"
```

- MLflow: scripts are resilient — they will fall back to a local `mlruns/` folder if a remote tracking server is unavailable.

---

If you'd like, I can also enable Git LFS and migrate the current large files into LFS for you.
