"""
modelling_mlflow.py
-------------------
Modul training Baseline Model (Deep Learning) yang sudah terintegrasi dengan MLflow.
Script ini mengadaptasi logika preprocessing original (Scaling + SMOTE) ke dalam
structure MLflow experiment tracking.

Changes:
- Refactored from Class-based to Functional-based (sesuai request).
- Integrated MLflow Tracking & Artifact Logging.
- Auto-logging metrics, params, and models using mlflow.tensorflow.

Author: Caleb Anthony
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import os
import shutil
import mlflow
import mlflow.tensorflow
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# --- KONFIGURASI MLFLOW ---
# Attempt to use a remote MLflow server; if it's unreachable, fall back to local file-based tracking
MLFLOW_SERVER_URI = "http://127.0.0.1:5000"
LOCAL_MLRUNS_URI = "file:./mlruns"

# Prefer remote server when available (common in dev with `mlflow server --port 5000`),
# but gracefully degrade to a local `mlruns` folder so the script still runs in CI/dev containers.
mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
os.makedirs("mlruns", exist_ok=True)

client = mlflow.tracking.MlflowClient()

# pastikan experiment ada
exp = client.get_experiment_by_name("Churn_Prediction_Deep_Learning")

if exp is None:
    client.create_experiment("Churn_Prediction_Deep_Learning")

mlflow.set_experiment("Churn_Prediction_Deep_Learning")

try:
    mlflow.set_experiment("Churn_Prediction_Deep_Learning")
except Exception as e:
    print("fallback ke local")


def train_model():
    print("=== Memulai Pipeline Training dengan MLflow ===")
    
    # 1. SETUP PATH & DATA
    # Menggunakan path relative seperti script asli
    BASE_DIR = Path(__file__).resolve().parent

    # Cari data di beberapa lokasi yang mungkin ada di repo (toleran terhadap penamaan folder)
    possible_paths = [
        BASE_DIR.parent / 'Eksperimen_SML_CalebAnthonyEvan' / 'churn_preprocessing' / 'clean_data.csv',
        BASE_DIR.parent / 'Eksperimen_SML_CalebAnthonyEvan' / 'preprocessing' / 'churn_preprocessing' / 'clean_data.csv',
        BASE_DIR / 'churn_preprocessing' / 'clean_data.csv',
    ]

    DATA_PATH = None
    for p in possible_paths:
        if p.exists():
            DATA_PATH = p
            break

    # Cek keberadaan data
    if DATA_PATH is None:
        checked = '\n'.join(str(p) for p in possible_paths)
        print(f"ERROR: Data tidak ditemukan. Telah memeriksa:\n{checked}")
        return

    print(f"Loading dataset dari: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 2. PREPROCESSING (Original Logic)
    target_col = 'default'
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    # Split Data (Stratified)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    # Scaling (StandardScaler)
    print("Scaling fitur...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE (Handling Imbalance)
    print("Menerapkan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Compute Class Weights (Untuk kestabilan loss function)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # 3. SETUP MODEL ARSITEKTUR
    input_dim = X_train.shape[1]
    
    def build_dnn_model():
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Recall', 'Precision']
        )
        return model

    # 4. MLFLOW RUN
    # Mengaktifkan Autologging untuk TensorFlow/Keras
    mlflow.tensorflow.autolog(log_models=True, log_datasets=False) #----> autolog()

    with mlflow.start_run(run_name="Baseline_DNN_Caleb"):
        print("MLflow Run Started...")
        
        # Build Model
        model = build_dnn_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        # Training
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        # 5. SAVING ARTIFACTS LOCALLY & LOGGING TO MLFLOW
        # Kita buat folder artifacts lokal dulu seperti di contoh
        local_artifact_dir = os.path.join("artifacts", "model_output")

        # Bersihkan folder jika sudah ada (agar fresh)
        if os.path.exists(local_artifact_dir):
            shutil.rmtree(local_artifact_dir)
        os.makedirs(local_artifact_dir, exist_ok=True)

        # Save Keras Model (.h5)
        model_save_path = os.path.join(local_artifact_dir, "model.h5")
        model.save(model_save_path)
        print(f"Model tersimpan lokal di: {model_save_path}")

        # Save Scaler (.pkl) -> PENTING: Scaler wajib satu paket dengan model
        scaler_save_path = os.path.join(local_artifact_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_save_path)
        print(f"Scaler tersimpan lokal di: {scaler_save_path}")
        
        # Generate & Save Training Plot
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training History')
        plt.legend()
        plot_path = os.path.join(local_artifact_dir, "loss_curve.png")
        plt.savefig(plot_path)
        plt.close()

        # Upload seluruh folder artifacts lokal ke MLflow Server
        print("Mengupload artifacts ke MLflow Server...")
        mlflow.log_artifacts(local_artifact_dir, artifact_path="custom_artifacts")

        # 6. EVALUASI AKHIR
        print("\n=== Evaluasi Model (Test Set) ===")
        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        report = classification_report(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_probs)
        
        print(report)
        print(f"ROC-AUC Score: {auc_score:.4f}")
        
        # Log manual metrics tambahan jika perlu (Autolog sudah handle sebagian besar)
        mlflow.log_metric("test_roc_auc", auc_score)

if __name__ == "__main__":
    train_model()


