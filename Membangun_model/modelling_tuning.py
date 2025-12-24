"""
modelling_tuning_mlflow.py
--------------------------
Modul Hyperparameter Tuning untuk Deep Learning yang di-refactor menjadi gaya Functional.
Dilengkapi dengan Advanced Logging MLflow (Plots, Metrics, Artifacts) mirip standar Scikit-Learn.

Logic:
1. Load Data -> Split -> Scale -> SMOTE (Original Logic).
2. Custom Grid Search Loop (untuk mencari best params Deep Learning).
3. Logging 'Best Model' secara komprehensif ke MLflow.

Changes:
- Functional Programming Style.
- Integrated Scikit-plot & Seaborn visualization.
- Manual Metric Logging (Train vs Test).

Author: Caleb Anthony
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import mlflow.sklearn
import os
import shutil
import joblib
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
# Compatibility shim: `scikit-plot` does `from scipy import interp`, but recent SciPy
# removed `interp`. Provide a compatible alias before importing `scikitplot` so this
# script runs with modern SciPy versions.
import scipy
if not hasattr(scipy, 'interp'):
    import numpy as _np
    scipy.interp = _np.interp

import scikitplot as skplt
from pathlib import Path
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, precision_score, 
    recall_score, roc_auc_score, ConfusionMatrixDisplay, 
    PrecisionRecallDisplay, RocCurveDisplay, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- KONFIGURASI ENVIRONMENT & AUTH ---
# Menggunakan konfigurasi repo Anda (CalebAnthony) namun dengan style setup environment variable
os.environ['MLFLOW_TRACKING_USERNAME'] = 'calebevan2208' # Sesuaikan dengan DagsHub User Anda
os.environ['MLFLOW_TRACKING_PASSWORD'] = '2205072872dc0f9aaf561b740bc01685c6417cca' # Masukkan token jika perlu
REMOTE_URI = 'https://dagshub.com/calebevan2208/SMSML_CalebAnthony.mlflow' 

mlflow.set_tracking_uri(REMOTE_URI)
mlflow.set_experiment("CalebAnthony_Churn_DeepLearning_Tuning")

# --- SETUP LOGGING CONSOLE ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def build_model(params, input_dim):
    """Helper function untuk menyusun arsitektur Keras."""
    model = Sequential([
        Dense(params['units_layer1'], activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(params['dropout_rate']),
        Dense(params['units_layer2'], activation='relu'),
        Dropout(params['dropout_rate']),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

def train_with_tuning():
    logger.info("=== Memulai Pipeline Tuning Deep Learning ===")

    # 1. SETUP PATH & DATA
    BASE_DIR = Path(__file__).resolve().parent

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

    if DATA_PATH is None:
        checked = '\n'.join(str(p) for p in possible_paths)
        logger.error(f"Data tidak ditemukan. Telah memeriksa:\n{checked}")
        return

    df = pd.read_csv(DATA_PATH)
    target_col = 'default'
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    # Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE (Handling Imbalance)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Class Weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_resampled),
        y=y_train_resampled
    )
    class_weight_dict = dict(enumerate(class_weights))

    # 2. GRID SEARCH MANUAL (Custom Loop)
    # Karena Keras tidak support GridSearchCV scikit-learn secara native tanpa wrapper,
    # kita gunakan loop sederhana untuk mencari parameter terbaik.
    
    param_grid = {
        'units_layer1': [64, 128],
        'units_layer2': [32, 64],
        'dropout_rate': [0.2, 0.3],
        'learning_rate': [0.001], # Disederhanakan untuk demo
        'batch_size': [32]
    }
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    logger.info(f"Total kombinasi hyperparameter: {len(param_combinations)}")
    
    best_score = 0.0
    best_model = None
    best_params = None
    best_history = None

    # Loop Tuning
    for i, params in enumerate(param_combinations):
        print(f"Testing kombinasi {i+1}/{len(param_combinations)}: {params}")
        
        model = build_model(params, input_dim=X_train.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train_resampled, y_train_resampled,
            validation_data=(X_test, y_test),
            epochs=20, # Epochs per trial
            batch_size=params['batch_size'],
            callbacks=[early_stop],
            class_weight=class_weight_dict,
            verbose=0
        )
        
        # Evaluasi sederhana untuk seleksi
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int)
        current_f1 = f1_score(y_test, y_pred, zero_division=0)
        
        if current_f1 > best_score:
            best_score = current_f1
            best_model = model
            best_params = params
            best_history = history
            print(f"  -> New Best Model Found! F1: {current_f1:.4f}")

    logger.info(f"Tuning Selesai. Best F1: {best_score:.4f}")
    logger.info(f"Best Params: {best_params}")

    # 3. LOGGING BEST MODEL TO MLFLOW (Detailed Version)
    # Kita buka Run baru khusus untuk menyimpan 'Best Model' beserta semua plot canggihnya
    
    # Generate Predictions untuk Logging
    y_train_prob = best_model.predict(X_train_resampled, verbose=0).ravel()
    y_train_pred = (y_train_prob > 0.5).astype(int)
    
    y_test_prob = best_model.predict(X_test, verbose=0).ravel()
    y_test_pred = (y_test_prob > 0.5).astype(int)

    with mlflow.start_run(run_name="Best_DeepLearning_Model"):
        mlflow.log_params(best_params)
        
        # A. Manual Logging Metrics (Train vs Test)
        metrics = {
            "training_accuracy": accuracy_score(y_train_resampled, y_train_pred),
            "training_f1": f1_score(y_train_resampled, y_train_pred),
            "training_log_loss": log_loss(y_train_resampled, y_train_prob),
            "training_roc_auc": roc_auc_score(y_train_resampled, y_train_prob),
            
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_log_loss": log_loss(y_test, y_test_prob),
            "test_precision": precision_score(y_test, y_test_pred),
            "test_recall": recall_score(y_test, y_test_pred),
            "test_roc_auc": roc_auc_score(y_test, y_test_prob)
        }
        
        for m_name, m_val in metrics.items():
            mlflow.log_metric(m_name, m_val)
            
        # B. Saving Artifacts Locally First
        artifact_dir = os.path.join("artifacts", "best_model_artifacts")
        if os.path.exists(artifact_dir):
            shutil.rmtree(artifact_dir)
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Save Scaler (Critical!)
        joblib.dump(scaler, os.path.join(artifact_dir, "scaler.pkl"))

        # C. PLOTTING (Adapted for Keras / Non-Sklearn Estimators)
        
        # 1. Confusion Matrix (Test Set)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax, cmap='Blues')
        plt.title("Confusion Matrix (Test Set)")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # 2. ROC Curve
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_test_prob, ax=ax, name="Best Model")
        plt.title("ROC Curve")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        # 3. Precision-Recall Curve
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_test, y_test_prob, ax=ax, name="Best Model")
        plt.title("Precision-Recall Curve")
        plt.savefig("pr_curve.png")
        mlflow.log_artifact("pr_curve.png")
        plt.close()

        # 4. Training History (Pengganti Feature Importance untuk Deep Learning)
        # Deep Learning tidak punya feature importance langsung seperti Random Forest
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(best_history.history['loss'], label='Train Loss')
        plt.plot(best_history.history['val_loss'], label='Val Loss')
        plt.plot(best_history.history['accuracy'], label='Train Acc')
        plt.plot(best_history.history['val_accuracy'], label='Val Acc')
        plt.title("Learning Curves (Loss & Accuracy)")
        plt.legend()
        plt.savefig("learning_curve.png")
        mlflow.log_artifact("learning_curve.png")
        plt.close()

        # 5. Lift Curve (Scikit-plot)
        # Membutuhkan format probabilitas (N, 2) untuk class 0 dan 1
        y_probas_formatted = np.column_stack((1 - y_test_prob, y_test_prob))
        
        fig, ax = plt.subplots()
        skplt.metrics.plot_lift_curve(y_test, y_probas_formatted, ax=ax)
        plt.title("Lift Curve")
        plt.savefig("lift_curve.png")
        mlflow.log_artifact("lift_curve.png")
        plt.close()

        # D. Save Model & Artifacts Folder
        # Save Keras Model
        best_model.save(os.path.join(artifact_dir, "model.h5"))
        
        # Log Artifact folder content
        mlflow.log_artifacts(artifact_dir, artifact_path="model_output")
        
        # Log Keras Model Native
        mlflow.tensorflow.log_model(best_model, "keras_model")
        
        # Cleanup Local
        shutil.rmtree(artifact_dir)
        
        print("\n=== Evaluasi Akhir (Test Set) ===")
        print(classification_report(y_test, y_test_pred))
        print(f"Artifacts logged to: {REMOTE_URI}")

if __name__ == "__main__":
    train_with_tuning()

