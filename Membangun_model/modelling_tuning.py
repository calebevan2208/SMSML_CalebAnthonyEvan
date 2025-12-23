"""
modelling_tuning.py
-------------------
Modul ini menjalankan Hyperparameter Tuning untuk Deep Learning Model menggunakan TensorFlow.
Seluruh hasil eksperimen dilacak menggunakan MLflow (ver 2.19.0).
Script ini otomatis memilih model terbaik berdasarkan metric validasi dan menyimpan artifacts.

Features:
- Custom Grid Search Loop for Deep Learning.
- MLflow Tracking (Params, Metrics, Artifacts).
- Prevention of Data Leakage (Split-then-Scale).
- Automatic Best Model Selection.
- Imbalanced Data Handling (SMOTE + Class Weights).

Environment: Python 3.12.7 | MLflow 2.19.0
Author: Caleb Anthony (Automated by System)
Date: 2025-10-30
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import dagshub
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- KONFIGURASI LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- KONFIGURASI PROJECT & MLFLOW ---
class TuningConfig:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    # Mengambil data dari preprocessing yang sudah bersih
    DATA_PATH = BASE_DIR.parent / 'Eksperimen_SML_CalebAnthony' / 'churn_preprocessing' / 'clean_data.csv'
    ARTIFACTS_DIR = BASE_DIR / 'artifacts'
    
    # MLflow Settings
    EXPERIMENT_NAME = "CalebAnthony_Churn_DeepLearning_v1"
    
    # Tuning Grid (Search Space)
    # Sense bisa menambah kombinasi di sini
    PARAM_GRID = {
        'units_layer1': [64, 128],
        'units_layer2': [32, 64],
        'dropout_rate': [0.2, 0.3],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32]
    }
    
    # Static Params
    EPOCHS = 20 # Maksimum epoch per trial
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

class ChurnTuner:
    """
    Kelas Orchestrator untuk Tuning dan Tracking MLflow.
    """
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_accuracy = 0.0
        self.best_model = None
        
        # Setup Output
        TuningConfig.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Setup MLflow
        dagshub.init(repo_owner='calebevan2208', repo_name='SMSML_CalebAnthony', mlflow=True)
        mlflow.set_experiment(TuningConfig.EXPERIMENT_NAME)
        logger.info(f"MLflow Experiment set to: {TuningConfig.EXPERIMENT_NAME}")

    def load_and_prepare_data(self) -> None:
        """
        Load data, split, dan scale.
        PENTING: Mencegah Data Leakage dengan fit scaler hanya di training data.
        """
        if not TuningConfig.DATA_PATH.exists():
            logger.critical(f"Data source not found at: {TuningConfig.DATA_PATH}")
            sys.exit(1)
            
        logger.info("Loading dataset...")
        df = pd.read_csv(TuningConfig.DATA_PATH)
        
        target_col = 'default'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
            
        X = df.drop(columns=[target_col], errors='ignore')
        y = df[target_col]
        
        # Split
        logger.info("Splitting dataset...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TuningConfig.TEST_SIZE, random_state=TuningConfig.RANDOM_STATE, stratify=y
        )
        
        # Scale
        logger.info("Scaling features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Apply SMOTE
        logger.info("Menerapkan SMOTE...")
        smote = SMOTE(random_state=TuningConfig.RANDOM_STATE)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        # Simpan Scaler untuk Production
        scaler_path = TuningConfig.ARTIFACTS_DIR / 'scaler_production.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Production Scaler saved to: {scaler_path}")

    def build_model(self, params: Dict[str, Any], input_dim: int) -> Sequential:
        """
        Membuat arsitektur model secara dinamis berdasarkan parameter.
        """
        model = Sequential()
        
        # Layer 1
        model.add(Dense(params['units_layer1'], activation='relu', input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
        
        # Layer 2
        model.add(Dense(params['units_layer2'], activation='relu'))
        model.add(Dropout(params['dropout_rate']))
        
        # Output Layer (Binary)
        model.add(Dense(1, activation='sigmoid'))
        
        optimizer = Adam(learning_rate=params['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        return model

    def run_tuning(self) -> None:
        """
        Menjalankan loop eksperimen berdasarkan kombinasi parameter (Grid Search Manual).
        Setiap kombinasi akan dicatat sebagai satu 'Run' di MLflow.
        """
        logger.info("Starting Hyperparameter Tuning...")
        
        # Generate semua kombinasi parameter
        keys, values = zip(*TuningConfig.PARAM_GRID.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        logger.info(f"Total combinations to test: {len(param_combinations)}")
        
        for i, params in enumerate(param_combinations):
            run_name = f"Run_{i+1}_lr_{params['learning_rate']}_units_{params['units_layer1']}"
            logger.info(f"--- Executing {run_name} ---")
            
            with mlflow.start_run(run_name=run_name):
                # 1. Log Hyperparameters
                mlflow.log_params(params)
                mlflow.log_param("epochs", TuningConfig.EPOCHS)
                
                # 2. Build & Train Model
                model = self.build_model(params, self.X_train.shape[1])
                
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                
                # Class Weights (Stabilization)
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(self.y_train),
                    y=self.y_train
                )
                class_weight_dict = dict(enumerate(class_weights))
                
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_test, self.y_test),
                    epochs=TuningConfig.EPOCHS,
                    batch_size=params['batch_size'],
                    callbacks=[early_stop],
                    class_weight=class_weight_dict,
                    verbose=0 # Silent training agar log tidak penuh
                )
                
                # 3. Evaluate
                y_pred_prob = model.predict(self.X_test)
                y_pred = (y_pred_prob > 0.5).astype(int)
                
                # Metrics Calculation
                acc = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, zero_division=0)
                auc = roc_auc_score(self.y_test, y_pred_prob)
                rec = recall_score(self.y_test, y_pred, zero_division=0)
                
                logger.info(f"   Result -> Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Recall: {rec:.4f}")
                
                # 4. Log Metrics to MLflow
                mlflow.log_metrics({
                    "test_accuracy": acc,
                    "test_f1_score": f1,
                    "test_auc": auc,
                    "test_recall": rec,
                    "final_train_loss": history.history['loss'][-1],
                    "final_val_loss": history.history['val_loss'][-1]
                })
                
                # 5. Log Model to MLflow (Artifacts)
                # MLflow 2.19+ supports logging tensorflow/keras models directly
                mlflow.tensorflow.log_model(model, "model")
                
                # 6. Best Model Tracking Logic (Changed to F1-Score)
                # Menggunakan F1-Score sebagai metric utama untuk seleksi model
                if f1 > self.best_accuracy: # Variable name is best_accuracy but used for best score
                    self.best_accuracy = f1
                    self.best_model = model
                    logger.info(f"   >>> New Best Model Found! (F1-Score: {f1:.4f})")
                    
                    # Simpan 'Best Model' secara lokal untuk kemudahan akses deployment
                    best_model_path = TuningConfig.ARTIFACTS_DIR / 'best_churn_model.h5'
                    model.save(best_model_path)
                    mlflow.log_artifact(str(best_model_path), artifact_path="best_model_file")

    def finish(self):
        """Final summary."""
        logger.info("="*50)
        logger.info("TUNING COMPLETED")
        logger.info(f"Best Score (F1) achieved: {self.best_accuracy:.4f}")
        logger.info(f"Check MLflow UI for details. Run: 'mlflow ui'")
        logger.info("="*50)

if __name__ == "__main__":
    tuner = ChurnTuner()
    tuner.load_and_prepare_data()
    tuner.run_tuning()
    tuner.finish()
