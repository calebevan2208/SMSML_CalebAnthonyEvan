"""
modelling.py
------------
Modul ini bertanggung jawab untuk melatih Baseline Model menggunakan Deep Learning (TensorFlow/Keras).
Script ini mencakup pipeline standar ML: Loading -> Splitting -> Scaling -> Training -> Evaluation -> Saving.

Fitur Utama:
1. Arsitektur Deep Neural Network (DNN) dengan Dropout untuk regularisasi.
2. Early Stopping untuk mencegah Overfitting dan menghemat waktu komputasi.
3. Penyimpanan Artifacts (Model .h5 & Scaler .pkl) untuk deployment.
4. Visualisasi Learning Curve (Loss & Accuracy).
5. Penanganan Imbalanced Data (SMOTE + Class Weights).

Author: Caleb Anthony (Automated by System)
Date: 2025-10-30
Version: 1.0 (Baseline Deep Learning)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# --- KONFIGURASI LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- KONFIGURASI PROJECT ---
class ModelConfig:
    # Path Handling
    BASE_DIR = Path(__file__).resolve().parent
    # Mengambil data bersih dari folder eksperimen (Single Source of Truth)
    DATA_PATH = BASE_DIR.parent / 'Eksperimen_SML_CalebAnthony' / 'churn_preprocessing' / 'clean_data.csv'
    
    # Output Artifacts
    ARTIFACTS_DIR = BASE_DIR / 'artifacts'
    MODEL_SAVE_PATH = ARTIFACTS_DIR / 'baseline_model.h5'
    SCALER_SAVE_PATH = ARTIFACTS_DIR / 'scaler.pkl'
    HISTORY_PLOT_PATH = ARTIFACTS_DIR / 'training_history.png'
    
    # Model Parameters (Baseline)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EPOCHS = 50          # Batas maksimal epoch
    BATCH_SIZE = 32      # Standar untuk dataset ukuran menengah
    LEARNING_RATE = 0.001
    PATIENCE = 5         # Stop training jika tidak ada perbaikan selama 5 epoch

class ChurnBaselineTrainer:
    """
    Kelas Trainer untuk Baseline Deep Learning Model.
    """
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        
        # Buat folder artifacts jika belum ada
        ModelConfig.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Trainer diinisialisasi. Artifacts akan disimpan di: {ModelConfig.ARTIFACTS_DIR}")

    def load_and_split_data(self) -> None:
        """
        Memuat data, memisahkan fitur & target, melakukan split, dan scaling.
        PENTING: Scaling dilakukan SETELAH split untuk mencegah Data Leakage.
        """
        if not ModelConfig.DATA_PATH.exists():
            logger.critical(f"Data tidak ditemukan di: {ModelConfig.DATA_PATH}")
            sys.exit(1)
            
        logger.info("Memuat dataset...")
        self.df = pd.read_csv(ModelConfig.DATA_PATH)
        
        # Definisi Target
        target_col = 'default'
        if target_col not in self.df.columns:
            raise ValueError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset.")
            
        # Pisahkan X (Features) dan y (Target)
        X = self.df.drop(columns=[target_col], errors='ignore')
        y = self.df[target_col]
        
        logger.info(f"Dimensi Fitur: {X.shape}, Target Distribution: {y.value_counts().to_dict()}")

        # Train-Test Split (Stratified agar proporsi churn terjaga)
        logger.info("Membagi data menjadi Train dan Test Set...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=ModelConfig.TEST_SIZE, 
            random_state=ModelConfig.RANDOM_STATE, 
            stratify=y
        )
        
        # Scaling (Standardization)
        logger.info("Melakukan Scaling fitur (StandardScaler)...")
        # Fit hanya pada TRAIN, Transform pada TRAIN dan TEST
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Apply SMOTE to Training Data (Handle Imbalance)
        logger.info("Menerapkan SMOTE untuk menangani Imbalanced Data...")
        smote = SMOTE(random_state=ModelConfig.RANDOM_STATE)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        logger.info(f"Distribusi Target setelah SMOTE: {self.y_train.value_counts().to_dict()}")

        # Simpan Scaler (Penting untuk tahap Deployment/Serving nanti)
        joblib.dump(self.scaler, ModelConfig.SCALER_SAVE_PATH)
        logger.info(f"Scaler tersimpan di: {ModelConfig.SCALER_SAVE_PATH}")

    def build_model(self) -> None:
        """
        Membangun arsitektur Neural Network (Multilayer Perceptron).
        Arsitektur: Input -> Dense(64) -> Dropout -> Dense(32) -> Output(Sigmoid)
        """
        input_dim = self.X_train.shape[1]
        logger.info(f"Membangun model Deep Learning dengan input dimension: {input_dim}...")
        
        self.model = Sequential([
            # Hidden Layer 1: Cukup besar untuk menangkap pola
            Dense(64, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(), # Menstabilkan learning
            Dropout(0.3),         # Mencegah overfitting (mematikan 30% neuron secara acak)
            
            # Hidden Layer 2: Mengerucut
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Output Layer: 1 Neuron untuk Binary Classification (0 s/d 1)
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(learning_rate=ModelConfig.LEARNING_RATE)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', # Wajib untuk binary classification
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.Precision(name='precision')
            ]
        )
        
        self.model.summary(print_fn=logger.info)

    def train(self) -> None:
        """
        Melatih model dengan Callbacks (EarlyStopping) dan Class Weights.
        """
        if self.model is None:
            raise ValueError("Model belum dibangun. Jalankan build_model() dulu.")
            
        logger.info("Memulai Training Model...")
        
        # Callbacks
        callbacks = [
            # Berhenti jika val_loss tidak membaik setelah 5 epoch (Patience)
            EarlyStopping(monitor='val_loss', patience=ModelConfig.PATIENCE, restore_best_weights=True),
            # Menyimpan model terbaik selama proses training (checkpointing)
            ModelCheckpoint(filepath=str(ModelConfig.MODEL_SAVE_PATH), monitor='val_loss', save_best_only=True)
        ]
        
        # Compute Class Weights
        # Meskipun sudah pakai SMOTE, class weights tetap berguna untuk kestabilan loss
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        logger.info(f"Class Weights digunakan: {class_weight_dict}")

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=ModelConfig.EPOCHS,
            batch_size=ModelConfig.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        self._plot_history(history)

    def _plot_history(self, history) -> None:
        """
        Helper function untuk visualisasi kurva Loss, Accuracy, dan AUC.
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        auc = history.history['auc']
        val_auc = history.history['val_auc']
        epochs_range = range(len(acc))

        plt.figure(figsize=(15, 5))
        
        # Plot Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        # Plot Loss
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        # Plot AUC
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, auc, label='Training AUC')
        plt.plot(epochs_range, val_auc, label='Validation AUC')
        plt.legend(loc='lower right')
        plt.title('Training and Validation AUC')
        
        plt.tight_layout()
        plt.savefig(ModelConfig.HISTORY_PLOT_PATH)
        plt.close()
        logger.info(f"Grafik training history tersimpan di: {ModelConfig.HISTORY_PLOT_PATH}")

    def evaluate(self) -> None:
        """
        Evaluasi performa model pada Test Set.
        """
        logger.info("Mengevaluasi model pada Test Set...")
        
        # Prediksi Probabilitas
        y_pred_probs = self.model.predict(self.X_test)
        # Thresholding (biasanya 0.5)
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        # Metrics
        print("\n" + "="*50)
        print("Laporan Klasifikasi (Test Set):")
        print(classification_report(self.y_test, y_pred))
        
        auc = roc_auc_score(self.y_test, y_pred_probs)
        print(f"ROC-AUC Score: {auc:.4f}")
        
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print("="*50 + "\n")

    def run(self):
        """
        Orkestrasi seluruh pipeline.
        """
        try:
            self.load_and_split_data()
            self.build_model()
            self.train()
            self.evaluate()
            logger.info("=== BASELINE MODELLING SELESAI ===")
        except Exception as e:
            logger.critical(f"Terjadi error fatal: {e}")
            raise

if __name__ == "__main__":
    trainer = ChurnBaselineTrainer()
    trainer.run()
