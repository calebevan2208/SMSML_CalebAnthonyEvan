"""
automate_CalebAnthony.py
------------------------
Modul ini bertanggung jawab untuk proses Ingestion (Data Retrieval) 
dan Preprocessing (Data Cleaning) untuk dataset Churn Prediction.

Author: Caleb Anthony (Automated by System)
Date: 2025-10-30
Version: 1.2 (Professional Edition)
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import requests
from io import StringIO
from pathlib import Path
from typing import Optional, Dict

# --- KONFIGURASI LOGGING ---
# Mengatur format log standar industri: Waktu - Level - Pesan
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- KONFIGURASI PROYEK & PATH ---
class ProjectConfig:
    """
    Kelas konfigurasi statis untuk menyimpan path file dan URL.
    Menggunakan Pathlib untuk kompatibilitas lintas OS (Windows/Linux/Mac).
    """
    BASE_DIR = Path(__file__).resolve().parent
    RAW_DIR = BASE_DIR / 'churn_raw'
    OUTPUT_DIR = BASE_DIR / 'churn_preprocessing'
    # Define secondary output path for Membangun_model
    MODEL_OUTPUT_DIR = BASE_DIR.parent / 'Membangun_model' / 'churn_preprocessing'
    
    RAW_FILE_PATH = RAW_DIR / 'data.csv'
    CLEAN_FILE_PATH = OUTPUT_DIR / 'clean_data.csv'
    MODEL_FILE_PATH = MODEL_OUTPUT_DIR / 'clean_data.csv'
    
    # URL Dataset Sumber
    DATA_SOURCE_URL = "https://docs.google.com/spreadsheets/d/1tovcDh4h56V03CA5VaCSsVKqZb2QGjZ3/export?format=csv"

# --- DEFINISI MAPPING KATEGORIKAL ---
# Dipisahkan di sini agar mudah diaudit dan diubah (Maintainability)
MAPPING_CONFIG = {
    'SEX': {
        'M': 1, 
        'F': 2
    },
    'EDUCATION': {
        'Graduate school': 1,
        'University': 2,
        'High School': 3,
        'Unknown': 4,
        'Others': 4,
        '0': 4  # Menangani noise '0' sebagai kategori 'Others'
    },
    'MARRIAGE': {
        'Married': 1,
        'Single': 2,
        'Other': 3,
        '0': 3  # Menangani noise '0' sebagai kategori 'Other'
    },
    'default': {
        'Y': 1,
        'N': 0
    }
}

class ChurnDataPipeline:
    """
    Kelas utama untuk menangani pipeline preprocessing data churn.
    """
    
    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None

    def ingest_data(self) -> None:
        """
        Mengambil data dari sumber.
        Logika: Cek file lokal terlebih dahulu. Jika tidak ada, unduh dari URL.
        """
        logger.info("Memulai proses Ingestion Data...")
        
        # Pastikan direktori raw ada
        ProjectConfig.RAW_DIR.mkdir(parents=True, exist_ok=True)

        if ProjectConfig.RAW_FILE_PATH.exists():
            logger.info(f"File raw ditemukan di lokal: {ProjectConfig.RAW_FILE_PATH}")
            try:
                self.raw_df = pd.read_csv(ProjectConfig.RAW_FILE_PATH)
            except Exception as e:
                logger.error(f"Gagal membaca file lokal: {e}")
                raise
        else:
            logger.warning("File lokal tidak ditemukan. Mencoba mengunduh dari remote source...")
            try:
                response = requests.get(ProjectConfig.DATA_SOURCE_URL, timeout=30)
                response.raise_for_status() # Raise error jika status code bukan 200
                
                # Membaca konten CSV
                self.raw_df = pd.read_csv(StringIO(response.text))
                
                # Menyimpan arsip raw
                self.raw_df.to_csv(ProjectConfig.RAW_FILE_PATH, index=False)
                logger.info(f"Unduhan berhasil. Raw data disimpan di: {ProjectConfig.RAW_FILE_PATH}")
                
            except requests.exceptions.RequestException as e:
                logger.critical(f"Gagal mengunduh data dari URL: {e}")
                raise

    def preprocess_data(self) -> None:
        """
        Melakukan pembersihan data, standarisasi kolom, dan encoding variabel kategorikal.
        """
        if self.raw_df is None:
            raise ValueError("Dataframe kosong. Jalankan ingest_data() terlebih dahulu.")
        
        logger.info("Memulai proses Preprocessing Data...")
        df = self.raw_df.copy()

        # 1. Standardize Column Names
        logger.info("Menstandarisasi nama kolom...")
        df.columns = df.columns.str.strip()
        
        # Rename PAY_0 -> PAY_1 sesuai standar dataset UCI Credit Card
        if 'PAY_0' in df.columns:
            df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
            logger.info("Kolom 'PAY_0' diubah menjadi 'PAY_1'.")

        # 2. Drop Irrelevant Columns
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)
            logger.info("Kolom 'ID' dihapus karena tidak relevan untuk pemodelan.")

        # 3. Categorical Encoding (Map & Clean)
        # Iterasi melalui konfigurasi mapping yang sudah didefinisikan
        for col, mapping_dict in MAPPING_CONFIG.items():
            if col in df.columns:
                logger.info(f"Memproses encoding kolom: {col}")
                
                # Konversi ke string dulu untuk memastikan '0' (int) dan '0' (str) tertangani
                df[col] = df[col].astype(str)
                
                # Cek nilai unik sebelum mapping untuk keperluan audit log
                unique_vals = df[col].unique()
                logger.debug(f"Nilai unik di {col} sebelum mapping: {unique_vals}")
                
                # Lakukan mapping
                df[col] = df[col].map(mapping_dict)
                
                # Validasi: Cek apakah ada nilai yang menjadi NaN (artinya tidak ada di dictionary mapping)
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Terdapat {nan_count} nilai NaN di kolom {col} setelah mapping. Mengisi dengan nilai default mayoritas.")
                    # Mengisi NaN dengan modus (nilai yang paling sering muncul)
                    mode_val = int(df[col].mode()[0])
                    df[col] = df[col].fillna(mode_val)
                
                # Pastikan tipe data menjadi integer setelah mapping
                df[col] = df[col].astype(int)

        # 4. Final Type Conversion
        # Pastikan seluruh kolom numerik
        logger.info("Memastikan konsistensi tipe data numerik...")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Hapus baris jika ada yang gagal dikonversi menjadi angka (sangat jarang terjadi jika mapping benar)
        initial_len = len(df)
        df.dropna(inplace=True)
        if len(df) < initial_len:
            logger.warning(f"Dihapus {initial_len - len(df)} baris yang mengandung nilai Null tak terduga.")

        self.clean_df = df
        logger.info(f"Preprocessing selesai. Dimensi data akhir: {self.clean_df.shape}")

    def save_data(self) -> None:
        """
        Menyimpan hasil data bersih ke direktori output.
        """
        if self.clean_df is None:
            logger.error("Tidak ada data bersih untuk disimpan.")
            return

        # Pastikan folder output ada
        ProjectConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            self.clean_df.to_csv(ProjectConfig.CLEAN_FILE_PATH, index=False)
            logger.info(f"Data bersih berhasil disimpan di: {ProjectConfig.CLEAN_FILE_PATH}")
            
            # Copy to Membangun_model directory
            ProjectConfig.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            self.clean_df.to_csv(ProjectConfig.MODEL_FILE_PATH, index=False)
            logger.info(f"Salinan data bersih disimpan di: {ProjectConfig.MODEL_FILE_PATH}")
            
        except Exception as e:
            logger.error(f"Gagal menyimpan file CSV: {e}")
            raise

    def run(self):
        """
        Metode orkestrasi utama untuk menjalankan seluruh pipeline.
        """
        logger.info("=== START PIPELINE AUTOMATION ===")
        try:
            self.ingest_data()
            self.preprocess_data()
            self.save_data()
            logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            
            # Tampilkan sampel data untuk verifikasi cepat di terminal
            if self.clean_df is not None:
                print("\n[Preview Data Output]")
                print(self.clean_df.head().to_string())
                print("\n[Data Info]")
                self.clean_df.info()

        except Exception as e:
            logger.critical(f"Pipeline berhenti karena error fatal: {e}")
            sys.exit(1)

if __name__ == "__main__":
    # Instansiasi dan jalankan pipeline
    pipeline = ChurnDataPipeline()
    pipeline.run()
