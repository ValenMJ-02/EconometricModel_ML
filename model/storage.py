import os
from google.cloud import storage
import pandas as pd
import joblib

BUCKET_NAME = "tu_bucket_name"  # Cambia esto por el nombre de tu bucket

def load_csv_from_gcs(file_path: str) -> pd.DataFrame:
    """
    Carga un archivo CSV desde Google Cloud Storage y lo devuelve como un DataFrame.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    
    temp_file = "/tmp/temp_data.csv"
    blob.download_to_filename(temp_file)
    
    return pd.read_csv(temp_file)

def save_model_to_gcs(model, model_filename: str):
    """
    Guarda un modelo entrenado en Google Cloud Storage.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")

    temp_file = "/tmp/model.pkl"
    joblib.dump(model, temp_file)
    blob.upload_from_filename(temp_file)

def load_model_from_gcs(model_filename: str):
    """
    Carga un modelo entrenado desde Google Cloud Storage.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")

    temp_file = "/tmp/model.pkl"
    blob.download_to_filename(temp_file)
    
    return joblib.load(temp_file)
