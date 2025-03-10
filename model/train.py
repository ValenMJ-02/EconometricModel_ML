import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from google.cloud import storage
from .preprocess import load_data_from_gcs, clean_data, encode_categorical

def train_and_save_model(bucket_name: str, file_name: str, model_path: str) -> None:
    """Entrena un modelo de regresión lineal y lo guarda en Google Cloud Storage."""
    df = load_data_from_gcs(bucket_name, file_name)
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    
    features = ["List Year", "Assessed Value", "Sales Ratio", "Property Type", "Residential Type", "Town"]
    target = "Sale Amount"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    joblib.dump(model, "model.pkl")
    upload_model_to_gcs(bucket_name, "model.pkl", model_path)
    
    print("Modelo entrenado y guardado exitosamente en Google Cloud Storage.")


def upload_model_to_gcs(bucket_name: str, local_model_path: str, gcs_model_path: str) -> None:
    """Sube el modelo entrenado a Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_model_path)
    blob.upload_from_filename(local_model_path)
