import pandas as pd
import joblib
from google.cloud import storage

def download_model_from_gcs(bucket_name: str, model_path: str, local_model_path: str) -> None:
    """Descarga el modelo desde Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.download_to_filename(local_model_path)

def load_model(local_model_path: str):
    """Carga el modelo desde un archivo local."""
    return joblib.load(local_model_path)

def make_prediction(model, input_data: pd.DataFrame) -> float:
    """Hace una predicción con el modelo entrenado."""
    return model.predict(input_data)[0]

# Ejemplo de uso
if __name__ == "__main__":
    bucket_name = "tu_bucket"
    model_path = "model.pkl"
    local_model_path = "model.pkl"
    
    download_model_from_gcs(bucket_name, model_path, local_model_path)
    model = load_model(local_model_path)
    
    sample_data = pd.DataFrame({
        "List Year": [2025],
        "Assessed Value": [250000],
        "Sales Ratio": [0.9],
        "Property Type": [1],
        "Residential Type": [2],
        "Town": [5]
    })
    
    predicted_price = make_prediction(model, sample_data)
    print(f"Precio estimado: ${predicted_price:.2f}")
