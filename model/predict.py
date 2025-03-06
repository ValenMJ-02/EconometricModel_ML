import sys
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Ahora podemos importar los módulos sin depender de la estructura de paquete
from model.storage import load_model_from_gcs
from model.preprocess import preprocess_data, encode_categorical_columns
from model.storage import load_csv_from_gcs


DATA_FILE_PATH = "data/Real_Estate_Sales_2001-2020_GL.csv"
MODEL_FILENAME = "real_estate_model.pkl"

def predict_price(town_name: str):
    """
    Predice el precio de las propiedades para los años 2025-2027 en una ciudad específica.
    """
    model = load_model_from_gcs(MODEL_FILENAME)
    df = load_csv_from_gcs(DATA_FILE_PATH)
    df, encoders = preprocess_data(df)

    if town_name not in encoders["Town"].classes_:
        print("Error: La ciudad ingresada no existe en los datos.")
        return
    
    town_encoded = encoders["Town"].transform([town_name])[0]
    future_years = [2025, 2026, 2027]
    estimates = {}

    for year in future_years:
        X_future = df[df["Town"] == town_encoded].copy()
        X_future["List Year"] = year
        predicted_prices = model.predict(X_future[["List Year", "Assessed Value", "Sales Ratio", "Property Type", "Residential Type", "Town"]])
        estimates[year] = np.mean(predicted_prices)

    return estimates
