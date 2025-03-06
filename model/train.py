import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Obtener la ruta absoluta del directorio raíz del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model.storage import load_csv_from_gcs, save_model_to_gcs
from model.preprocess import preprocess_data



DATA_FILE_PATH = "data/Real_Estate_Sales_2001-2020_GL.csv"
MODEL_FILENAME = "real_estate_model.pkl"

def train_model():
    """
    Entrena un modelo de RandomForestRegressor y lo guarda en Google Cloud Storage.
    """
    df = load_csv_from_gcs(DATA_FILE_PATH)
    df, _ = preprocess_data(df)
    
    features = ["List Year", "Assessed Value", "Sales Ratio", "Property Type", "Residential Type", "Town"]
    target = "Sale Amount"

    df = df.dropna(subset=[target])
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    save_model_to_gcs(model, MODEL_FILENAME)

    y_pred = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
