import sys
import os
import pandas as pd
from sklearn.linear_model import LinearRegression  # Importar LinearRegression
from model.model_training import evaluate_model
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.model_training import train_model, evaluate_model
from model.future_predictions import generate_future_data, predict_future_prices


   
import pandas as pd

def predict_future(inputs, model):
    """
    Predice los precios de las casas para los próximos años.
    
    Args:
        inputs: Diccionario que contiene 'X_train' (pd.DataFrame) y 'num_years' (int).
        model: Modelo entrenado.
    
    Returns:
        pd.DataFrame: Predicciones futuras.
    """
    X_train = inputs["X_train"]
    num_years = inputs["num_years"]
    
    # Obtener el último año en los datos históricos
    last_year = X_train['yearbuilt'].max()
    
    # Generar datos futuros
    future_data = generate_future_data(last_year, num_years)
    
    # Predecir precios futuros
    feature_columns = X_train.columns.tolist()
    future_predictions = predict_future_prices(model, future_data, feature_columns)
    return future_predictions
