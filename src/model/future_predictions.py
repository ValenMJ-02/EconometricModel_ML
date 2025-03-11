import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime

def generate_future_data(last_year, num_years):
    """
    Genera datos futuros para los próximos años.
    
    Args:
        last_year (int): El último año en los datos históricos.
        num_years (int): Número de años futuros a predecir.
    
    Returns:
        pd.DataFrame: DataFrame con años futuros.
    """
    # Obtener el año actual
    current_year = datetime.now().year
    
    # Si last_year es mayor que el año actual, usamos last_year como base
    base_year = max(last_year, current_year)
    
    # Generar años futuros
    future_years = np.arange(base_year + 1, base_year + num_years + 1)
    future_data = pd.DataFrame({'yearbuilt': future_years})
    return future_data

def predict_future_prices(model, future_data, feature_columns):
    """
    Predice los precios de las casas para los años futuros.
    
    Args:
        model: Modelo entrenado.
        future_data (pd.DataFrame): Datos futuros.
        feature_columns (list): Lista de columnas de características.
    
    Returns:
        pd.DataFrame: DataFrame con predicciones.
    """
    # Asegurarse de que future_data tenga las mismas columnas que los datos de entrenamiento
    for col in feature_columns:
        if col not in future_data.columns:
            future_data[col] = 0  # Rellenar con valores por defecto

    # Predecir
    future_predictions = model.predict(future_data[feature_columns])
    future_data['predicted_price'] = np.expm1(future_predictions)  # Revertir log1p
    return future_data