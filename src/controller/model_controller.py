import sys
import os
import pandas as pd
from sklearn.linear_model import LinearRegression  # Importar LinearRegression
from model.model_training import evaluate_model
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.model_training import train_model, evaluate_model
from model.future_predictions import generate_future_data, predict_future_prices


def train_and_evaluate_model(X_train: pd.DataFrame, train_target: pd.Series, X_val: pd.DataFrame,
                             validation_target: pd.Series, X_test: pd.DataFrame, test_target: pd.Series):
    """Entrena y evalúa el modelo."""
    model = LinearRegression().fit(X_train, train_target)
    evaluate_model(model, X_train, train_target, "Training")
    evaluate_model(model, X_val, validation_target, "Validation")
    evaluate_model(model, X_test, test_target, "Test")
    return model  # Devuelve el modelo entrenado
    
    
def predict_future(model, X_train: pd.DataFrame, num_years: int = 3) -> pd.DataFrame:
    """
    Predice los precios de las casas para los próximos años.
    
    Args:
        model: Modelo entrenado.
        X_train (pd.DataFrame): Datos de entrenamiento.
        num_years (int): Número de años futuros a predecir.
    
    Returns:
        pd.DataFrame: Predicciones futuras.
    """
    # Obtener el último año en los datos históricos
    last_year = X_train['yearbuilt'].max()
    
    # Generar datos futuros
    future_data = generate_future_data(last_year, num_years)
    
    # Predecir precios futuros
    feature_columns = X_train.columns.tolist()
    future_predictions = predict_future_prices(model, future_data, feature_columns)
    return future_predictions