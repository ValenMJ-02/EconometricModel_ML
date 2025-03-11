import sys
import os
from sklearn.linear_model import LinearRegression  # Importar LinearRegression
from model.model_training import evaluate_model
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.model_training import train_model, evaluate_model
from model.future_predictions import generate_future_data, predict_future_prices


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrena y evalúa el modelo."""
    model = LinearRegression().fit(X_train, y_train)
    evaluate_model(model, X_train, y_train, "Training")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")
    return model  # Devuelve el modelo entrenado
    
def predict_future(model, X_train, num_years=3):
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