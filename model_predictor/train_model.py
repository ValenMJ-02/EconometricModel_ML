"""
train_model.py
==============
Este script se encarga de:
  - Cargar y preprocesar el CSV de entrenamiento.
  - Entrenar el modelo usando la clase RealEstatePredictor.
  - Guardar el modelo entrenado en un archivo 'model.pkl' usando joblib.

Para ejecutarlo, asegúrate de que el CSV se encuentre en la ruta correcta (por ejemplo, en la carpeta data/).
"""

import joblib
from model import RealEstatePredictor

def train_and_save_model(csv_path: str, model_output_path: str):
    """
    Entrena el modelo y lo guarda en un archivo.

    Args:
        csv_path (str): Ruta al archivo CSV de entrenamiento.
        model_output_path (str): Ruta donde se guardará el modelo entrenado.
    """
    print("Cargando y entrenando el modelo. Esto puede tardar si el CSV es grande...")
    predictor = RealEstatePredictor(csv_path)
    print("Modelo entrenado correctamente.")
    
    # Guardar el objeto predictor (que incluye la pipeline entrenada y la información del preprocesamiento)
    joblib.dump(predictor, model_output_path)
    print(f"Modelo guardado en: {model_output_path}")

if __name__ == "__main__":
    # Ajusta estas rutas según tu estructura de proyecto.
    csv_path = "../data/Real_Estate_Sales_2001-2020_GL.csv"  # CSV de entrenamiento (125MB aprox.)
    model_output_path = "../model_predictor/model.pkl"
    train_and_save_model(csv_path, model_output_path)
