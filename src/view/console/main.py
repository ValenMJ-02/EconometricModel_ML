import sys
import os
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from model.data_preparation import load_data
from controller.data_controller import engineer_features, divide_dataframes
from controller.model_controller import train_and_evaluate_model, predict_future

def plot_future_predictions(future_predictions: pd.DataFrame): 
    """
    Muestra un gráfico de líneas con las predicciones futuras.
    
    Args:
        future_predictions (pd.DataFrame): Predicciones futuras.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(future_predictions['yearbuilt'], future_predictions['predicted_price'], marker='o')
    plt.title('Predicción de Precios de Casas en los Próximos Años')
    plt.xlabel('Año')
    plt.ylabel('Precio Predicho')
    plt.grid(True)
    plt.show()


def display_future_predictions(future_predictions: pd.DataFrame):
    """
    Muestra un gráfico de líneas con las predicciones futuras.
    
    Args:
        future_predictions (pd.DataFrame): Predicciones futuras.
    """
    print(tabulate(future_predictions[['yearbuilt', 'predicted_price']], headers='keys', tablefmt='pretty'))
    

def main():
    print("Cargando y preparando datos...")
    dataframe_train, dataframe_validation_features, dataframe_test, train_target, validation_target, test_target, df_full_train = divide_dataframes('data/train.csv')
    
    print("Realizando ingeniería de características...")
    dataframe_train, dataframe_validation_features, dataframe_test = engineer_features(dataframe_train, dataframe_validation_features, dataframe_test, df_full_train)
    
    print("Entrenando y evaluando el modelo...")
    selected_columns = [
        "lotarea", "grlivarea", "1stflrsf", "mssubclass", "overallcond",
        "bsmtunfsf", "garagearea", "yearbuilt", "overallqual", "bsmtfinsf1", "group_neighborhood",
        "group_exterior2nd", "fireplaces", "openporchsf", "2ndflrsf",
        "group_housestyle", "masvnrarea", "lotfrontage", "lotconfig", "yearremodadd", "screenporch",
        'mszoning', 'lotshape', 'landcontour', 'landslope', 'condition1', "bedroomabvgr",
        'roofstyle', 'roofmatl', 'exterior1st', 'exterqual', 'extercond', 'foundation',
        'bsmtqual', 'bsmtexposure', 'heatingqc', 'centralair', 'electrical',
        'functional', 'garagequal', 'paveddrive'
    ]
    X_train = dataframe_train[selected_columns]
    X_val = dataframe_validation_features[selected_columns]
    X_test = dataframe_test[selected_columns]
    
    # Entrenar y evaluar el modelo
    model = train_and_evaluate_model(X_train, train_target, X_val, validation_target, X_test, test_target)
    
    # Predecir precios futuros
    print("\nPrediciendo precios futuros...")
    future_predictions = predict_future(model, X_train, num_years=3)
    print(future_predictions[['yearbuilt', 'predicted_price']])

if __name__ == "__main__":
    main()