import sys
import os
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from model.data_preparation import load_data
from controller.data_controller import engineer_features, divide_dataframes
from controller.model_controller import predict_future
from model.model_training import train_model, evaluate_model

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
    """ 
    Main function to load data, train the model, and make predictions. 

    This function performs the following steps:
    1. Loads and preprocesses the data.
    2. Performs feature engineering.
    3. Trains and evaluates the machine learning model.
    4. Predicts future house prices.
    """
    print("Loading and preparing data...")
    
    # Definir los argumentos faltantes
    target_column = "saleprice"
    bins_labels = ([0, 100000, 200000, 300000, 450000, 760000], [0, 1, 2, 3, 4])

    # Llamar a divide_dataframes con los argumentos correctos
    dataframe_train, dataframe_validation_features, dataframe_test, train_target, validation_target, test_target, df_full_train = divide_dataframes(
        'data/train.csv', target_column, bins_labels
    )
    
    print("Performing feature engineering...")
    dataframe_train, dataframe_validation_features, dataframe_test = engineer_features(
        (dataframe_train, dataframe_validation_features, dataframe_test), df_full_train, 
        ['neighborhood', 'exterior2nd', 'housestyle']
    )
    
    print("Training and evaluating the model...")
    
    # Selected features for training
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

    # Extract selected features for training, validation, and testing
    X_train = dataframe_train[selected_columns]
    X_val = dataframe_validation_features[selected_columns]
    X_test = dataframe_test[selected_columns]
    
    # Train and evaluate the model
    model = train_model(X_train, train_target)
    evaluate_model(model, X_train, train_target, "Training")
    evaluate_model(model, X_val, validation_target, "Validation")
    evaluate_model(model, X_test, test_target, "Test")

    # Predict future house prices
    print("\nPredicting future prices...")
    future_predictions = predict_future(model, X_train, num_years=3)
    print(future_predictions[['yearbuilt', 'predicted_price']])


if __name__ == "__main__":
    main()
