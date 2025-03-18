"""
Main module to run the housing price prediction project.
This script loads data, performs feature engineering, trains the model,
evaluates performance, and predicts future house prices.
"""

import sys
import os
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# Append the project root to sys.path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from model.data_preparation import load_data
from controller.data_controller import engineer_features, divide_dataframes
from controller.model_controller import predict_future
from model.model_training import train_model, evaluate_model

def plot_future_predictions(future_predictions: pd.DataFrame) -> None:
    """
    Plot future house price predictions as a line graph.

    Args:
        future_predictions (pd.DataFrame): DataFrame containing future predictions with 'yearbuilt' and 'predicted_price' columns.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(future_predictions['yearbuilt'], future_predictions['predicted_price'], marker='o')
    plt.title('Predicción de Precios de Casas en los Próximos Años')
    plt.xlabel('Año')
    plt.ylabel('Precio Predicho')
    plt.grid(True)
    plt.show()

def display_future_predictions(future_predictions: pd.DataFrame) -> None:
    """
    Display future predictions in a tabulated format.

    Args:
        future_predictions (pd.DataFrame): DataFrame with predictions to display.
    """
    print(tabulate(future_predictions[['yearbuilt', 'predicted_price']], headers='keys', tablefmt='pretty'))

def main() -> None:
    """
    Main function to execute the housing price prediction pipeline.
    
    Steps:
    1. Load and prepare data.
    2. Perform feature engineering.
    3. Train and evaluate the machine learning model.
    4. Predict future house prices.
    """
    print("Loading and preparing data...")
    
    target_column: str = "saleprice"
    bins_labels = ([0, 100000, 200000, 300000, 450000, 760000], [0, 1, 2, 3, 4])

    # Divide dataframes and prepare target variables
    dataframe_train, dataframe_validation_features, dataframe_test, train_target, validation_target, test_target, df_full_train = divide_dataframes(
        'data/train.csv', target_column, bins_labels
    )
    
    print("Performing feature engineering...")
    dataframe_train, dataframe_validation_features, dataframe_test = engineer_features(
        (dataframe_train, dataframe_validation_features, dataframe_test), df_full_train, 
        ['neighborhood', 'exterior2nd', 'housestyle']
    )
    
    print("Training and evaluating the model...")
    # Define selected features for training
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
    
    X_train: pd.DataFrame = dataframe_train[selected_columns]
    X_val: pd.DataFrame = dataframe_validation_features[selected_columns]
    X_test: pd.DataFrame = dataframe_test[selected_columns]
    
    # Train model and evaluate on training, validation, and test sets
    model = train_model(X_train, train_target)
    evaluate_model(model, X_train, train_target, "Training")
    evaluate_model(model, X_val, validation_target, "Validation")
    evaluate_model(model, X_test, test_target, "Test")

    # Prepare input data for future prediction
    inputs = {
        "X_train": X_train,
        "num_years": 3
    }

    print("\nPredicting future prices...")
    future_predictions: pd.DataFrame = predict_future(inputs, model)
    print(future_predictions[['yearbuilt', 'predicted_price']])
    # Optionally, plot the predictions
    plot_future_predictions(future_predictions)

if __name__ == "__main__":
    main()
