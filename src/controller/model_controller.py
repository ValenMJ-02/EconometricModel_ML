"""
Controller module for model-related operations such as future predictions.
"""

from typing import Dict, Any
import pandas as pd
from model.future_predictions import generate_future_data, predict_future_prices

def predict_future(inputs: Dict[str, Any], model: Any) -> pd.DataFrame:
    """
    Predicts future house prices based on training data and a trained model.

    Args:
        inputs (Dict[str, Any]): Dictionary containing:
            - "x_train": Training features DataFrame.
            - "num_years": Number of future years to predict.
        model: Trained machine learning model.

    Returns:
        pd.DataFrame: DataFrame with future predictions.
    """
    x_train = inputs["x_train"]
    num_years = inputs["num_years"]
    # Determine the base year from the training data
    last_year = x_train['yearbuilt'].max()
    # Generate future data
    future_data = generate_future_data(last_year, num_years)
    # Use all columns in x_train as features for prediction
    feature_columns = x_train.columns.tolist()
    future_predictions = predict_future_prices(model, future_data, feature_columns)
    return future_predictions
