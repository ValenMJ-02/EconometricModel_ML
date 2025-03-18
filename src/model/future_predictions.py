"""
Module for generating future data and predicting future house prices.
"""

from datetime import datetime
from typing import List, Any
import numpy as np
import pandas as pd

def generate_future_data(last_year: int, num_years: int) -> pd.DataFrame:
    """
    Generates future data based on the last year in historical data.

    Args:
        last_year (int): The last year in the historical data.
        num_years (int): Number of future years to generate.

    Returns:
        pd.DataFrame: DataFrame containing future years in 'yearbuilt' column.
    """
    current_year: int = datetime.now().year
    base_year: int = max(last_year, current_year)
    future_years: np.ndarray = np.arange(base_year + 1, base_year + num_years + 1)
    future_data: pd.DataFrame = pd.DataFrame({'yearbuilt': future_years})
    return future_data

def predict_future_prices(model: Any, future_data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Predicts future house prices using the trained model.

    Args:
        model: The trained machine learning model.
        future_data (pd.DataFrame): DataFrame containing future data.
        feature_columns (List[str]): List of feature columns used for prediction.

    Returns:
        pd.DataFrame: Future data with an added 'predicted_price' column.
    """
    # Ensure future_data has the necessary feature columns; fill missing ones with default 0
    for col in feature_columns:
        if col not in future_data.columns:
            future_data[col] = 0

    future_predictions: np.ndarray = model.predict(future_data[feature_columns])
    # Reverse the log1p transformation on predictions
    future_data['predicted_price'] = np.expm1(future_predictions)
    return future_data
