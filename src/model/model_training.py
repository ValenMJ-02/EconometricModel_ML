"""
Module for training and evaluating the machine learning model.
"""

import os
import sys
from typing import Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Append parent directory for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def train_model(X_train: pd.DataFrame, train_target: Union[pd.Series, np.ndarray]) -> LinearRegression:
    """
    Trains a Linear Regression model using the provided training data.

    Args:
        X_train (pd.DataFrame): The feature matrix used for training.
        train_target (pd.Series or np.ndarray): The target variable for training.

    Returns:
        LinearRegression: The trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, train_target)
    return model

def evaluate_model(model: LinearRegression, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], dataset_name: str) -> None:
    """
    Evaluates the trained model and prints the R² score.

    Args:
        model (LinearRegression): The trained model.
        X (pd.DataFrame): Feature matrix of the dataset to evaluate.
        y (pd.Series or np.ndarray): True target values.
        dataset_name (str): Name of the dataset (e.g., "Training", "Validation", "Test").

    Returns:
        None
    """
    score: float = model.score(X, y)
    print(f"{dataset_name} set score: {score:.2f}")
