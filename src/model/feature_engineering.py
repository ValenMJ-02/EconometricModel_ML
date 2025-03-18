"""
Module for feature engineering tasks such as target transformation, grouping, and encoding.
"""

import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd

# Append parent directory for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def transform_target(dataframes: Dict[str, pd.DataFrame], target_column: str) -> tuple:
    """
    Transforms the target variable using log1p and removes it from the dataframes.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary containing 'train', 'validation', and 'test' DataFrames.
        target_column (str): The column name of the target variable.

    Returns:
        tuple: Contains transformed target arrays for train, validation, and test,
               and the updated DataFrames (train, validation, test) with the target column removed.
    """
    train_target: np.ndarray = np.log1p(dataframes["train"][target_column].values)
    validation_target: np.ndarray = np.log1p(dataframes["validation"][target_column].values)
    test_target: np.ndarray = np.log1p(dataframes["test"][target_column].values)
    
    dataframes["train"] = dataframes["train"].drop(columns=[target_column])
    dataframes["validation"] = dataframes["validation"].drop(columns=[target_column])
    dataframes["test"] = dataframes["test"].drop(columns=[target_column])
    
    return train_target, validation_target, test_target, dataframes["train"], dataframes["validation"], dataframes["test"]

def group_by_mean_and_bin(dataframes: Dict[str, pd.DataFrame], column_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Groups data by the mean 'saleprice' for a given categorical column and bins the groups.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary with keys 'dataframe' and 'dataframe_full'.
        column_info (Dict[str, Any]): Dictionary containing:
            - "column_name": Name of the categorical column.
            - "bins": List of bin edges.
            - "labels": Labels for the bins.

    Returns:
        pd.DataFrame: The updated DataFrame with a new column for the group.
    """
    dataframe: pd.DataFrame = dataframes["dataframe"]
    dataframe_full: pd.DataFrame = dataframes["dataframe_full"]
    column_name: str = column_info["column_name"]
    bins = column_info["bins"]
    labels = column_info["labels"]

    # Calculate mean saleprice per category
    mean_prices = dataframe_full.groupby(column_name)['saleprice'].mean()
    # Bin the mean prices into groups
    groups = pd.cut(mean_prices, bins=bins, labels=labels)
    
    grouped_dataframe: pd.DataFrame = pd.DataFrame({
        column_name: mean_prices.index,
        f'average_saleprice_{column_name}': mean_prices.values,
        f'group_{column_name}': groups
    }).reset_index(drop=True)
    
    # Merge the grouped data with the original dataframe
    dataframe = dataframe.merge(grouped_dataframe[[column_name, f'group_{column_name}']], on=column_name, how='left')
    return dataframe

def encode_categorical_columns(dataframe: pd.DataFrame, encoder: Any) -> pd.DataFrame:
    """
    Encodes categorical columns in the DataFrame using the provided encoder.

    Args:
        dataframe (pd.DataFrame): DataFrame containing categorical columns.
        encoder: Encoder instance (e.g., LabelEncoder) for transforming categorical values.

    Returns:
        pd.DataFrame: DataFrame with categorical columns encoded as integers.
    """
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        dataframe[column] = encoder.fit_transform(dataframe[column])
    return dataframe
