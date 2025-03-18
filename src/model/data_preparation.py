"""
Module for data loading, splitting, and preparation.
"""

import os
import sys
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Append parent directory for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# Import custom exceptions
from model.exceptions import InvalidSplitSizeError

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and formats the column names.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset with formatted column names.
    """
    data: pd.DataFrame = pd.read_csv(file_path)
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
    return data

def split_data(data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training, validation, and test sets.
    
    The process is as follows:
    1. Split the data into a full training set (df_full_train) and test set (dataframe_test) using test_size.
    2. Split df_full_train into training (dataframe_train) and validation (dataframe_validation_features) sets.

    Args:
        data (pd.DataFrame): The complete dataset.
        test_size (float, optional): Proportion of the data to include in the test split. Must be between 0 and 1. Defaults to 0.2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            A tuple containing the training set, validation set, test set, and the full training set before the final split.

    Raises:
        InvalidSplitSizeError: If test_size is not between 0 and 1.
    """
    if not (0 < test_size < 1):
        raise InvalidSplitSizeError(test_size)
    
    df_full_train, dataframe_test = train_test_split(data, test_size=test_size, random_state=1)
    dataframe_train, dataframe_validation_features = train_test_split(df_full_train, test_size=0.25, random_state=1)
    
    # Reset indices for consistency
    dataframe_train = dataframe_train.reset_index(drop=True)
    dataframe_validation_features = dataframe_validation_features.reset_index(drop=True)
    dataframe_test = dataframe_test.reset_index(drop=True)
    
    return dataframe_train, dataframe_validation_features, dataframe_test, df_full_train

def prepare_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataset by handling missing values.

    Specific operations:
    - Replace missing values in 'fireplacequ' with 'NA'.
    - Replace missing values in 'garageyrblt' with 0.
    - For other columns:
      - If categorical, fill missing with the mode.
      - If numerical, fill missing with the mean.

    Args:
        dataframe (pd.DataFrame): The dataset to be processed.

    Returns:
        pd.DataFrame: The processed dataset with missing values filled.
    """
    if 'fireplacequ' in dataframe.columns:
        dataframe['fireplacequ'] = dataframe['fireplacequ'].fillna('NA')
    if 'garageyrblt' in dataframe.columns:
        dataframe['garageyrblt'] = dataframe['garageyrblt'].fillna(0)
    
    for col in dataframe.columns:
        if col not in ['fireplacequ', 'garageyrblt']:
            if dataframe[col].dtype == 'object' or dataframe[col].dtype.name == 'category':
                if not dataframe[col].isnull().all():
                    dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
                else:
                    dataframe[col] = dataframe[col].fillna('NA')
            else:
                dataframe[col] = dataframe[col].fillna(dataframe[col].astype(float).mean())
    
    return dataframe
