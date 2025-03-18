"""
Controller module for data-related operations such as loading, splitting, and feature engineering.
"""
from typing import Tuple, List, Any, Dict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model.data_preparation import load_data, split_data, prepare_data
from model.feature_engineering import transform_target, group_by_mean_and_bin, encode_categorical_columns


def divide_dataframes(file_path: str, target_column: str, bins_labels: Tuple[List[float], List[int]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                                                                                 pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """
    Loads the data, splits it into training, validation, and test sets, transforms the target variable,
    and prepares the data.

    Args:
        file_path (str): Path to the CSV data file.
        target_column (str): Column name of the target variable.
        bins_labels (Tuple[List[float], List[int]]): Tuple containing bins and labels for grouping.

    Returns:
        Tuple containing:
            - dataframe_train (pd.DataFrame): Training data.
            - dataframe_validation_features (pd.DataFrame): Validation data.
            - dataframe_test (pd.DataFrame): Test data.
            - train_target (pd.Series): Transformed target for training.
            - validation_target (pd.Series): Transformed target for validation.
            - test_target (pd.Series): Transformed target for testing.
            - full_training_dataset (pd.DataFrame): Full training set before splitting.
    """
    try:
        # Load dataset from the CSV file
        data = load_data(file_path)
        
        # Split dataset into training, validation, and test sets
        dataframe_train, dataframe_validation_features, dataframe_test, full_training_dataset = split_data(data)
        
        # Transform the target variable
        dataframes = {
            "train": dataframe_train,
            "validation": dataframe_validation_features,
            "test": dataframe_test
        }

        train_target, validation_target, test_target, dataframe_train, dataframe_validation_features, dataframe_test = transform_target(dataframes, target_column)


        # Preprocess the datasets before training
        dataframe_train = prepare_data(dataframe_train)
        dataframe_validation_features = prepare_data(dataframe_validation_features)
        dataframe_test = prepare_data(dataframe_test)
        full_training_dataset = prepare_data(full_training_dataset)

        return dataframe_train, dataframe_validation_features, dataframe_test, train_target, validation_target, test_target, full_training_dataset
    except FileNotFoundError:
        raise Exception(f"File '{file_path}' not found.")
    except Exception as e:
        raise Exception(f"Error while splitting data: {str(e)}")

def engineer_features(dataframes: tuple, full_training_dataset: pd.DataFrame, categorical_columns: list):
    """
    Applies feature engineering to the provided DataFrames by grouping and binning categorical columns.

    Args:
        dataframes_tuple (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): Tuple containing training, validation, and test DataFrames.
        full_training_dataset (pd.DataFrame): Full training dataset used as reference for grouping.
        categorical_columns (List[str]): List of categorical column names to engineer.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The updated training, validation, and test DataFrames.
    """
    try:
        dataframe_train, dataframe_validation_features, dataframe_test = dataframes

        # Define bins and labels for categorical feature grouping
        bins = [0, 100000, 200000, 300000, 450000, 760000]
        labels = [0, 1, 2, 3, 4]

        # Apply mean grouping and binning to categorical columns
        for column in categorical_columns:
            dataframes = {
                "dataframe_full": full_training_dataset
            }
            
            column_info = {
                "column_name": column,
                "bins": bins,
                "labels": labels
            }

            dataframes["dataframe"] = dataframe_train
            dataframe_train = group_by_mean_and_bin(dataframes, column_info)

            dataframes["dataframe"] = dataframe_validation_features
            dataframe_validation_features = group_by_mean_and_bin(dataframes, column_info)

            dataframes["dataframe"] = dataframe_test
            dataframe_test = group_by_mean_and_bin(dataframes, column_info)

        # Apply categorical encoding using LabelEncoder
        encoder = LabelEncoder()
        dataframe_train = encode_categorical_columns(dataframe_train, encoder)
        dataframe_validation_features = encode_categorical_columns(dataframe_validation_features, encoder)
        dataframe_test = encode_categorical_columns(dataframe_test, encoder)

        return dataframe_train, dataframe_validation_features, dataframe_test
    except KeyError as e:
        raise Exception(f"Feature engineering error: Missing column {str(e)}")
    except Exception as e:
        raise Exception(f"Feature engineering error: {str(e)}")
