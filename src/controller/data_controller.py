import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model.data_preparation import load_data, split_data, prepare_data
from model.feature_engineering import transform_target, group_by_mean_and_bin, encode_categorical_columns

# Add the parent directory to sys.path to allow importing modules from other directories
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def divide_dataframes(file_path: str, target_column: str, bins_labels: tuple):
    """
    Loads and splits the dataset into training, validation, and test sets.

    Args:
        file_path (str): Path to the dataset file.
        target_column (str): Name of the target column.
        bins_labels (tuple): Tuple containing bins and labels for grouping.

    Returns:
        tuple: DataFrames containing training, validation, and test sets, along with targets.
    """
    try:
        # Load dataset from the CSV file
        data = load_data(file_path)
        
        # Split dataset into training, validation, and test sets
        dataframe_train, dataframe_validation_features, dataframe_test, df_full_train = split_data(data)
        
        # Transform the target variable
        train_target, validation_target, test_target, dataframe_train, dataframe_validation_features, dataframe_test = transform_target(
            dataframe_train, dataframe_validation_features, dataframe_test, target_column
        )

        # Preprocess the datasets before training
        dataframe_train = prepare_data(dataframe_train)
        dataframe_validation_features = prepare_data(dataframe_validation_features)
        dataframe_test = prepare_data(dataframe_test)
        df_full_train = prepare_data(df_full_train)

        return dataframe_train, dataframe_validation_features, dataframe_test, train_target, validation_target, test_target, df_full_train
    except FileNotFoundError:
        raise Exception(f"File '{file_path}' not found.")
    except Exception as e:
        raise Exception(f"Error while splitting data: {str(e)}")

def engineer_features(dataframes: tuple, df_full_train: pd.DataFrame, categorical_columns: list):
    """
    Applies feature engineering techniques to enhance the dataset.

    Args:
        dataframes (tuple): Tuple containing training, validation, and test DataFrames.
        df_full_train (pd.DataFrame): Complete dataset for reference.
        categorical_columns (list): List of categorical columns to transform.

    Returns:
        tuple: Transformed DataFrames for training, validation, and testing.
    """
    try:
        dataframe_train, dataframe_validation_features, dataframe_test = dataframes

        # Define bins and labels for categorical feature grouping
        bins = [0, 100000, 200000, 300000, 450000, 760000]
        labels = [0, 1, 2, 3, 4]

        # Apply mean grouping and binning to categorical columns
        for column in categorical_columns:
            dataframe_train = group_by_mean_and_bin(dataframe_train, df_full_train, column, bins, labels)
            dataframe_validation_features = group_by_mean_and_bin(dataframe_validation_features, df_full_train, column, bins, labels)
            dataframe_test = group_by_mean_and_bin(dataframe_test, df_full_train, column, bins, labels)

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
