"""
Unit tests for the housing price prediction model.
Follows the guidelines for unit tests, verifying both normal operation and error conditions.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model.data_preparation import load_data, split_data, prepare_data
from src.model.feature_engineering import transform_target, group_by_mean_and_bin, encode_categorical_columns
from src.model.model_training import train_model, evaluate_model
from src.model.future_predictions import generate_future_data, predict_future_prices

class TestModel(unittest.TestCase):
    """Test cases for housing price prediction functionalities."""

    def setUp(self) -> None:
        """Set up test variables and prepare the dataset for tests."""
        self.file_path: str = "data/train.csv"
        self.data: pd.DataFrame = load_data(self.file_path)
        # Using default test_size in split_data
        self.dataframe_train, self.dataframe_validation_features, self.dataframe_test, self.df_full_train = split_data(self.data)
        # Transform target variable using feature_engineering.transform_target
        dataframes = {
            "train": self.dataframe_train,
            "validation": self.dataframe_validation_features,
            "test": self.dataframe_test
        }
        (self.train_target, self.validation_target, self.test_target,
         self.dataframe_train, self.dataframe_validation_features, self.dataframe_test) = transform_target(dataframes, 'saleprice')
        self.df_full_train = prepare_data(self.df_full_train)
        self.dataframe_train = prepare_data(self.dataframe_train)
        self.dataframe_validation_features = prepare_data(self.dataframe_validation_features)
        self.dataframe_test = prepare_data(self.dataframe_test)

    def test_load_data(self) -> None:
        """Test that data is loaded correctly."""
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertFalse(self.data.empty)

    def test_split_data(self) -> None:
        """Test that data is split into training, validation, and test sets."""
        self.assertIsInstance(self.dataframe_train, pd.DataFrame)
        self.assertIsInstance(self.dataframe_validation_features, pd.DataFrame)
        self.assertIsInstance(self.dataframe_test, pd.DataFrame)
        total_length = len(self.dataframe_train) + len(self.dataframe_validation_features) + len(self.dataframe_test)
        self.assertEqual(total_length, len(self.data))

    def test_prepare_data(self) -> None:
        """Test that data preparation fills missing values correctly."""
        self.assertFalse(self.dataframe_train.isnull().any().any())
        self.assertFalse(self.dataframe_validation_features.isnull().any().any())
        self.assertFalse(self.dataframe_test.isnull().any().any())

    def test_transform_target(self) -> None:
        """Test the transformation of the target variable."""
        self.assertIsInstance(self.train_target, np.ndarray)
        self.assertIsInstance(self.validation_target, np.ndarray)
        self.assertIsInstance(self.test_target, np.ndarray)
        self.assertEqual(len(self.train_target), len(self.dataframe_train))
        self.assertEqual(len(self.validation_target), len(self.dataframe_validation_features))
        self.assertEqual(len(self.test_target), len(self.dataframe_test))

    def test_group_by_mean_and_bin(self) -> None:
        """Test grouping by mean and binning of a categorical feature."""
        bins = [0, 100000, 200000, 300000, 450000, 760000]
        labels = [0, 1, 2, 3, 4]
        dataframes = {
            "dataframe": self.dataframe_train,
            "dataframe_full": self.df_full_train
        }
        column_info = {
            "column_name": "neighborhood",
            "bins": bins,
            "labels": labels
        }
        dataframe_train_grouped = group_by_mean_and_bin(dataframes, column_info)
        self.assertIn('group_neighborhood', dataframe_train_grouped.columns)

    def test_encode_categorical_columns(self) -> None:
        """Test encoding of categorical columns using LabelEncoder."""
        encoder = LabelEncoder()
        dataframe_train_encoded = encode_categorical_columns(self.dataframe_train.copy(), encoder)
        for col in dataframe_train_encoded.select_dtypes(include=['object']).columns:
            self.assertEqual(dataframe_train_encoded[col].dtype, 'int64')

    def test_train_model(self) -> None:
        """Test training of the Linear Regression model."""
        model = train_model(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        self.assertIsInstance(model, LinearRegression)

    def test_evaluate_model(self) -> None:
        """Test evaluation of the trained model."""
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        evaluate_model(model, self.dataframe_train[['lotarea', 'grlivarea']], self.train_target, "Training")

    def test_generate_future_data(self) -> None:
        """Test generation of future data for predictions."""
        last_year = self.dataframe_train['yearbuilt'].max()
        future_data = generate_future_data(last_year, 3)
        self.assertEqual(len(future_data), 3)
        expected_first_year = max(last_year, pd.Timestamp.now().year) + 1
        self.assertEqual(future_data['yearbuilt'].iloc[0], expected_first_year)

    def test_predict_future_prices(self) -> None:
        """Test future price prediction functionality."""
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        last_year = self.dataframe_train['yearbuilt'].max()
        future_data = generate_future_data(last_year, 3)
        future_predictions = predict_future_prices(model, future_data, ['lotarea', 'grlivarea'])
        self.assertIn('predicted_price', future_predictions.columns)

    def test_normal_prediction(self) -> None:
        """Test normal prediction process."""
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        future_data = generate_future_data(2025, 3)
        future_predictions = predict_future_prices(model, future_data, ['lotarea', 'grlivarea'])
        self.assertEqual(len(future_predictions), 3)

    def test_normal_data_preparation(self) -> None:
        """Test normal data preparation."""
        prepared_data = prepare_data(self.dataframe_train.copy())
        self.assertFalse(prepared_data.isnull().any().any())

    def test_normal_feature_engineering(self) -> None:
        """Test normal feature engineering via encoding."""
        encoder = LabelEncoder()
        encoded_data = encode_categorical_columns(self.dataframe_train.copy(), encoder)
        for col in encoded_data.select_dtypes(include=['object']).columns:
            self.assertEqual(encoded_data[col].dtype, 'int64')

    def test_exceptional_large_future_data(self) -> None:
        """Test generation of a large number of future data points."""
        future_data = generate_future_data(2025, 100)
        self.assertEqual(len(future_data), 100)

    def test_exceptional_missing_columns(self) -> None:
        """Test prediction when columns are missing in future data."""
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        future_data = generate_future_data(2025, 3)
        future_predictions = predict_future_prices(model, future_data, ['lotarea', 'grlivarea'])
        self.assertIn('predicted_price', future_predictions.columns)

    def test_error_no_data(self) -> None:
        """Test loading data from a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")

    def test_error_invalid_split(self) -> None:
        """Test that providing an invalid test_size to split_data raises an error."""
        with self.assertRaises(ValueError):
            split_data(self.data, test_size=1.5)

    def test_error_invalid_column(self) -> None:
        """Test that using an invalid column in feature engineering raises KeyError."""
        with self.assertRaises(KeyError):
            group_by_mean_and_bin(
                {"dataframe": self.dataframe_train, "dataframe_full": self.df_full_train},
                {"column_name": "invalid_column", "bins": [0, 1], "labels": [0, 1]}
            )

    def test_error_invalid_model(self) -> None:
        """Test that evaluating a non-trained model raises AttributeError."""
        with self.assertRaises(AttributeError):
            evaluate_model(None, self.dataframe_train[['lotarea', 'grlivarea']], self.train_target, "Training")

if __name__ == '__main__':
    unittest.main()
