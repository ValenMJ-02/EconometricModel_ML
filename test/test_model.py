import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Importar las funciones a probar
from src.model.data_preparation import load_data, split_data, prepare_data
from src.model.feature_engineering import transform_target, group_by_mean_and_bin, encode_categorical_columns
from src.model.model_training import train_model, evaluate_model
from src.model.future_predictions import generate_future_data, predict_future_prices

class TestModel(unittest.TestCase):
    """Clase para pruebas unitarias del modelo."""

    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.file_path = "data/train.csv"
        self.data = load_data(self.file_path)
        self.df_train, self.df_val, self.df_test, self.df_full_train = split_data(self.data)
        self.y_train, self.y_val, self.y_test, self.df_train, self.df_val, self.df_test = transform_target(
            self.df_train, self.df_val, self.df_test, 'saleprice'
        )
        self.df_full_train = prepare_data(self.df_full_train)
        self.df_train = prepare_data(self.df_train)
        self.df_val = prepare_data(self.df_val)
        self.df_test = prepare_data(self.df_test)

    def test_load_data(self):
        """Prueba la función load_data."""
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertFalse(self.data.empty)

    def test_split_data(self):
        """Prueba la función split_data."""
        self.assertIsInstance(self.df_train, pd.DataFrame)
        self.assertIsInstance(self.df_val, pd.DataFrame)
        self.assertIsInstance(self.df_test, pd.DataFrame)
        self.assertEqual(len(self.df_train) + len(self.df_val) + len(self.df_test), len(self.data))

    def test_prepare_data(self):
        """Prueba la función prepare_data."""
        self.assertFalse(self.df_train.isnull().any().any())
        self.assertFalse(self.df_val.isnull().any().any())
        self.assertFalse(self.df_test.isnull().any().any())

    def test_transform_target(self):
        """Prueba la función transform_target."""
        self.assertIsInstance(self.y_train, np.ndarray)
        self.assertIsInstance(self.y_val, np.ndarray)
        self.assertIsInstance(self.y_test, np.ndarray)
        self.assertEqual(len(self.y_train), len(self.df_train))
        self.assertEqual(len(self.y_val), len(self.df_val))
        self.assertEqual(len(self.y_test), len(self.df_test))

    def test_group_by_mean_and_bin(self):
        """Prueba la función group_by_mean_and_bin."""
        bins = [0, 100000, 200000, 300000, 450000, 760000]
        labels = [0, 1, 2, 3, 4]
        df_train_grouped = group_by_mean_and_bin(self.df_train, self.df_full_train, 'neighborhood', bins, labels)
        self.assertIn('group_neighborhood', df_train_grouped.columns)

    def test_encode_categorical_columns(self):
        """Prueba la función encode_categorical_columns."""
        encoder = LabelEncoder()
        df_train_encoded = encode_categorical_columns(self.df_train, encoder)
        self.assertTrue(all(df_train_encoded[col].dtype == 'int64' for col in df_train_encoded.select_dtypes(include=['object']).columns))

    def test_train_model(self):
        """Prueba la función train_model."""
        model = train_model(self.df_train[['lotarea', 'grlivarea']], self.y_train)
        self.assertIsInstance(model, LinearRegression)

    def test_evaluate_model(self):
        """Prueba la función evaluate_model."""
        model = LinearRegression().fit(self.df_train[['lotarea', 'grlivarea']], self.y_train)
        evaluate_model(model, self.df_train[['lotarea', 'grlivarea']], self.y_train, "Training")

    def test_generate_future_data(self):
        """Prueba la función generate_future_data."""
        last_year = self.df_train['yearbuilt'].max()
        future_data = generate_future_data(last_year, 3)
        self.assertEqual(len(future_data), 3)
        self.assertEqual(future_data['yearbuilt'].iloc[0], last_year + 1)

    def test_predict_future_prices(self):
        """Prueba la función predict_future_prices."""
        model = LinearRegression().fit(self.df_train[['lotarea', 'grlivarea']], self.y_train)
        last_year = self.df_train['yearbuilt'].max()
        future_data = generate_future_data(last_year, 3)
        future_predictions = predict_future_prices(model, future_data, ['lotarea', 'grlivarea'])
        self.assertIn('predicted_price', future_predictions.columns)

if __name__ == '__main__':
    unittest.main()