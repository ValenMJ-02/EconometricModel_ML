import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder  # Importar LabelEncoder

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
        self.dataframe_train, self.dataframe_validation_features, self.dataframe_test, self.df_full_train = split_data(self.data)
        self.train_target, self.validation_target, self.test_target, self.dataframe_train, self.dataframe_validation_features, self.dataframe_test = transform_target(
            self.dataframe_train, self.dataframe_validation_features, self.dataframe_test, 'saleprice'
        )
        self.df_full_train = prepare_data(self.df_full_train)
        self.dataframe_train = prepare_data(self.dataframe_train)
        self.dataframe_validation_features = prepare_data(self.dataframe_validation_features)
        self.dataframe_test = prepare_data(self.dataframe_test)

    def test_load_data(self):
        """Prueba la función load_data."""
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertFalse(self.data.empty)

    def test_split_data(self):
        """Prueba la función split_data."""
        self.assertIsInstance(self.dataframe_train, pd.DataFrame)
        self.assertIsInstance(self.dataframe_validation_features, pd.DataFrame)
        self.assertIsInstance(self.dataframe_test, pd.DataFrame)
        self.assertEqual(len(self.dataframe_train) + len(self.dataframe_validation_features) + len(self.dataframe_test), len(self.data))

    def test_prepare_data(self):
        """Prueba la función prepare_data."""
        self.assertFalse(self.dataframe_train.isnull().any().any())
        self.assertFalse(self.dataframe_validation_features.isnull().any().any())
        self.assertFalse(self.dataframe_test.isnull().any().any())

    def test_transform_target(self):
        """Prueba la función transform_target."""
        self.assertIsInstance(self.train_target, np.ndarray)
        self.assertIsInstance(self.validation_target, np.ndarray)
        self.assertIsInstance(self.test_target, np.ndarray)
        self.assertEqual(len(self.train_target), len(self.dataframe_train))
        self.assertEqual(len(self.validation_target), len(self.dataframe_validation_features))
        self.assertEqual(len(self.test_target), len(self.dataframe_test))

    def test_group_by_mean_and_bin(self):
        """Prueba la función group_by_mean_and_bin."""
        bins = [0, 100000, 200000, 300000, 450000, 760000]
        labels = [0, 1, 2, 3, 4]
        dataframe_train_grouped = group_by_mean_and_bin(self.dataframe_train, self.df_full_train, 'neighborhood', bins, labels)
        self.assertIn('group_neighborhood', dataframe_train_grouped.columns)

    def test_encode_categorical_columns(self):
        """Prueba la función encode_categorical_columns."""
        encoder = LabelEncoder()
        dataframe_train_encoded = encode_categorical_columns(self.dataframe_train, encoder)
        self.assertTrue(all(dataframe_train_encoded[col].dtype == 'int64' for col in dataframe_train_encoded.select_dtypes(include=['object']).columns))

    def test_train_model(self):
        """Prueba la función train_model."""
        model = train_model(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        self.assertIsInstance(model, LinearRegression)

    def test_evaluate_model(self):
        """Prueba la función evaluate_model."""
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        evaluate_model(model, self.dataframe_train[['lotarea', 'grlivarea']], self.train_target, "Training")

    def test_generate_future_data(self):
        """Prueba la función generate_future_data."""
        last_year = self.dataframe_train['yearbuilt'].max()
        future_data = generate_future_data(last_year, 3)
        self.assertEqual(len(future_data), 3)
        
        self.assertEqual(future_data['yearbuilt'].iloc[0], 2026)

    def test_predict_future_prices(self):
        """Prueba la función predict_future_prices."""
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        last_year = self.dataframe_train['yearbuilt'].max()
        future_data = generate_future_data(last_year, 3)
        future_predictions = predict_future_prices(model, future_data, ['lotarea', 'grlivarea'])
        self.assertIn('predicted_price', future_predictions.columns)

    # Pruebas normales
    def test_normal_prediction(self):
        """Prueba una predicción normal."""
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        future_data = generate_future_data(2025, 3)
        future_predictions = predict_future_prices(model, future_data, ['lotarea', 'grlivarea'])
        self.assertEqual(len(future_predictions), 3)

    def test_normal_data_preparation(self):
        """Prueba la preparación normal de datos."""
        prepared_data = prepare_data(self.dataframe_train)
        self.assertFalse(prepared_data.isnull().any().any())

    def test_normal_feature_engineering(self):
        """Prueba la ingeniería de características normal."""
        encoder = LabelEncoder()
        encoded_data = encode_categorical_columns(self.dataframe_train, encoder)
        self.assertTrue(all(encoded_data[col].dtype == 'int64' for col in encoded_data.select_dtypes(include=['object']).columns))

    # Pruebas excepcionales
    def test_exceptional_large_future_data(self):
        """Prueba la generación de una gran cantidad de datos futuros."""
        future_data = generate_future_data(2025, 100)
        self.assertEqual(len(future_data), 100)

    def test_exceptional_missing_columns(self):
        """Prueba la predicción con columnas faltantes."""
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        future_data = generate_future_data(2025, 3)
        future_predictions = predict_future_prices(model, future_data, ['lotarea', 'grlivarea'])
        self.assertIn('predicted_price', future_predictions.columns)

    def test_exceptional_high_values(self):
        """Prueba la predicción con valores extremadamente altos."""
        self.dataframe_train['lotarea'] = 1e6
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        future_data = generate_future_data(2025, 3)
        future_predictions = predict_future_prices(model, future_data, ['lotarea', 'grlivarea'])
        self.assertTrue((future_predictions['predicted_price'] > 1e6).all())

    # Pruebas de error
    def test_error_no_data(self):
        """Prueba la carga de datos sin archivo."""
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")

    def test_error_invalid_split(self):
        """Prueba la división de datos con tamaño de prueba inválido."""
        with self.assertRaises(ValueError):
            split_data(self.data, test_size=1.5)

    def test_error_invalid_column(self):
        """Prueba la ingeniería de características con columna inválida."""
        with self.assertRaises(KeyError):
            group_by_mean_and_bin(self.dataframe_train, self.df_full_train, 'invalid_column', [0, 1], [0, 1])

    def test_error_invalid_model(self):
        """Prueba la evaluación de un modelo no entrenado."""
        with self.assertRaises(AttributeError):
            evaluate_model(None, self.dataframe_train[['lotarea', 'grlivarea']], self.train_target, "Training")

if __name__ == '__main__':
    unittest.main()