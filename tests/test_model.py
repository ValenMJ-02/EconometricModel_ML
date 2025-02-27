import unittest
import pandas as pd
from model_predictor.model import RealEstatePredictor

class TestRealEstatePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Crear un DataFrame dummy para simular el CSV original
        data = {
            "Serial Number": [1, 2, 3, 4],
            "List Year": [2020, 2020, 2021, 2021],
            "Date Recorded": ["2020-01-01", "2020-06-01", "2021-03-01", "2021-07-01"],
            "Town": ["New Haven", "Hartford", "New Haven", "Hartford"],
            "Address": ["Address1", "Address2", "Address3", "Address4"],
            "Assessed Value": [200000, 250000, 210000, 260000],
            "Sale Amount": [220000, 270000, 230000, 280000],
            "Sales Ratio": [1.1, 1.08, 1.095, 1.0769],
            "Property Type": ["Single Family", "Single Family", "Condo", "Condo"],
            "Residential Type": ["Single Family", "Single Family", "Condo", "Condo"],
            "Non Use Code": ["", "", "", ""],
            "Assessor Remarks": ["", "", "", ""],
            "OPM remarks": ["", "", "", ""],
            "Location": ["Loc1", "Loc2", "Loc1", "Loc2"]
        }
        cls.dummy_df = pd.DataFrame(data)
        # Guardar el DataFrame dummy a CSV para ser utilizado en los tests
        cls.test_csv = "data/dummy_real_estate.csv"
        cls.dummy_df.to_csv(cls.test_csv, index=False)
        # Crear una instancia del predictor utilizando el CSV dummy
        cls.predictor = RealEstatePredictor(cls.test_csv)
        
    # ----- Casos de prueba normales (3) -----
    def test_normal_valid_input(self):
        # Entrada válida: "New Haven", "Single Family" y un rango de precios que se solapa.
        city = "New Haven"
        property_type = "Single Family"
        price_min = 210000
        price_max = 230000
        self.assertEqual(self.predictor.validate_city(city), city)
        self.assertEqual(self.predictor.validate_property_type(property_type), property_type)
        validated_range = self.predictor.validate_price_range(price_min, price_max)
        self.assertEqual(validated_range, (210000.0, 230000.0))
        
    def test_normal_future_prediction(self):
        # Verificar que se obtienen predicciones para 2025-2028.
        city = "Hartford"
        property_type = "Single Family"
        predictions = self.predictor.get_future_predictions(city, property_type)
        self.assertEqual(set(predictions.keys()), {2025, 2026, 2027, 2028})
        for year, pred in predictions.items():
            self.assertIsInstance(pred, float)
            
    def test_normal_get_median_assessed_value(self):
        # Verificar la mediana para Hartford, Single Family.
        city = "Hartford"
        property_type = "Single Family"
        median_value = self.predictor.get_median_assessed_value(city, property_type)
        self.assertEqual(median_value, 250000)
        
    # ----- Casos extraordinarios (3) -----
    def test_extraordinary_extreme_price_range(self):
        # Rango de precios muy amplio que se solapa con el histórico.
        city = "New Haven"
        property_type = "Condo"
        price_min = 100000   # muy bajo
        price_max = 1000000  # muy alto
        validated_range = self.predictor.validate_price_range(price_min, price_max)
        self.assertEqual(validated_range, (100000.0, 1000000.0))
        
    def test_extraordinary_city_with_whitespace(self):
        # Ciudad con espacios adicionales.
        city = "  Hartford  "
        validated_city = self.predictor.validate_city(city)
        self.assertEqual(validated_city, "Hartford")
        
    def test_extraordinary_property_type_with_whitespace(self):
        # Tipo de propiedad con espacios adicionales.
        property_type = "  Condo "
        validated_property = self.predictor.validate_property_type(property_type)
        self.assertEqual(validated_property, "Condo")
        
    # ----- Casos de error (4) -----
    def test_error_invalid_city(self):
        # Ciudad inexistente.
        with self.assertRaises(ValueError):
            self.predictor.validate_city("CiudadInexistente")
            
    def test_error_invalid_property_type(self):
        # Tipo de propiedad no permitido.
        with self.assertRaises(ValueError):
            self.predictor.validate_property_type("Villa")
            
    def test_error_invalid_price_range_order(self):
        # Rango de precios en orden incorrecto.
        with self.assertRaises(ValueError):
            self.predictor.validate_price_range(300000, 200000)
            
    def test_error_price_range_no_overlap(self):
        # Rango que no se solapa con el histórico.
        with self.assertRaises(ValueError):
            self.predictor.validate_price_range(100000, 150000)