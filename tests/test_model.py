import unittest
import pandas as pd
from model_predictor.model import RealEstatePredictor

class TestRealEstatePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = {
            "Serial Number": [1, 2, 3, 4],
            "List Year": [2020, 2020, 2021, 2021],
            "Date Recorded": ["2020-01-01", "2020-06-01", "2021-03-01", "2021-07-01"],
            "Town": ["New Haven", "Hartford", "Stamford", "Chaplin"],
            "Address": ["Address1", "Address2", "Address3", "Address4"],
            "Assessed Value": [200000, 250000, 300000, 150000],
            "Sale Amount": [220000, 270000, 350000, 180000],
            "Sales Ratio": [1.1, 1.08, 1.095, 1.0769],
            "Property Type": ["Single Family", "Single Family", "Condo", "Residential"],
            "Residential Type": ["Single Family", "Single Family", "Condo", "Residential"],
            "Non Use Code": ["", "", "", ""],
            "Assessor Remarks": ["", "", "", ""],
            "OPM remarks": ["", "", "", ""],
            "Location": ["Loc1", "Loc2", "Loc3", "Loc4"]
        }
        cls.dummy_df = pd.DataFrame(data)
        cls.test_csv = "dummy_real_estate.csv"
        cls.dummy_df.to_csv(cls.test_csv, index=False)
        cls.predictor = RealEstatePredictor(cls.test_csv)

    def test_query_valid_city(self):
        city = "Hartford"
        property_type = "Single Family"
        price_range = (200000, 500000)
        year = 2026
        result = self.predictor.query_real_estate(year, city, property_type, price_range)
        self.assertTrue(all(price_range[0] <= price <= price_range[1] for price in result))

    def test_query_large_price_range(self):
        city = "Stamford"
        property_type = "Condo"
        price_range = (50000, 1000000)
        year = 2027
        result = self.predictor.query_real_estate(year, city, property_type, price_range)
        self.assertTrue(all(price_range[0] <= price <= price_range[1] for price in result))
        self.assertGreater(len(result), 0)

    def test_query_uncommon_city(self):
        city = "Chaplin"
        property_type = "Residential"
        price_range = (100000, 400000)
        year = 2026
        result = self.predictor.query_real_estate(year, city, property_type, price_range)
        if result:
            self.assertTrue(all(price_range[0] <= price <= price_range[1] for price in result))
        else:
            self.assertEqual(result, "No hay suficientes registros para esta ciudad.")


class TestRealEstatePredictorExceptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = {
            "Serial Number": [1, 2, 3, 4, 5, 6],
            "List Year": [2020, 2020, 2021, 2021, 2022, 2022],
            "Date Recorded": ["2020-01-01", "2020-06-01", "2021-03-01", "2021-07-01", "2022-05-01", "2022-08-01"],
            "Town": ["Bridgeport", "New Haven", "Bridgeport", "New Haven", "Tolland", "Tolland"],
            "Address": ["Address1", "Address2", "Address3", "Address4", "Address5", "Address6"],
            "Assessed Value": [250000, 5000000, 260000, 5500000, 270000, 280000],
            "Sale Amount": [270000, 6000000, 280000, 7000000, 15000, 18000],
            "Sales Ratio": [1.08, 1.2, 1.09, 1.25, 1.05, 1.03],
            "Property Type": ["Single Family", "Condo", "Single Family", "Condo", "Single Family", "Single Family"],
            "Residential Type": ["Single Family", "Condo", "Single Family", "Condo", "Single Family", "Single Family"],
            "Non Use Code": ["", "", "", "", "", ""],
            "Assessor Remarks": ["", "", "", "", "", ""],
            "OPM remarks": ["", "", "", "", "", ""],
            "Location": ["Loc1", "Loc2", "Loc3", "Loc4", "Loc5", "Loc6"]
        }
        cls.dummy_df = pd.DataFrame(data)
        cls.test_csv = "dummy_real_estate_exceptions.csv"
        cls.dummy_df.to_csv(cls.test_csv, index=False)
        cls.predictor = RealEstatePredictor(cls.test_csv)

    def test_query_without_price_range(self):
        city = "Bridgeport"
        property_type = "Single Family"
        year = 2025
        result = self.predictor.query_real_estate(year, city, property_type)
        self.assertGreater(len(result), 0)  # Debe haber al menos un resultado

    def test_query_high_price_range(self):
        city = "New Haven"
        property_type = "Condo"
        price_range = (5000000, 50000000)
        year = 2026
        result = self.predictor.query_real_estate(year, city, property_type, price_range)
        self.assertIn("advertencia", result.lower())  # Debe contener una advertencia si no hay datos

    def test_query_no_sales_in_price_range(self):
        city = "Tolland"
        property_type = "Single Family"
        price_range = (10000, 20000)
        year = 2026
        result = self.predictor.query_real_estate(year, city, property_type, price_range)
        self.assertEqual(result, "No hay registros en ese rango de precios.")


class TestRealEstatePredictorErrors(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        data = {
            "Serial Number": [1, 2, 3, 4, 5],
            "List Year": [2023, 2024, 2025, 2025, 2026],
            "Date Recorded": ["2023-02-01", "2024-05-01", "2025-07-01", "2025-09-01", "2026-12-01"],
            "Town": ["Bridgeport", "New Haven", "Hartford", "Stamford", "Bridgeport"],
            "Sale Amount": [250000, 450000, 500000, 300000, 400000],
            "Property Type": ["Single Family", "Condo", "Single Family", "Condo", "Multi Family"]
        }
        cls.dummy_df = pd.DataFrame(data)
        cls.test_csv = "dummy_real_estate_errors.csv"
        cls.dummy_df.to_csv(cls.test_csv, index=False)
        cls.predictor = RealEstatePredictor(cls.test_csv)

    def test_invalid_year(self):
        with self.assertRaises(ValueError) as context:
            self.predictor.query_real_estate(2030, "Medellín", "Single Family", (200000, 500000))
        self.assertIn("El año ingresado no es válido", str(context.exception))

    def test_invalid_city(self):
        with self.assertRaises(ValueError) as context:
            self.predictor.query_real_estate(2025, "Medellín", "Single Family", (200000, 500000))
        self.assertIn("La ciudad ingresada no está en la base de datos", str(context.exception))

    def test_invalid_price_range(self):
        with self.assertRaises(ValueError) as context:
            self.predictor.query_real_estate(2025, "Hartford", "Single Family", (500000, 200000))
        self.assertIn("El rango de precios ingresado no es válido", str(context.exception))

    def test_invalid_property_type(self):
        with self.assertRaises(ValueError) as context:
            self.predictor.query_real_estate(2025, "New Haven", "Luxury Villa", (100000, 500000))
        self.assertIn("El tipo de propiedad ingresado no es válido", str(context.exception))

    def test_non_numeric_price_range(self):
        with self.assertRaises(TypeError) as context:
            self.predictor.query_real_estate(2025, "Stamford", "Condo", ("abc", "xyz"))
        self.assertIn("Los valores del rango de precios deben ser numéricos", str(context.exception))


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)