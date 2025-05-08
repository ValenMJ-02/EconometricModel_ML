import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model.predicted_prices import PredictedPrices
from src.controller.predicted_prices_controller import PredictedPricesController
from src.model.future_predictions import generate_future_data, predict_future_prices


class TestPredictedPrices(unittest.TestCase):
    
    def setUpClass():
        PredictedPricesController.dropTable()
        PredictedPricesController.createTable()
    
    def testInsertInto1(self, city: str):
        model = LinearRegression().fit(self.dataframe_train[['lotarea', 'grlivarea']], self.train_target)
        future_data = generate_future_data(2025, 3)
        future_predictions = predict_future_prices(model, future_data, ['lotarea', 'grlivarea'])

        predicted_price = PredictedPrices(city, str(future_predictions))

        PredictedPricesController.insertIntoTable(predicted_price)

        search_price = PredictedPricesController.queryCityPrices(predicted_price.city)

        self.assertTrue(search_price.isEqual(predicted_price))