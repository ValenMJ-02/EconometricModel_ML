import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import sys
import os
import json  # Importar la librería JSON

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model.predicted_prices import PredictedPrices
from src.controller.predicted_prices_controller import PredictedPricesController
from src.model.future_predictions import generate_future_data, predict_future_prices


class TestPredictedPrices(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        PredictedPricesController.dropTable()
        PredictedPricesController.createTable()

    def testInsertInto1(self):
        import pandas as pd, numpy as np, json, os
        from sklearn.linear_model import LinearRegression
        from src.model.predicted_prices import PredictedPrices
        from src.controller.predicted_prices_controller import PredictedPricesController
        from src.model.future_predictions import generate_future_data, predict_future_prices

        # 1) Carga el CSV real
        data_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
        )
        df = pd.read_csv(data_file)

        # 2) Renombra columnas al formato esperado por tu pipeline
        df = df.rename(columns={
            "LotArea":    "lotarea",
            "GrLivArea":  "grlivarea",
            "SalePrice":  "saleprice",
            "YearBuilt":  "yearbuilt"
        })

        # 3) Entrena el modelo con las dos features reales
        X = df[["lotarea", "grlivarea"]]
        # entrenamos sobre log1p(saleprice) para que expm1() inverso funcione
        y = np.log1p(df["saleprice"])
        model = LinearRegression().fit(X, y)

        # 4) Genera datos futuros (ejemplo: 5 años)
        last_year = int(df["yearbuilt"].max())
        future_data = generate_future_data(last_year, 5)
        # Para poder predecir, usamos la media histórica de las features
        future_data["lotarea"]   = X["lotarea"].mean()
        future_data["grlivarea"] = X["grlivarea"].mean()

        # 5) Obtiene el DataFrame con predicciones
        future_predictions_df = predict_future_prices(
            model, future_data, ["lotarea", "grlivarea"]
        )

        # 6) Limpiar inf/nan antes de JSON
        future_predictions_df["predicted_price"] = future_predictions_df[
            "predicted_price"
        ].replace([np.inf, -np.inf, np.nan], None)

        # 7) Creamos la lista de registros que vamos a guardar
        future_list = future_predictions_df[
            ["yearbuilt", "predicted_price"]
        ].to_dict(orient="records")
        future_json = json.dumps(future_list)

        # 8) Insertar en BD y consultar
        city = "Manchester"
        predicted_price = PredictedPrices(city, future_json)
        PredictedPricesController.insertIntoTable(predicted_price)
        search_price = PredictedPricesController.queryCityPrices(city)

        # 9) Comprobaciones
        self.assertIsNotNone(search_price, "No se recuperó ningún registro")
        self.assertEqual(search_price.city, city)
        # isEqual compara lista de dicts ordenada y JSON interno
        self.assertTrue(
            search_price.isEqual(predicted_price),
            f"Los precios insertados {predicted_price.prices} "
            f"no coinciden con los recuperados {search_price.prices}"
        )

    def testInsertManualList(self):
        """Inserta manualmente una lista pequeña de precios y la recupera."""
        city = "TestManual"
        # Lista de años y precios de ejemplo
        future_list = [
            {"yearbuilt": 2022, "predicted_price": 123.45},
            {"yearbuilt": 2023, "predicted_price": 234.56}
        ]
        future_json = json.dumps(future_list)

        predicted_price = PredictedPrices(city, future_json)
        PredictedPricesController.insertIntoTable(predicted_price)

        search_price = PredictedPricesController.queryCityPrices(city)
        self.assertIsNotNone(search_price)
        self.assertEqual(search_price.city, city)
        self.assertTrue(
            search_price.isEqual(predicted_price),
            f"Esperaba {predicted_price.prices} y obtuve {search_price.prices}"
        )

    def testInsertMultipleCities(self):
        """Inserta varios registros en distintas ciudades y los consulta."""
        entries = {
            "CityA": [{"yearbuilt": 2024, "predicted_price": 10.0}],
            "CityB": [{"yearbuilt": 2025, "predicted_price": 20.0}]
        }
        # Inserción
        for city, plist in entries.items():
            jp = json.dumps(plist)
            PredictedPricesController.insertIntoTable(PredictedPrices(city, jp))

        # Consulta y comprobación
        for city, plist in entries.items():
            sp = PredictedPricesController.queryCityPrices(city)
            self.assertIsNotNone(sp, f"No recuperó datos para {city}")
            self.assertEqual(sp.city, city)
            self.assertEqual(sp.prices, plist)