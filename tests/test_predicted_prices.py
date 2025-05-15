import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from psycopg2 import IntegrityError

import sys
import os
import json

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
        if search_price is not None:
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
        if search_price is not None:
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
            if sp is not None:
                self.assertEqual(sp.city, city)
                self.assertEqual(sp.prices, plist)

    def testInsertNullCityRaisesError(self):
        """Error al insertar sin ciudad (NULL) debe lanzar IntegrityError."""
        future_list = [{"yearbuilt": 2030, "predicted_price": 100.0}]
        future_json = json.dumps(future_list)
        predicted_price = PredictedPrices(None, future_json)
        with self.assertRaises(IntegrityError):
            PredictedPricesController.insertIntoTable(predicted_price)

    def testUpdateSingleCity(self):
        """Inserción inicial y luego UPDATE de una lista distinta para la misma ciudad."""
        city = "CityUpdateA"
        # insert inicial
        orig = PredictedPrices(city, json.dumps([{"yearbuilt": 2022, "predicted_price": 50.0}]))
        PredictedPricesController.insertIntoTable(orig)
        # nuevo conjunto de precios
        updated_list = [{"yearbuilt": 2023, "predicted_price": 75.0}]
        updated = PredictedPrices(city, json.dumps(updated_list))
        PredictedPricesController.updateCityPrices(updated)
        # comprobación
        result = PredictedPricesController.queryCityPrices(city)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertTrue(result.isEqual(updated))

    def testUpdateOneCityDoesNotAffectOthers(self):
        """UPDATE en CityB no debe modificar datos de CityC."""
        # inserción paralela
        a = PredictedPrices("CityB", json.dumps([{"yearbuilt": 2021, "predicted_price": 10.0}]))
        b = PredictedPrices("CityC", json.dumps([{"yearbuilt": 2021, "predicted_price": 20.0}]))
        PredictedPricesController.insertIntoTable(a)
        PredictedPricesController.insertIntoTable(b)
        # actualizamos solo CityB
        new_b = PredictedPrices("CityB", json.dumps([{"yearbuilt": 2022, "predicted_price": 15.0}]))
        PredictedPricesController.updateCityPrices(new_b)
        # verificamos
        rb = PredictedPricesController.queryCityPrices("CityB")
        rc = PredictedPricesController.queryCityPrices("CityC")
        if rb is not None and rc is not None:
            self.assertTrue(rb.isEqual(new_b))
            self.assertTrue(rc.isEqual(b))

    def testUpdateWithEmptyPrices(self):
        """UPDATE con lista vacía deja prices = [] en la tabla."""
        city = "CityEmpty"
        # insert inicial con algo
        initial = PredictedPrices(city, json.dumps([{"yearbuilt": 2020, "predicted_price": 5.0}]))
        PredictedPricesController.insertIntoTable(initial)
        # actualizamos a lista vacía
        empty = PredictedPrices(city, [])
        PredictedPricesController.updateCityPrices(empty)
        # comprobación
        resp = PredictedPricesController.queryCityPrices(city)
        self.assertIsNotNone(resp)
        if resp is not None:
            self.assertEqual(resp.prices, [])

    def testUpdateNullCityRaisesError(self):
        """Error al actualizar sin ciudad (NULL) debe lanzar IntegrityError."""
        future_list = [{"yearbuilt": 2040, "predicted_price": 100.0}]
        future_json = json.dumps(future_list)
        predicted_price = PredictedPrices(None, future_json)
        with self.assertRaises(IntegrityError):
            PredictedPricesController.updateCityPrices(predicted_price)

    def testDeleteExistingCity(self):
        """DELETE de un registro existente elimina correctamente."""
        city = "CityDelA"
        init = PredictedPrices(city, json.dumps([{"yearbuilt": 2025, "predicted_price": 99.9}]))
        PredictedPricesController.insertIntoTable(init)
        PredictedPricesController.deleteCityPrices(city)
        result = PredictedPricesController.queryCityPrices(city)
        self.assertIsNone(result)

    def testDeleteOneCityDoesNotAffectOthers(self):
        """DELETE en CityX no debe modificar CityY."""
        a = PredictedPrices("CityX", json.dumps([{"yearbuilt": 2026, "predicted_price": 10.0}]))
        b = PredictedPrices("CityY", json.dumps([{"yearbuilt": 2026, "predicted_price": 20.0}]))
        PredictedPricesController.insertIntoTable(a)
        PredictedPricesController.insertIntoTable(b)
        PredictedPricesController.deleteCityPrices("CityX")
        self.assertIsNone(PredictedPricesController.queryCityPrices("CityX"))
        remaining = PredictedPricesController.queryCityPrices("CityY")
        self.assertIsNotNone(remaining)
        if remaining is not None:
            self.assertTrue(remaining.isEqual(b))

    def testDeleteNonExistentCityNoError(self):
        """DELETE en ciudad no existente no lanza error."""
        # no debe lanzar excepción
        PredictedPricesController.deleteCityPrices("NoCity")
        # query devuelve None
        self.assertIsNone(PredictedPricesController.queryCityPrices("NoCity"))

    def testQueryEmptyTableReturnsNone(self):
        """SELECT en tabla vacía devuelve None."""
        # después de recreate
        PredictedPricesController.dropTable()
        PredictedPricesController.createTable()
        self.assertIsNone(PredictedPricesController.queryCityPrices("AnyCity"))

    def testQueryEmptyPricesField(self):
        """INSERT con prices=[] y SELECT devuelve objeto con lista vacía."""
        city = "CityEmptySel"
        initial = PredictedPrices(city, [])
        PredictedPricesController.insertIntoTable(initial)
        result = PredictedPricesController.queryCityPrices(city)
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(result.prices, [])

    def testQueryNullCityRaisesError(self):
        """SELECT con city=None debe lanzar IntegrityError."""
        with self.assertRaises(IntegrityError):
            PredictedPricesController.queryCityPrices(None)

