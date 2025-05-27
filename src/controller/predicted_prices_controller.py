import sys
import psycopg2
import json
from typing import Optional
sys.path.append("src")

from model.predicted_prices import PredictedPrices

import secret_config

class PredictedPricesController:
    @classmethod
    def dropTable(cls):
        query = "DROP TABLE if exists predicted_prices"

        cursor = cls.getCursor()

        cursor.execute(query)

        cursor.connection.commit()

    @classmethod
    def createTable(cls):
        cursor = cls.getCursor()

        with open("./sql/create_table.sql", "r") as query_file:
            query = query_file.read()
            cursor.execute(query)

            cursor.connection.commit()

    @classmethod
    def insertIntoTable(cls, predicted_price: PredictedPrices):
        cursor = cls.getCursor()

        # serializamos la lista de precios a JSON
        predicted_price.city = predicted_price.city.strip().lower()
        prices_json = json.dumps(predicted_price.prices)

        query = "INSERT INTO predicted_prices (city, prices) VALUES (%s, %s);"
        cursor.execute(query, (predicted_price.city, prices_json))

        cursor.connection.commit()

    @classmethod
    def queryCityPrices(cls, city: Optional[str]) -> Optional[PredictedPrices]:
        if city is None:
            raise psycopg2.IntegrityError("city must not be null")
        city = city.strip().lower()
        cursor = cls.getCursor()
        query = "SELECT id, predicted_at, city, prices FROM predicted_prices WHERE city = %s;"
        cursor.execute(query, (city,))
        row = cursor.fetchone()
         
        if row:
            # row[2]=city, row[3]=prices (JSON o string)
            retrieved_city = row[2]
            retrieved_prices = row[3]
            # Pasamos los dos argumentos en orden, no como keyword
            return PredictedPrices(retrieved_city, retrieved_prices)
         
        return None

    @classmethod
    def updateCityPrices(cls, predicted_price: PredictedPrices):
        """
        UPDATE the prices JSON for a given city.
        """
        if predicted_price.city is None:
            raise psycopg2.IntegrityError("city must not be null")
        predicted_price.city = predicted_price.city.strip().lower()
        cursor = cls.getCursor()
        prices_json = json.dumps(predicted_price.prices)
        query = """
            UPDATE predicted_prices
               SET prices = %s
             WHERE city = %s;
        """
        cursor.execute(query, (prices_json, predicted_price.city))
        cursor.connection.commit()

    @classmethod
    def deleteCityPrices(cls, city: str):
        """
        DELETE record by city.
        """
        cursor = cls.getCursor()
        city = city.strip().lower()
        query = "DELETE FROM predicted_prices WHERE city = %s;"
        cursor.execute(query, (city,))
        cursor.connection.commit()

    @classmethod
    def getCursor(cls):
        connection = psycopg2.connect(database=secret_config.PGDATABASE, user=secret_config.PGUSER, password=secret_config.PGPASSWORD, host=secret_config.PGHOST)

        cursor = connection.cursor()

        return cursor
