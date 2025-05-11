import sys
import psycopg2
import json
sys.path.append("src")

from model.predicted_prices import PredictedPrices

import secret_config

class PredictedPricesController:

    def dropTable():
        query = "DROP TABLE if exists predicted_prices"

        cursor = PredictedPricesController.getCursor()

        cursor.execute(query)

        cursor.connection.commit()

    def createTable():
        cursor = PredictedPricesController.getCursor()

        with open("./sql/create_table.sql", "r") as query_file:
            query = query_file.read()
            cursor.execute(query)

            cursor.connection.commit()


    def insertIntoTable(predicted_price: PredictedPrices):
        cursor = PredictedPricesController.getCursor()

        # serializamos la lista de precios a JSON
        prices_json = json.dumps(predicted_price.prices)

        query = "INSERT INTO predicted_prices (city, prices) VALUES (%s, %s);"
        cursor.execute(query, (predicted_price.city, prices_json))

        cursor.connection.commit()

    def queryCityPrices(city: str) -> PredictedPrices:
         cursor = PredictedPricesController.getCursor()
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

    def getCursor():
        connection = psycopg2.connect(database=secret_config.PGDATABASE, user=secret_config.PGUSER, password=secret_config.PGPASSWORD, host=secret_config.PGHOST)

        cursor = connection.cursor()

        return cursor