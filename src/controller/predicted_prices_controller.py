import sys
import psycopg2
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

        with open("./sql/create_table.sql", "r") as query:
            cursor.execute(query)

            cursor.connection.commit()


    def insertIntoTable(predicted_price: PredictedPrices):
        cursor = PredictedPricesController.getCursor()

        query = f"INSERT INTO predicted_prices (city, prices) VALUES ('{predicted_price.city}', '{predicted_price.prices}');"

        cursor.execute(query)

        cursor.connection.commit()

    def queryCityPrices(city: str) -> PredictedPrices:
        cursor = PredictedPricesController().getCursor()

        query = f"SELECT id, predicted_at, city, prices FROM predicted_prices WHERE city = '{city}';"

        cursor.execute(query)

        row = cursor.fetchone()

        result = PredictedPrices(city=row[2], prices=row[3])

        return result

    def getCursor():
        connection = psycopg2.connect(database=secret_config.PGDATABASE, user=secret_config.PGUSER, password=secret_config.PGPASSWORD, host=secret_config.PGHOST)

        cursor = connection.cursor()

        return cursor