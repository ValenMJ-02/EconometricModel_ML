import sys
import psycopg2
sys.path.append("src")

from model.predicted_prices import PredictedPrices

import secret_config

class PredictedPricesController:

    def dropTable(self):
        query = "DROP TABLE if exists predicted_prices"

        cursor = self.getCursor()

        cursor.execute(query)

        cursor.connection.commit()

    def createTable(self):
        cursor = self.getCursor()

        with open("./sql/create_table.sql", "r") as query:
            cursor.execute(query)

            cursor.connection.commit()


    def insertIntoTable(self, city: str, prices: str):
        cursor = self.getCursor()

        query = f"INSERT INTO predicted_prices (city, prices) VALUES ('{city}', '{prices}');"

        cursor.execute(query)

        cursor.connection.commit()

    def queryCityPrices(self, city: str):
        cursor = self.getCursor()

        query = f"SELECT id, predicted_at, city, prices FROM predicted_prices WHERE city = '{city}';"

        cursor.execute(query)

        cursor.connection.commit()

    def getCursor():
        connection = psycopg2.connect(database=secret_config.PGDATABASE, user=secret_config.PGUSER, password=secret_config.PGPASSWORD, host=secret_config.PGHOST)

        cursor = connection.cursor()

        return cursor