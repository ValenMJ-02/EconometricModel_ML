from flask import Flask, request, render_template, json
import sys


sys.path.append("src")

from src.controller.predicted_prices_controller import PredictedPricesController
from src.model.predicted_prices import PredictedPrices

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/buscar', methods=['GET', 'POST'])
def buscar():
    if request.method == "POST":
        city = request.form["inputData"]
        city_fetched = PredictedPricesController.queryCityPrices(city)
        return city_fetched.toDict()
    return render_template("buscar.html")
    

if __name__ == '__main__':
    PredictedPricesController.createTable()
    prices_data = [
    {"yearbuilt": 2000, "saleprice": 250000},
    {"yearbuilt": 2005, "saleprice": 300000},
    {"yearbuilt": 2010, "saleprice": 350000}
]
    predicted_prices = PredictedPrices(city="medellin", prices_data=prices_data)
    PredictedPricesController.insertIntoTable(predicted_prices)
    app.run(debug=True)
    PredictedPricesController.dropTable()
