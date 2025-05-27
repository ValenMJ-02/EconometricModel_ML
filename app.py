from flask import Flask, request, render_template, redirect, url_for
from controller.predicted_prices_controller import PredictedPricesController
from model.predicted_prices import PredictedPrices
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/buscar', methods=['GET', 'POST'])
def buscar():
    if request.method == 'POST':
        city = request.form['inputData']
        city_fetched = PredictedPricesController.queryCityPrices(city)
        city_data = city_fetched.to_dict() if city_fetched else {}
        return render_template('mostrar_resultado.html', city_data=city_data)
    return render_template('buscar.html')

@app.route('/insertar', methods=['GET', 'POST'])
def insertar():
    if request.method == 'POST':
        city = request.form['city']
        prices = request.form['prices']
        predicted_prices = PredictedPrices(city=city, prices=json.loads(prices))
        PredictedPricesController.insertIntoTable(predicted_prices)
        return redirect(url_for('index'))
    return render_template('insertar.html')

@app.route('/actualizar', methods=['GET', 'POST'])
def actualizar():
    if request.method == 'POST':
        city = request.form['city']
        prices = request.form['prices']
        predicted_prices = PredictedPrices(city=city, prices=json.loads(prices))
        PredictedPricesController.updateCityPrices(predicted_prices)
        return redirect(url_for('index'))
    return render_template('actualizar.html')

@app.route('/eliminar', methods=['POST'])
def eliminar():
    city = request.form['city']
    PredictedPricesController.deleteCityPrices(city)
    return redirect(url_for('index'))

@app.route('/crear_tabla')
def crear_tabla():
    PredictedPricesController.createTable()
    return redirect(url_for('index'))

@app.route('/borrar_tabla')
def borrar_tabla():
    PredictedPricesController.dropTable()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
