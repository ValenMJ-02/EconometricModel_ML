from flask import Blueprint, request, render_template, redirect, url_for
import json
import sys
sys.path.append('src')
from controller.predicted_prices_controller import PredictedPricesController
from model.predicted_prices import PredictedPrices

actualizar_bp = Blueprint('actualizar', __name__)

@actualizar_bp.route('/actualizar', methods=['GET', 'POST'])
def actualizar():
    if request.method == 'POST':
        city = request.form['city']
        prices_text = request.form['prices']

        try:
            predicted_prices = PredictedPrices(city, prices_text)
        except json.JSONDecodeError:
            return render_template('actualizar.html', error="Formato de precios inválido. Ingrese un JSON válido.")

        PredictedPricesController.updateCityPrices(predicted_prices)
        return redirect(url_for('index'))

    return render_template('actualizar.html')