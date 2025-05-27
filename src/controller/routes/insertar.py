from flask import Blueprint, request, render_template, redirect, url_for
import json
import sys
sys.path.append('src')

from controller.predicted_prices_controller import PredictedPricesController
from model.predicted_prices import PredictedPrices

insertar_bp = Blueprint('insertar', __name__)

@insertar_bp.route('/insertar', methods=['GET', 'POST'])
def insertar():
    if request.method == 'POST':
        city = request.form['city']
        prices_text = request.form['prices']  # Espera una cadena JSON, p. ej. '["100","200","300"]'

        try:
            # Pasamos la cadena JSON directamente al constructor
            predicted_prices = PredictedPrices(city, prices_text)
        except json.JSONDecodeError:
            return render_template('insertar.html', error="Formato de precios inválido. Ingrese un JSON válido.")

        PredictedPricesController.insertIntoTable(predicted_prices)
        return redirect(url_for('index'))

    return render_template('insertar.html')