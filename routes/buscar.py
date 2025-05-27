from flask import Blueprint, render_template, request
import sys
sys.path.append('src')

from controller.predicted_prices_controller import PredictedPricesController

buscar_bp = Blueprint('buscar', __name__)



@buscar_bp.route('/buscar', methods=['GET', 'POST'])
def buscar():
    if request.method == 'POST':
        city = request.form.get('inputData')
        city_fetched = PredictedPricesController.queryCityPrices(city)

        if city_fetched:
            city_data = {
                'city': city_fetched.city,
                'prices': city_fetched.prices
            }
        else:
            city_data = None

        return render_template('mostrar_resultado.html', city_data=city_data)

    return render_template('buscar.html')