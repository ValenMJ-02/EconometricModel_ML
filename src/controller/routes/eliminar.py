from flask import Blueprint, render_template, request, redirect, url_for
import sys
sys.path.append('src')
from controller.predicted_prices_controller import PredictedPricesController

eliminar_bp = Blueprint('eliminar', __name__)


@eliminar_bp.route('/eliminar', methods=['GET','POST'])
def eliminar():
    if request.method == 'GET':
        return render_template('eliminar.html')
    city = request.form['city']
    PredictedPricesController.deleteCityPrices(city)
    return redirect(url_for('index'))