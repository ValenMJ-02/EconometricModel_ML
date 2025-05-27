from flask import Blueprint, redirect, url_for
import sys
sys.path.append('src')

from controller.predicted_prices_controller import PredictedPricesController

admin_tablas_bp = Blueprint('admin_tablas', __name__)

@admin_tablas_bp.route('/crear_tabla')
def crear_tabla():
    PredictedPricesController.createTable()
    return redirect(url_for('index'))


@admin_tablas_bp.route('/borrar_tabla')
def borrar_tabla():
    PredictedPricesController.dropTable()
    return redirect(url_for('index'))