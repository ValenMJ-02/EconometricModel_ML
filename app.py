from flask import Flask, render_template
from src.controller.routes.insertar import insertar_bp
from src.controller.routes.actualizar import actualizar_bp
from src.controller.routes.buscar import buscar_bp
from src.controller.routes.admin_tablas import admin_tablas_bp
from src.controller.routes.eliminar import eliminar_bp

app = Flask(__name__)

app.register_blueprint(insertar_bp)
app.register_blueprint(actualizar_bp)
app.register_blueprint(buscar_bp)
app.register_blueprint(admin_tablas_bp)
app.register_blueprint(eliminar_bp)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
