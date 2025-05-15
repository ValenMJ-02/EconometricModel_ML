import sys, os, json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.controller.predicted_prices_controller import PredictedPricesController
from src.model.predicted_prices           import PredictedPrices
from src.controller.data_controller       import engineer_features, divide_dataframes
from src.model.model_training             import train_model
from src.controller.model_controller      import predict_future

def insert_flow():
    city = input("Ciudad a predecir: ").strip()
    if not city:
        print("Debe indicar una ciudad.")
        return

    try:
        num_years = int(input("¿Cuántos años a futuro (por defecto 3)? ").strip() or 3)
    except ValueError:
        print("Entrada inválida, usando 3 años.")
        num_years = 3

    # 1) Dividir y preparar datos
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "train.csv"))
    # mismo esquema de bins que en main.py
    bins, labels = ([0.0,100000.0,200000.0,300000.0,450000.0,760000.0], [0,1,2,3,4])
    df_tr, df_valf, df_te, y_tr, _, _, full = divide_dataframes(base, "saleprice", (bins, labels))

    # 2) Ingeniería de características
    df_tr, df_valf, df_te = engineer_features(
        (df_tr, df_valf, df_te), full,
        ['neighborhood', 'exterior2nd', 'housestyle']
    )

    # 3) Entrenar modelo sobre las mismas columnas que en main.py
    selected = [
        "lotarea","grlivarea","1stflrsf","mssubclass","overallcond","bsmtunfsf",
        "garagearea","yearbuilt","overallqual","bsmtfinsf1","group_neighborhood",
        "group_exterior2nd","fireplaces","openporchsf","2ndflrsf","group_housestyle",
        "masvnrarea","lotfrontage","lotconfig","yearremodadd","screenporch",
        "mszoning","lotshape","landcontour","landslope","condition1","bedroomabvgr",
        "roofstyle","roofmatl","exterior1st","exterqual","extercond","foundation",
        "bsmtqual","bsmtexposure","heatingqc","centralair","electrical"
    ]
    X_tr = df_tr[selected]
    model = train_model(X_tr, y_tr)

    # 4) Generar y almacenar predicciones
    future_df = predict_future({"x_train": X_tr, "num_years": num_years}, model)
    records   = future_df[['yearbuilt','predicted_price']].to_dict(orient='records')

    pp = PredictedPrices(city, records)
    PredictedPricesController.insertIntoTable(pp)
    print(f"✅ Predicciones para «{city}» insertadas ({len(records)} años).")

def query_flow():
    city = input("Ciudad a consultar: ").strip()
    res = PredictedPricesController.queryCityPrices(city)
    if not res:
        print("No hay registros.")
    else:
        print(f"Ciudad: {res.city}")
        print(json.dumps(res.prices, indent=2))

def delete_flow():
    city = input("Ciudad a eliminar: ").strip()
    PredictedPricesController.deleteCityPrices(city)
    print(f"Se eliminaron (si existían) las predicciones de «{city}».")

def main():
    ops = {
        "1": ("Insertar predicciones", insert_flow),
        "2": ("Consultar",           query_flow),
        "3": ("Eliminar",            delete_flow),
        "4": ("Salir",               None)
    }
    while True:
        print("\n=== Menú BD de Predicciones ===")
        for k,(lab,_) in ops.items():
            print(f"{k}. {lab}")
        c = input("Opción: ").strip()
        if c=="4": break
        acción = ops.get(c)
        if acción and acción[1]:
            try: acción[1]()
            except Exception as e: print("¡Error!", e)
        else:
            print("Opción no válida.")

if __name__=="__main__":
    main()
