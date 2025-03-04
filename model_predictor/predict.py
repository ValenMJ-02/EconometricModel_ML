import pickle
import numpy as np
import preprocess
import matplotlib.pyplot as plt

# Cargar modelo entrenado
with open("model_predictor/model.pkl", "rb") as f:
    model = pickle.load(f)

# Cargar datos y encoders
file_path = "Real_Estate_Sales_2001-2020_GL.csv"
df, encoders = preprocess.load_and_preprocess_data(file_path)

def predict_price(town_name, max_price):
    if town_name not in encoders["Town"].classes_:
        print("Error: La ciudad ingresada no existe en los datos.")
        return
    
    # Convertir town a número
    town_encoded = encoders["Town"].transform([town_name])[0]

    # Predecir para 2025, 2026 y 2027
    future_years = [2025, 2026, 2027]
    estimates = {}

    for year in future_years:
        X_future = df[df["Town"] == town_encoded].copy()
        X_future["List Year"] = year
        predicted_prices = model.predict(X_future[["List Year", "Assessed Value", "Sales Ratio", "Property Type", "Residential Type", "Town"]])
        
        estimates[year] = np.mean(predicted_prices)
    
    # Graficar
    plt.figure(figsize=(8, 5))
    plt.plot(future_years, list(estimates.values()), marker='o', linestyle='-', color='blue', label="Predicción")
    plt.xlabel("Año")
    plt.ylabel("Precio promedio ($)")
    plt.title(f"Predicción de precios en {town_name}")
    plt.legend()
    plt.grid()
    plt.show()

    return estimates
