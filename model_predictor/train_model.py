import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import preprocess

# Cargar y procesar datos
file_path = "Real_Estate_Sales_2001-2020_GL.csv"
df, encoders = preprocess.load_and_preprocess_data(file_path)

# Definir características y variable objetivo
features = ["List Year", "Assessed Value", "Sales Ratio", "Property Type", "Residential Type", "Town"]
target = "Sale Amount"

df = df.dropna(subset=[target])
X = df[features]
y = df[target]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo entrenado
with open("model_predictor/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluar modelo
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
