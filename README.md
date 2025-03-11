---

# **Econometric Model - Predicción de Precios de Casas**

Este proyecto tiene como objetivo predecir el precio de las casas utilizando un modelo de regresión lineal. El código está organizado siguiendo el patrón MVC (Model-View-Controller) y está diseñado para ser modular, escalable y fácil de mantener.

---

## **Estructura del Proyecto**

```
EconometricModel_ML/
├── src/
|    ├── model/
|    │   ├── __init__.py
|    │   ├── data_preparation.py
|    │   ├── feature_engineering.py
|    │   ├── model_training.py
|    │   └── future_predictions.py
|    ├── view/
|    │   ├── console/
|    |   |   └─ main.py
|    │   └── results/
|    |       └─ visualization.py
|    └── controller/
|    │   ├── __init__.py
|        ├── data_controller.py
|        └── model_controller.py
├── tests/
|    ├── test_data_preparation.py
|    ├── test_feature_engineering.py
|    └── test_model_training.py
├── data/
|    └── train.csv
├── .gitignore
├── config.py
└── README.md
```

---

## **Descripción de los Archivos**

### **1. `src/model/`**
Contiene la lógica del negocio y las operaciones relacionadas con los datos y el modelo.

- **`data_preparation.py`**: Funciones para cargar, limpiar y preparar los datos.
- **`feature_engineering.py`**: Funciones para crear nuevas características (features) y transformar los datos.
- **`model_training.py`**: Funciones para entrenar y evaluar modelos.
- **`future_predictions.py`**: Funciones para generar datos futuros y hacer predicciones.

### **2. `src/view/`**
Maneja la interacción con el usuario y la visualización de resultados.

- **`console/main.py`**: Punto de entrada para la interfaz de consola.
- **`results/visualization.py`**: Funciones para visualizar resultados (gráficos, métricas, etc.).

### **3. `src/controller/`**
Gestiona la comunicación entre el modelo y la vista.

- **`data_controller.py`**: Controla la carga y preparación de los datos.
- **`model_controller.py`**: Controla el entrenamiento y la evaluación del modelo.

### **4. `tests/`**
Contiene pruebas unitarias para cada módulo del proyecto.

- **`test_data_preparation.py`**: Pruebas para la preparación de datos.
- **`test_feature_engineering.py`**: Pruebas para la ingeniería de características.
- **`test_model_training.py`**: Pruebas para el entrenamiento y evaluación del modelo.

### **5. `data/`**
Contiene los datos utilizados para entrenar y evaluar el modelo.

- **`train.csv`**: Conjunto de datos de entrenamiento.

### **6. `config.py`**
Archivo de configuración para rutas y parámetros del proyecto.

### **7. `.gitignore`**
Archivo para ignorar archivos y carpetas que no deben ser rastreados por Git.

---

## **Cómo Ejecutar el Proyecto**

### **1. Instalación de Dependencias**

Asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tabulate
```

### **2. Ejecución del Programa**

Para ejecutar el programa, usa el siguiente comando:

```bash
python src/view/console/main.py
```

Esto cargará los datos, entrenará el modelo y hará predicciones futuras.

---

## **Pruebas Unitarias**

Para ejecutar las pruebas unitarias, usa el siguiente comando:

```bash
python -m unittest tests/test_model.py
```

---

## **Explicación del Código**

### **1. Carga y Preparación de Datos**

- **`load_data(file_path)`**: Carga los datos desde un archivo CSV.
- **`split_data(data)`**: Divide los datos en conjuntos de entrenamiento, validación y prueba.
- **`prepare_data(dataframe)`**: Llena valores nulos y prepara los datos para el modelado.

### **2. Ingeniería de Características**

- **`transform_target(df_train, df_val, df_test, target_column)`**: Transforma la variable objetivo usando `log1p`.
- **`group_by_mean_and_bin(df, df_full, column_name, bins, labels)`**: Agrupa los datos por la media de `saleprice` y los divide en bins.
- **`encode_categorical_columns(df, encoder)`**: Codifica columnas categóricas usando `LabelEncoder`.

### **3. Entrenamiento y Evaluación del Modelo**

- **`train_model(X_train, y_train)`**: Entrena un modelo de regresión lineal.
- **`evaluate_model(model, X, y, dataset_name)`**: Evalúa el modelo y muestra el puntaje.

### **4. Predicciones Futuras**

- **`generate_future_data(last_year, num_years)`**: Genera datos futuros para los próximos años.
- **`predict_future_prices(model, future_data, feature_columns)`**: Predice los precios de las casas para los años futuros.

---

## **Visualización de Resultados**

### **1. Gráfico de Líneas**

Muestra la tendencia de los precios a lo largo del tiempo.

```python
import matplotlib.pyplot as plt

def plot_future_predictions(future_predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(future_predictions['yearbuilt'], future_predictions['predicted_price'], marker='o')
    plt.title('Predicción de Precios de Casas en los Próximos Años')
    plt.xlabel('Año')
    plt.ylabel('Precio Predicho')
    plt.grid(True)
    plt.show()
```

### **2. Tabla Formateada**

Muestra las predicciones en una tabla legible.

```python
from tabulate import tabulate

def display_future_predictions(future_predictions):
    print(tabulate(future_predictions[['yearbuilt', 'predicted_price']], headers='keys', tablefmt='pretty'))
```

---

## **Contribuciones**

Si deseas contribuir a este proyecto, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añade nueva funcionalidad'`).
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un Pull Request.

---

## **Licencia**

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---
