# *Econometric Model - Predicción de Precios de Casas*

*Hecho por:*

- ***Kevin Sebastián Cifuentes López.***

- ***Mariana Lopera Correa.***

En este proyecto, se desarrolla un modelo de Machine Learning para predecir los precios de venta de viviendas en Ames, Iowa, Estados Unidos. El conjunto de datos utilizado fue sacado de Kaggle [link datos](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/code), este contiene información detallada sobre diversas características de las casas, como el tamaño del lote, el área habitable, la calidad de construcción, el año de construcción, entre otros. El objetivo es utilizar estas características para entrenar un modelo de regresión lineal que pueda predecir con precisión el precio de venta de una casa.

El modelo no solo se enfoca en predecir los precios actuales, sino que también está diseñado para realizar proyecciones futuras, permitiendo estimar cómo podrían evolucionar los precios de las viviendas en los próximos años. Esto es especialmente útil para inversionistas, agentes inmobiliarios y propietarios que deseen tomar decisiones informadas basadas en tendencias del mercado.

## **Cómo Ejecutar el Proyecto**

### **1. Instalación de Dependencias**

Asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tabulate
```

### **2. Clonar el repositorio**

```bash
git clone https://github.com/Mayday3003/EconometricModel_ML.git
```

### **3. Ejecución del Programa**

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




## **Explicación del Código**

### **1. Carga y Preparación de Datos**

- **`load_data(file_path)`**: Carga los datos desde un archivo CSV.
- **`split_data(data)`**: Divide los datos en conjuntos de entrenamiento, validación y prueba.
- **`divide_dataframes(dataframe)`**: Llena valores nulos y prepara los datos para el modelado.

### **2. Ingeniería de Características**

- **`transform_target(dataframe_train, dataframe_validation_features, dataframe_test, target_column)`**: Transforma la variable objetivo usando `log1p`.
- **`group_by_mean_and_bin(df, df_full, column_name, bins, labels)`**: Agrupa los datos por la media de `saleprice` y los divide en bins.
- **`encode_categorical_columns(df, encoder)`**: Codifica columnas categóricas usando `LabelEncoder`.

### **3. Entrenamiento y Evaluación del Modelo**

- **`train_model(x_train, train_target)`**: Entrena un modelo de regresión lineal.
- **`evaluate_model(model, X, y, dataset_name)`**: Evalúa el modelo y muestra el puntaje.

### **4. Predicciones Futuras**

- **`generate_future_data(last_year, num_years)`**: Genera datos futuros para los próximos años.
- **`predict_future_prices(model, future_data, feature_columns)`**: Predice los precios de las casas para los años futuros.

---

## **Visualización de Resultados**

### **Consola**

#### *1. Gráfico de Líneas*

Muestra la tendencia de los precios a lo largo del tiempo.


#### *2. Tabla Formateada*

Muestra una tabla de esta forma, para facilitar la obsrvación de los resultados

``
   yearbuilt  predicted_price
0       2026      2746.285057
1       2027      2750.765683
2       2028      2755.253615
``
## **Contribuciones**

Si deseas contribuir a este proyecto, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añade nueva funcionalidad'`).
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`).
5. Abre un Pull Request.

---
## La Aplicación Web
La aplicación web, hecha con Flask, es una interfaz fácil de usar que te permite gestionar información clave y consultar las predicciones de precios generadas por el modelo de Machine Learning. Funciona como un panel de control intuitivo donde puedes:

Manejar tu Información: Podrás añadir, buscar y actualizar. Esto es útil para llevar un registro de todo lo relacionado con las casas.
Ver y Guardar Predicciones de Precios: La página te permitirá ver las predicciones de precios de casas que hizo nuestro modelo. También podrás guardar esas predicciones en la base de datos para consultarlas más tarde, quizás buscando por ciudad.
Usarla Fácilmente: Todo está organizado de forma clara con tablas y formularios, lo que la hace muy sencilla de usar.
Toda la información se guarda de forma segura en una base de datos en la nube (Neon).

## Cómo Ejecutar el Proyecto
Para poner en marcha el proyecto, sigue estos pasos. Puedes usarlo de dos maneras: como un programa en la consola (para el modelo de predicción) o como una página web.

### 1. Instalación de Dependencias
Primero, asegúrate de tener todas las herramientas necesarias. Abre tu consola y ejecuta este comando para instalar las librerías de Python listadas en el archivo requirements.txt:

```

pip install -r requirements.txt
```
### 2. Clonar el Repositorio
Si aún no lo tienes, descarga el proyecto a tu computadora con Git:

```
git clone https://github.com/ValenMJ-02/EconometricModel_ML.git
cd EconometricModel_ML
```
### 3. Configuración de la Base de Datos (Neon)
La página web necesita un lugar para guardar tus datos y las predicciones. Usamos Neon, una base de datos en la nube.

#### a. Paso 1: Consigue tu base de datos Neon.
Ve a Neon y crea una cuenta y una base de datos nueva. Cuando esté lista, te darán unos datos clave para conectarte (como la dirección, el usuario y la contraseña).

#### b. Paso 2: Dile al proyecto dónde está tu base de datos.
En la carpeta principal de tu proyecto, crea un archivo llamado secret_config.py (si no existe) y pega ahí los datos que te dio Neon. ¡Importante! Este archivo tiene información privada, así que no lo subas a GitHub. Asegúrate de que esté listado en tu archivo .gitignore.

Python

# secret_config.py
PGHOST     = os.getenv("PGHOST")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER     = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")
#### c. Paso 3: Deja la base de datos lista.
Después de configurar la conexión, necesitamos crear los "espacios" (tablas) donde se guardará tu información. Abre tu consola, ve a la carpeta principal del proyecto y ejecuta este comando. Esto preparará la base de datos para guardar clientes, propiedades, hipotecas, pagos y las predicciones de precios:

```

python -c "from src.controller.predicted_prices_controller import PredictedPricesController; from src.controller.model_controller import model_controller; from src.controller.data_controller import data_controller"
```
### 4. Ejecución del Programa (Modos de Uso)
#### a. Ejecutar la Página Web
Para usar la página web que te ayuda a gestionar información y ver predicciones:

##### i. En tu computadora:
Puedes visitarla directamente si ya está en internet, en:
https://econometricmodel-ml.onrender.com

## **Licencia**

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---
