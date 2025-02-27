"""
model.py
========
Este módulo contiene la clase RealEstatePredictor, que se encarga de:
  - Cargar y preprocesar los datos desde un archivo CSV.
  - Entrenar un modelo de regresión (RandomForestRegressor) utilizando una Pipeline de scikit-learn.
  - Validar entradas del usuario (ciudad, tipo de propiedad, rango de precios).
  - Realizar predicciones del 'Sale Amount' (precio de venta) para un año dado,
    y calcular predicciones futuras para los años 2025-2028.

El flujo del modelo es el siguiente:
  1. Se carga el CSV usando pandas, se limpian los datos críticos y se convierten columnas
     relevantes (por ejemplo, 'List Year', 'Sale Amount' y 'Assessed Value') a tipos adecuados.
  2. Se extrae la variable 'Year' a partir de 'List Year' para capturar la dimensión temporal.
  3. Se almacenan las ciudades y tipos de propiedad válidos, además del rango histórico de precios.
  4. Se define una Pipeline con un ColumnTransformer para preprocesar:
       - Variables numéricas: 'Assessed Value' y 'Year' se estandarizan.
       - Variables categóricas: 'Town' y 'Property Type' se codifican mediante OneHotEncoder.
  5. Se entrena un RandomForestRegressor con 100 estimadores y un estado aleatorio fijo para la reproducibilidad.
  6. La clase ofrece métodos para validar entradas y realizar predicciones, incluyendo la extrapolación
     para los años 2025-2028.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class RealEstatePredictor:
    """
    Clase para preprocesar datos de bienes raíces, entrenar un modelo de Random Forest
    y realizar predicciones del 'Sale Amount' para diferentes años.
    
    Atributos:
        csv_path (str): Ruta al archivo CSV con los datos.
        data (DataFrame): Datos preprocesados.
        pipeline (Pipeline): Pipeline entrenada que incluye preprocesamiento y el modelo.
        valid_cities (list): Lista de ciudades disponibles en los datos.
        valid_property_types (list): Lista de tipos de propiedad válidos.
        sale_amount_range (tuple): Rango histórico de 'Sale Amount' en los datos.
    """
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.pipeline = None
        self.valid_cities = None
        self.valid_property_types = None
        self.sale_amount_range = None
        
        # Cargar, limpiar y preprocesar los datos
        self.load_and_preprocess_data()
        
        # Entrenar el modelo
        self.train_model()
        
    def load_and_preprocess_data(self):
        """
        Carga y limpia el CSV.
        
        Pasos realizados:
          - Se eliminan filas que no tengan datos críticos en 'Sale Amount', 'Assessed Value',
            'Town', 'Property Type' o 'List Year'.
          - Se convierte la columna 'List Year' a entero y se guarda en una nueva columna 'Year'.
          - Se convierten 'Sale Amount' y 'Assessed Value' a valores numéricos.
          - Se filtran las filas que tengan valores no positivos en 'Sale Amount' o 'Assessed Value'.
          - Se almacenan las ciudades y tipos de propiedad válidos, además del rango histórico
            de precios de venta.
        """
        # Para evitar warnings de mezcla de tipos, se puede usar low_memory=False o especificar dtypes.
        df = pd.read_csv(self.csv_path, low_memory=False)
        
        # Eliminar filas con datos críticos faltantes
        df = df.dropna(subset=['Sale Amount', 'Assessed Value', 'Town', 'Property Type', 'List Year'])
        
        # Convertir 'List Year' a entero y crear la columna 'Year'
        try:
            df['Year'] = df['List Year'].astype(int)
        except Exception as e:
            raise ValueError("Error al convertir 'List Year' a entero: " + str(e))
        
        # Convertir 'Sale Amount' y 'Assessed Value' a numéricos
        df['Sale Amount'] = pd.to_numeric(df['Sale Amount'], errors='coerce')
        df['Assessed Value'] = pd.to_numeric(df['Assessed Value'], errors='coerce')
        df = df.dropna(subset=['Sale Amount', 'Assessed Value'])
        
        # Filtrar filas con valores no positivos
        df = df[(df['Sale Amount'] > 0) & (df['Assessed Value'] > 0)]
        self.data = df
        
        # Almacenar ciudades y tipos de propiedad válidos
        self.valid_cities = sorted(self.data['Town'].unique())
        valid_types = self.data['Property Type'].dropna().unique()
        valid_types = [pt for pt in valid_types if str(pt).strip() != ""]
        self.valid_property_types = sorted(valid_types)
        
        # Guardar el rango histórico de 'Sale Amount'
        self.sale_amount_range = (self.data['Sale Amount'].min(), self.data['Sale Amount'].max())
        
    def train_model(self):
        """
        Entrena el modelo usando una Pipeline que:
          - Preprocesa las variables:
              * Numéricas: 'Assessed Value' y 'Year' se escalan con StandardScaler.
              * Categóricas: 'Town' y 'Property Type' se codifican con OneHotEncoder.
          - Entrena un RandomForestRegressor para predecir 'Sale Amount'.
        
        La Pipeline se compone de:
          1. ColumnTransformer para preprocesar los datos.
          2. RandomForestRegressor como modelo predictivo.
        """
        X = self.data[['Assessed Value', 'Year', 'Town', 'Property Type']]
        y = self.data['Sale Amount']
        
        numerical_features = ['Assessed Value', 'Year']
        categorical_features = ['Town', 'Property Type']
        
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        self.pipeline.fit(X, y)
        
    def validate_city(self, city: str) -> str:
        """
        Valida que la ciudad ingresada esté en los datos.
        
        Args:
            city (str): Nombre de la ciudad ingresada.
        
        Returns:
            str: Ciudad validada (con espacios recortados).
            
        Raises:
            ValueError: Si la ciudad no existe en los datos.
        """
        city = city.strip()
        if city not in self.valid_cities:
            raise ValueError(f"La ciudad '{city}' no existe en el conjunto de datos. Ciudades válidas: {self.valid_cities}")
        return city
        
    def validate_property_type(self, property_type: str) -> str:
        """
        Valida que el tipo de propiedad ingresado esté entre los tipos permitidos.
        
        Args:
            property_type (str): Tipo de propiedad ingresado.
        
        Returns:
            str: Tipo de propiedad validado (con espacios recortados).
            
        Raises:
            ValueError: Si el tipo de propiedad no es válido.
        """
        property_type = property_type.strip()
        if property_type not in self.valid_property_types:
            raise ValueError(f"El tipo de propiedad '{property_type}' no es válido. Tipos válidos: {self.valid_property_types}")
        return property_type
        
    def validate_price_range(self, price_min, price_max) -> tuple:
        """
        Valida el rango de precios ingresado.
        
        Se convierte a float, se verifica que el mínimo sea menor que el máximo y
        se comprueba que el rango se solape con el rango histórico de 'Sale Amount'.
        
        Args:
            price_min: Precio mínimo (puede ser numérico o string convertible).
            price_max: Precio máximo (puede ser numérico o string convertible).
            
        Returns:
            tuple: (price_min, price_max) convertidos a float.
            
        Raises:
            ValueError: Si el rango no es numérico, si el mínimo es mayor o igual al máximo,
                        o si no se solapa con el rango histórico.
        """
        try:
            price_min = float(price_min)
            price_max = float(price_max)
        except:
            raise ValueError("El rango de precios debe ser numérico.")
            
        if price_min >= price_max:
            raise ValueError("El precio mínimo debe ser menor que el precio máximo.")
            
        historical_min, historical_max = self.sale_amount_range
        if price_max < historical_min or price_min > historical_max:
            raise ValueError(f"El rango de precios ({price_min}, {price_max}) no se solapa con el rango histórico de precios ({historical_min}, {historical_max}).")
        return price_min, price_max
        
    def get_median_assessed_value(self, city: str, property_type: str) -> float:
        """
        Calcula la mediana de 'Assessed Value' para una combinación de ciudad y tipo de propiedad.
        
        Args:
            city (str): Ciudad validada.
            property_type (str): Tipo de propiedad validado.
            
        Returns:
            float: Mediana de 'Assessed Value'.
            
        Raises:
            ValueError: Si no se encuentran datos para la combinación dada.
        """
        df_filtered = self.data[(self.data['Town'] == city) & (self.data['Property Type'] == property_type)]
        if df_filtered.empty:
            raise ValueError(f"No se encontraron datos para la ciudad '{city}' y tipo de propiedad '{property_type}'.")
        return df_filtered['Assessed Value'].median()
        
    def predict_sale_amount(self, city: str, property_type: str, year: int) -> float:
        """
        Predice el 'Sale Amount' para una combinación de ciudad, tipo de propiedad y año.
        
        Se utiliza la mediana de 'Assessed Value' para la combinación ciudad-tipo y se construye
        un DataFrame de una sola fila con las características necesarias para la predicción.
        
        Args:
            city (str): Ciudad ingresada (se valida internamente).
            property_type (str): Tipo de propiedad ingresado (se valida internamente).
            year (int): Año para el que se quiere predecir.
            
        Returns:
            float: Predicción del 'Sale Amount' para la muestra dada.
        """
        city = self.validate_city(city)
        property_type = self.validate_property_type(property_type)
        assessed_value = self.get_median_assessed_value(city, property_type)
        
        sample = pd.DataFrame({
            'Assessed Value': [assessed_value],
            'Year': [int(year)],
            'Town': [city],
            'Property Type': [property_type]
        })
        prediction = self.pipeline.predict(sample)
        return prediction[0]
        
    def get_future_predictions(self, city: str, property_type: str) -> dict:
        """
        Realiza predicciones del 'Sale Amount' para los años 2025, 2026, 2027 y 2028.
        
        Args:
            city (str): Ciudad ingresada.
            property_type (str): Tipo de propiedad ingresado.
            
        Returns:
            dict: Diccionario con la forma {año: predicción}.
        """
        predictions = {}
        for future_year in [2025, 2026, 2027, 2028]:
            pred = self.predict_sale_amount(city, property_type, future_year)
            predictions[future_year] = pred
        return predictions
