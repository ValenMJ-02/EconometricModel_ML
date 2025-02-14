# Modelo de Predicción de Precios de Bienes Raíces en Connecticut

Este proyecto utiliza un modelo de Machine Learning basado en **Random Forest** para predecir los precios de bienes raíces en estado de Connecticut, Estados Unidos. El modelo se entrena con datos de la última década y permite realizar consultas futuras sobre el valor estimado de las propiedades.

## Requisitos

- Python 3.x
- Bibliotecas necesarias:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

Se pueden instalar las dependencias con el siguiente comando:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Datos

Los datos son sacados de **Kaggle** [Link](https://www.kaggle.com/datasets/arjavaniket/real-estate-sales-2001-2022-gl)

El dataset utilizado para entrenar y testear el modelo contiene información de ventas de bienes raíces en Connecticut entre los años 2001 y 2022. Para este modelo, se seleccionaron los datos de la última década y para decidir las variables que se usaron en el modelo se realizó una matriz de correlación de Pearson, finalmente, se filtraron las siguientes columnas relevantes:

- **Assessed Value**: Valor tasado de la propiedad.
- **Sale Amount**: Precio de venta.
- **Sales Ratio**: Relación entre el valor tasado y el precio de venta.
- **Property Type**: Tipo de propiedad.
- **Residential Type**: Tipo de vivienda.
- **Location**: Ubicación de la propiedad.

## Proceso de Modelado

1. **Carga y preprocesamiento de datos**:
   - Se filtraron los datos de los últimos 10 años.
   - Se trataron los valores nulos y duplicados.
   - Se convierten variables categóricas en variables numéricas usando *One-Hot Encoding*.
   
2. **División del conjunto de datos**:
   - 80% para entrenamiento.
   - 20% para prueba.

3. **Entrenamiento del modelo**:
   - Se utiliza un **Random Forest Regressor** con 100 árboles.
   - Se normalizan los datos numéricos con `StandardScaler`.
   - Se ajusta el modelo y se evalúa con las métricas **MAE** y **R2-score**.

4. **Predicción futura**:
   - Se pueden ingresar valores de ubicación, tipo de propiedad y otras características para obtener una estimación del precio futuro.





## Casos de Prueba

Se han definido los siguientes tipos de pruebas para garantizar la robustez del modelo:

1. **Casos Normales**:
   - Predicción con datos típicos de propiedades.
   - Evaluación del modelo con datos reales.

2. **Casos Extraordinarios**:
   - Propiedades con valores extremadamente altos o bajos.
   - Ubicaciones poco comunes en el conjunto de datos.

3. **Casos de Error**:
   - Datos faltantes o incorrectos.
   - Tipo de propiedad desconocido.

---

Este proyecto permite estimar el valor de bienes raíces en Medellín utilizando datos históricos y modelos de Machine Learning. Para mejorar su rendimiento, se recomienda ajustar los hiperparámetros y utilizar técnicas de validación cruzada.

