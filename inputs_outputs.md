# Predicción de Precios de Bienes Raíces en Medellín

## Descripción del Proyecto
Este proyecto desarrolla un modelo de aprendizaje automático basado en **Random Forest** utilizando la biblioteca `sklearn` para predecir los precios de bienes raíces en Medellín. Se permite la carga de datos en formatos como `CSV` o `DataFrame` y ofrece una interfaz en consola para consultas por barrio o rango de precios.

## Instalación y Requisitos
Antes de ejecutar el proyecto, asegúrese de tener instaladas las siguientes dependencias:

```bash
pip install numpy pandas scikit-learn
```

## Estructura del Proyecto
El código se compone de los siguientes módulos:

- `modelo.py`: Contiene la implementación del modelo de **Random Forest**.
- `procesamiento.py`: Encargado de la limpieza y transformación de los datos.
- `interfaz.py`: Proporciona la interfaz en consola para interactuar con el usuario.

## Entradas del Programa
El usuario puede ingresar datos en dos formatos:
1. **Archivo CSV**: Conjunto de datos históricos de precios de bienes raíces en Medellín.
2. **Entrada manual en consola**: Permite consultar predicciones ingresando:
   - `Año de consulta`
   - `Barrio de Medellín`
   - `Rango de precios`

## Procesamiento de Datos
1. **Carga de Datos**: Se importan desde un archivo CSV o DataFrame.
2. **Limpieza y Normalización**: Se eliminan datos faltantes y se convierten variables categóricas en numéricas.
3. **Entrenamiento del Modelo**: Se utiliza un conjunto de datos dividido en 80% para entrenamiento y 20% para prueba.
4. **Predicción**: El modelo genera estimaciones basadas en las entradas del usuario.

## Salidas del Programa
El programa proporciona los siguientes tipos de salidas:

1. **Predicciones de Precios**: Devuelve el precio estimado basado en las entradas del usuario.
2. **Mensajes de Advertencia**: Si no se encuentran datos suficientes para una consulta, se informa al usuario.
3. **Errores de Entrada**: Manejo de errores como años fuera del rango de datos o barrios inexistentes.

## Casos de Prueba
El programa ha sido validado con los siguientes casos de prueba:

- **Casos Normales**: Consultas con datos válidos.
- **Casos Extraordinarios**: Situaciones límite como precios extremos o barrios poco frecuentes.
- **Casos de Error**: Manejo de entradas inválidas como ciudades inexistentes o formatos incorrectos.

## Uso del Programa
Para ejecutar la interfaz en consola, utilice:
```bash
python interfaz.py
```

El usuario podrá seleccionar opciones para realizar consultas sobre precios de bienes raíces en Medellín de acuerdo con los datos ingresados.

## Contacto
Para más información o contribuciones, comuníquese con el equipo de desarrollo.

fc.........................................................................