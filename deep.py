import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")

sns.set()

# Cargar los datos
data = pd.read_csv('data/train.csv')

# Limpieza de nombres de columnas
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

# Verificar las columnas
print("Columnas en el DataFrame:", data.columns)

# División de los datos
df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Resetear índices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Transformación de la variable objetivo
y_train = np.log1p(df_train['saleprice'].values)
y_val = np.log1p(df_val['saleprice'].values)
y_test = np.log1p(df_test['saleprice'].values)

# Eliminar la columna 'saleprice' de los DataFrames
df_train = df_train.drop(columns=['saleprice'])
df_val = df_val.drop(columns=['saleprice'])
df_test = df_test.drop(columns=['saleprice'])

# Función para preparar los datos
def Prepare_Data(dataframe):
    # Manejo específico para 'fireplacequ' y 'garageyrblt'
    if 'fireplacequ' in dataframe.columns:
        dataframe['fireplacequ'] = dataframe['fireplacequ'].fillna('NA')
    if 'garageyrblt' in dataframe.columns:
        dataframe['garageyrblt'] = dataframe['garageyrblt'].fillna(0)

    # Llenar valores nulos en el resto de las columnas
    for col in dataframe.columns:
        if col not in ['fireplacequ', 'garageyrblt']:
            if dataframe[col].dtype == 'object' or dataframe[col].dtype.name == 'category':
                # Verificar si hay valores no nulos antes de usar mode()
                if not dataframe[col].isnull().all():
                    dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
                else:
                    dataframe[col] = dataframe[col].fillna('NA')  # Valor por defecto para columnas categóricas
            else:
                # Solo aplicar la media a columnas numéricas
                dataframe[col] = dataframe[col].fillna(dataframe[col].astype(float).mean())

    return dataframe


# Limpieza de cada DataFrame
df_full_train = Prepare_Data(df_full_train)
df_train = Prepare_Data(df_train)
df_val = Prepare_Data(df_val)
df_test = Prepare_Data(df_test)

# Análisis de riesgo por categoría
global_saleprice = data['saleprice'].mean()

for c in df_train.select_dtypes(include=['object']).columns:
    print(c)
    df_group = data.groupby(c)['saleprice'].agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_saleprice
    df_group['risk'] = df_group['mean'] / global_saleprice
    print(df_group)  # Reemplazado display con print
    print("\n")


# Selección de columnas categóricas
string_type_cols = df_train.select_dtypes(include=['object']).columns.tolist()

# Función para calcular la información mutua
def mutual_info_saleprice_score(series):
    return mutual_info_score(series, df_full_train['saleprice'])

# Mutual information para variables categóricas
mutual_info_obj = df_full_train[string_type_cols].apply(mutual_info_saleprice_score)
mutual_info_obj.sort_values(ascending=False)

# Mutual information para variables numéricas
numerical_type_cols = df_train.select_dtypes(include=['number']).columns.tolist()
mutual_info_num = df_full_train[numerical_type_cols].apply(mutual_info_saleprice_score)
mutual_info_num.sort_values(ascending=False)

# Función para agrupar por media y bin
def group_by_mean_and_bin(df, column_name, bins, labels):
    mean_prices = df_full_train.groupby(column_name)['saleprice'].mean()
    groups = pd.cut(mean_prices, bins=bins, labels=labels)
    grouped_df = pd.DataFrame({
        column_name: mean_prices.index,
        f'average_saleprice_{column_name}': mean_prices.values,
        f'group_{column_name}': groups
    }).reset_index(drop=True)
    df = df.merge(grouped_df[[column_name, f'group_{column_name}']], on=column_name, how='left')
    return df

# Definición de bins y labels
bins = [0, 100000, 200000, 300000, 450000, 760000]
labels = [0, 1, 2, 3, 4]

# Aplicación de la función a las columnas 'neighborhood', 'exterior2nd', y 'housestyle'
df_train = group_by_mean_and_bin(df_train, 'neighborhood', bins, labels)
df_val = group_by_mean_and_bin(df_val, 'neighborhood', bins, labels)
df_test = group_by_mean_and_bin(df_test, 'neighborhood', bins, labels)

df_train = group_by_mean_and_bin(df_train, 'exterior2nd', bins, labels)
df_val = group_by_mean_and_bin(df_val, 'exterior2nd', bins, labels)
df_test = group_by_mean_and_bin(df_test, 'exterior2nd', bins, labels)

df_train = group_by_mean_and_bin(df_train, 'housestyle', bins, labels)
df_val = group_by_mean_and_bin(df_val, 'housestyle', bins, labels)
df_test = group_by_mean_and_bin(df_test, 'housestyle', bins, labels)

# Codificación de variables categóricas
encoder = LabelEncoder()

categorical_columns = df_train.select_dtypes(include=['object']).columns
categorical_columns_test = df_test.select_dtypes(include=['object']).columns
categorical_columns_val = df_val.select_dtypes(include=['object']).columns

for column in categorical_columns:
    df_train[column] = encoder.fit_transform(df_train[column])

for column in categorical_columns_val:
    df_val[column] = encoder.fit_transform(df_val[column])

for column in categorical_columns_test:
    df_test[column] = encoder.fit_transform(df_test[column])

# Selección de columnas
selected_columns = ["lotarea", "grlivarea", "1stflrsf", "mssubclass", "overallcond",
                    "bsmtunfsf", "garagearea", "yearbuilt", "overallqual", "bsmtfinsf1" ,"group_neighborhood",
                    "group_exterior2nd", "fireplaces", "openporchsf", "2ndflrsf",
                    "group_housestyle", "masvnrarea", "lotfrontage", "lotconfig", "yearremodadd", "screenporch",
                    'mszoning', 'lotshape', 'landcontour', 'landslope', 'condition1', "bedroomabvgr",
                    'roofstyle', 'roofmatl', 'exterior1st','exterqual', 'extercond', 'foundation',
                    'bsmtqual', 'bsmtexposure','heatingqc', 'centralair', 'electrical',
                    'functional', 'garagequal', 'paveddrive']

df_selected = df_train[selected_columns]

# Preparación de datos de prueba
X_test = df_test[selected_columns]
X_test = Prepare_Data(X_test)

# Preparación de datos de entrenamiento y validación
X_train_ = df_train[selected_columns]
X_val = Prepare_Data(df_val[selected_columns])
X_val.lotfrontage = X_val.lotfrontage.fillna(X_val.lotfrontage.mean())
X_val.masvnrarea = X_val.masvnrarea.fillna(X_val.masvnrarea.mean())

# Entrenamiento del modelo
linear_model = LinearRegression().fit(X_train_, y_train)

# Impresión de scores
print("Training set score: {:.2f}".format(linear_model.score(X_train_, y_train)))
print("Test set score: {:.2f}".format(linear_model.score(X_test, y_test)))
print("Validation set score: {:.2f}".format(linear_model.score(X_val, y_val)))

# Predicciones
y_pred_linear = linear_model.predict(X_val)