# 1. Importaciones
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sns.set()

# 2. Definición de funciones
def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
    return data

def split_data(data, test_size=0.2, val_size=0.25, random_state=1):
    """Divide los datos en conjuntos de entrenamiento, validación y prueba."""
    df_full_train, df_test = train_test_split(data, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_full_train, test_size=val_size, random_state=random_state)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_val, df_test, df_full_train

def prepare_data(dataframe):
    """Prepara los datos llenando valores nulos."""
    if 'fireplacequ' in dataframe.columns:
        dataframe['fireplacequ'] = dataframe['fireplacequ'].fillna('NA')
    if 'garageyrblt' in dataframe.columns:
        dataframe['garageyrblt'] = dataframe['garageyrblt'].fillna(0)

    for col in dataframe.columns:
        if col not in ['fireplacequ', 'garageyrblt']:
            if dataframe[col].dtype == 'object' or dataframe[col].dtype.name == 'category':
                if not dataframe[col].isnull().all():
                    dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
                else:
                    dataframe[col] = dataframe[col].fillna('NA')
            else:
                dataframe[col] = dataframe[col].fillna(dataframe[col].astype(float).mean())
    return dataframe

def transform_target(df_train, df_val, df_test, target_column):
    """Transforma la variable objetivo usando log1p y la elimina de los DataFrames."""
    y_train = np.log1p(df_train[target_column].values)
    y_val = np.log1p(df_val[target_column].values)
    y_test = np.log1p(df_test[target_column].values)
    df_train = df_train.drop(columns=[target_column])
    df_val = df_val.drop(columns=[target_column])
    df_test = df_test.drop(columns=[target_column])
    return y_train, y_val, y_test, df_train, df_val, df_test

def calculate_mutual_info(df, target_column):
    """Calcula la información mutua para una columna categórica."""
    def mutual_info_saleprice_score(series):
        return mutual_info_score(series, df[target_column])
    return mutual_info_saleprice_score

def group_by_mean_and_bin(df, df_full, column_name, bins, labels):
    """Agrupa los datos por la media de 'saleprice' y los divide en bins."""
    mean_prices = df_full.groupby(column_name)['saleprice'].mean()
    groups = pd.cut(mean_prices, bins=bins, labels=labels)
    grouped_df = pd.DataFrame({
        column_name: mean_prices.index,
        f'average_saleprice_{column_name}': mean_prices.values,
        f'group_{column_name}': groups
    }).reset_index(drop=True)
    df = df.merge(grouped_df[[column_name, f'group_{column_name}']], on=column_name, how='left')
    return df

def encode_categorical_columns(df, encoder):
    """Codifica columnas categóricas usando LabelEncoder."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])
    return df

def train_model(X_train, y_train):
    """Entrena un modelo de regresión lineal."""
    model = LinearRegression().fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, dataset_name):
    """Evalúa el modelo y muestra el puntaje."""
    score = model.score(X, y)
    print(f"{dataset_name} set score: {score:.2f}")

# 3. Función principal
def main():
    # Cargar datos
    data = load_data('data/train.csv')
    
    # Dividir datos
    df_train, df_val, df_test, df_full_train = split_data(data)
    
    # Transformar la variable objetivo
    y_train, y_val, y_test, df_train, df_val, df_test = transform_target(df_train, df_val, df_test, 'saleprice')
    
    # Preparar datos
    df_full_train = prepare_data(df_full_train)
    df_train = prepare_data(df_train)
    df_val = prepare_data(df_val)
    df_test = prepare_data(df_test)
    
    # Calcular información mutua
    mutual_info_saleprice_score = calculate_mutual_info(df_full_train, 'saleprice')
    
    # Definir bins y labels
    bins = [0, 100000, 200000, 300000, 450000, 760000]
    labels = [0, 1, 2, 3, 4]
    
    # Agrupar y binning
    for column in ['neighborhood', 'exterior2nd', 'housestyle']:
        df_train = group_by_mean_and_bin(df_train, df_full_train, column, bins, labels)
        df_val = group_by_mean_and_bin(df_val, df_full_train, column, bins, labels)
        df_test = group_by_mean_and_bin(df_test, df_full_train, column, bins, labels)
    
    # Codificar columnas categóricas
    encoder = LabelEncoder()
    df_train = encode_categorical_columns(df_train, encoder)
    df_val = encode_categorical_columns(df_val, encoder)
    df_test = encode_categorical_columns(df_test, encoder)
    
    # Seleccionar columnas
    selected_columns = [
        "lotarea", "grlivarea", "1stflrsf", "mssubclass", "overallcond",
        "bsmtunfsf", "garagearea", "yearbuilt", "overallqual", "bsmtfinsf1", "group_neighborhood",
        "group_exterior2nd", "fireplaces", "openporchsf", "2ndflrsf",
        "group_housestyle", "masvnrarea", "lotfrontage", "lotconfig", "yearremodadd", "screenporch",
        'mszoning', 'lotshape', 'landcontour', 'landslope', 'condition1', "bedroomabvgr",
        'roofstyle', 'roofmatl', 'exterior1st', 'exterqual', 'extercond', 'foundation',
        'bsmtqual', 'bsmtexposure', 'heatingqc', 'centralair', 'electrical',
        'functional', 'garagequal', 'paveddrive'
    ]
    
    X_train = df_train[selected_columns]
    X_val = df_val[selected_columns]
    X_test = df_test[selected_columns]
    
    # Entrenar modelo
    model = train_model(X_train, y_train)
    
    # Evaluar modelo
    evaluate_model(model, X_train, y_train, "Training")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")

# 4. Ejecución del programa
if __name__ == "__main__":
    main()