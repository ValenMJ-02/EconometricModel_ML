import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


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