import sys
import os
from sklearn.preprocessing import LabelEncoder
from model.data_preparation import load_data, split_data, prepare_data
from model.feature_engineering import transform_target, group_by_mean_and_bin, encode_categorical_columns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def divide_dataframes(file_path: str):
    """
    Carga y divide los datos en conjuntos de entrenamiento, validación y prueba.

    Args:
        file_path (str): Ruta del archivo de datos.

    Returns:
        Tuple de DataFrames: Conjuntos de datos divididos.
    """
    try:
        data = load_data(file_path)
        dataframe_train, dataframe_validation_features, dataframe_test, df_full_train = split_data(data)
        train_target, validation_target, test_target, dataframe_train, dataframe_validation_features, dataframe_test = transform_target(
            dataframe_train, dataframe_validation_features, dataframe_test, 'saleprice'
        )
        df_full_train = prepare_data(df_full_train)
        dataframe_train = prepare_data(dataframe_train)
        dataframe_validation_features = prepare_data(dataframe_validation_features)
        dataframe_test = prepare_data(dataframe_test)

        return dataframe_train, dataframe_validation_features, dataframe_test, train_target, validation_target, test_target, df_full_train
    except FileNotFoundError:
        raise Exception(f"El archivo '{file_path}' no fue encontrado.")
    except Exception as e:
        raise Exception(f"Error al dividir los datos: {str(e)}")


def engineer_features(dataframe_train, dataframe_validation_features, dataframe_test, df_full_train):
    """
    Aplica ingeniería de características a los conjuntos de datos.

    Args:
        dataframe_train (pd.DataFrame): Datos de entrenamiento.
        dataframe_validation_features (pd.DataFrame): Datos de validación.
        dataframe_test (pd.DataFrame): Datos de prueba.
        df_full_train (pd.DataFrame): Conjunto de datos completo para referencia.

    Returns:
        Tuple de DataFrames: Datos con características transformadas.
    """
    try:
        bins = [0, 100000, 200000, 300000, 450000, 760000]
        labels = [0, 1, 2, 3, 4]

        for column in ['neighborhood', 'exterior2nd', 'housestyle']:
            dataframe_train = group_by_mean_and_bin(dataframe_train, df_full_train, column, bins, labels)
            dataframe_validation_features = group_by_mean_and_bin(dataframe_validation_features, df_full_train, column, bins, labels)
            dataframe_test = group_by_mean_and_bin(dataframe_test, df_full_train, column, bins, labels)

        encoder = LabelEncoder()
        dataframe_train = encode_categorical_columns(dataframe_train, encoder)
        dataframe_validation_features = encode_categorical_columns(dataframe_validation_features, encoder)
        dataframe_test = encode_categorical_columns(dataframe_test, encoder)

        return dataframe_train, dataframe_validation_features, dataframe_test
    except KeyError as e:
        raise Exception(f"Error en la ingeniería de características: columna faltante {str(e)}")
    except Exception as e:
        raise Exception(f"Error en la ingeniería de características: {str(e)}")
