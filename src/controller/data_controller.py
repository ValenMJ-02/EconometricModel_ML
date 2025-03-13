import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.preprocessing import LabelEncoder  # Importar LabelEncoder
from model.data_preparation import load_data, split_data, prepare_data
from model.feature_engineering import transform_target, group_by_mean_and_bin, encode_categorical_columns

def divide_dataframes(file_path):
    data = load_data(file_path)
    dataframe_train, dataframe_validation_features, dataframe_test, df_full_train = split_data(data)
    train_target, validation_target, test_target, dataframe_train, dataframe_validation_features, dataframe_test = transform_target(dataframe_train, dataframe_validation_features, dataframe_test, 'saleprice')
    df_full_train = prepare_data(df_full_train)
    dataframe_train = prepare_data(dataframe_train)
    dataframe_validation_features = prepare_data(dataframe_validation_features)
    dataframe_test = prepare_data(dataframe_test)
    return dataframe_train, dataframe_validation_features, dataframe_test, train_target, validation_target, test_target, df_full_train

def engineer_features(dataframe_train, dataframe_validation_features, dataframe_test, df_full_train):
    """Realiza ingeniería de características."""
    bins = [0, 100000, 200000, 300000, 450000, 760000]
    labels = [0, 1, 2, 3, 4]
    for column in ['neighborhood', 'exterior2nd', 'housestyle']:
        dataframe_train = group_by_mean_and_bin(dataframe_train, df_full_train, column, bins, labels)
        dataframe_validation_features = group_by_mean_and_bin(dataframe_validation_features, df_full_train, column, bins, labels)
        dataframe_test = group_by_mean_and_bin(dataframe_test, df_full_train, column, bins, labels)
    encoder = LabelEncoder()  # Ahora LabelEncoder está definido
    dataframe_train = encode_categorical_columns(dataframe_train, encoder)
    dataframe_validation_features = encode_categorical_columns(dataframe_validation_features, encoder)
    dataframe_test = encode_categorical_columns(dataframe_test, encoder)
    return dataframe_train, dataframe_validation_features, dataframe_test