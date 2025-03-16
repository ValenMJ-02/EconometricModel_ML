import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

def transform_target(dataframe_train, dataframe_validation_features, dataframe_test, target_column):
    """Transforma la variable objetivo usando log1p y la elimina de los DataFrames."""
    train_target = np.log1p(dataframe_train[target_column].values)
    validation_target = np.log1p(dataframe_validation_features[target_column].values)
    test_target = np.log1p(dataframe_test[target_column].values)
    dataframe_train = dataframe_train.drop(columns=[target_column])
    dataframe_validation_features = dataframe_validation_features.drop(columns=[target_column])
    dataframe_test = dataframe_test.drop(columns=[target_column])
    return train_target, validation_target, test_target, dataframe_train, dataframe_validation_features, dataframe_test

def group_by_mean_and_bin(dataframe, dataframe_full, column_name, bins, labels):
    """Agrupa los datos por la media de 'saleprice' y los divide en bins."""
    mean_prices = dataframe_full.groupby(column_name)['saleprice'].mean()
    groups = pd.cut(mean_prices, bins=bins, labels=labels)
    grouped_dataframe = pd.DataFrame({
        column_name: mean_prices.index,
        f'average_saleprice_{column_name}': mean_prices.values,
        f'group_{column_name}': groups
    }).reset_index(drop=True)
    dataframe = dataframe.merge(grouped_dataframe[[column_name, f'group_{column_name}']], on=column_name, how='left')
    return dataframe

def encode_categorical_columns(dataframe, encoder):
    """Codifica columnas categóricas usando LabelEncoder."""
    categorical_columns = dataframe.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        dataframe[column] = encoder.fit_transform(dataframe[column])
    return dataframe