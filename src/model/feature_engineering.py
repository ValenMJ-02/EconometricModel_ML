import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def transform_target(df_train, df_val, df_test, target_column):
    """Transforma la variable objetivo usando log1p y la elimina de los DataFrames."""
    y_train = np.log1p(df_train[target_column].values)
    y_val = np.log1p(df_val[target_column].values)
    y_test = np.log1p(df_test[target_column].values)
    df_train = df_train.drop(columns=[target_column])
    df_val = df_val.drop(columns=[target_column])
    df_test = df_test.drop(columns=[target_column])
    return y_train, y_val, y_test, df_train, df_val, df_test

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