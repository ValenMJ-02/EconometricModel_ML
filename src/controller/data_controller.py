from model.data_preparation import load_data, split_data, prepare_data
from model.feature_engineering import transform_target, group_by_mean_and_bin, encode_categorical_columns

def load_and_prepare_data(file_path):
    """Carga, divide y prepara los datos."""
    data = load_data(file_path)
    df_train, df_val, df_test, df_full_train = split_data(data)
    y_train, y_val, y_test, df_train, df_val, df_test = transform_target(df_train, df_val, df_test, 'saleprice')
    df_full_train = prepare_data(df_full_train)
    df_train = prepare_data(df_train)
    df_val = prepare_data(df_val)
    df_test = prepare_data(df_test)
    return df_train, df_val, df_test, y_train, y_val, y_test, df_full_train

def engineer_features(df_train, df_val, df_test, df_full_train):
    """Realiza ingeniería de características."""
    bins = [0, 100000, 200000, 300000, 450000, 760000]
    labels = [0, 1, 2, 3, 4]
    for column in ['neighborhood', 'exterior2nd', 'housestyle']:
        df_train = group_by_mean_and_bin(df_train, df_full_train, column, bins, labels)
        df_val = group_by_mean_and_bin(df_val, df_full_train, column, bins, labels)
        df_test = group_by_mean_and_bin(df_test, df_full_train, column, bins, labels)
    encoder = LabelEncoder()
    df_train = encode_categorical_columns(df_train, encoder)
    df_val = encode_categorical_columns(df_val, encoder)
    df_test = encode_categorical_columns(df_test, encoder)
    return df_train, df_val, df_test