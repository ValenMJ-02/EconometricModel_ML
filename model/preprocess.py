import pandas as pd
from sklearn.preprocessing import LabelEncoder

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas irrelevantes del DataFrame.
    """
    columns_to_drop = ["Non Use Code", "Assessor Remarks", "OPM remarks", "Location", "Address"]
    return df.drop(columns=columns_to_drop, errors="ignore")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maneja los valores faltantes en las columnas necesarias.
    """
    df["Date Recorded"] = df["Date Recorded"].fillna(df["Date Recorded"].mode()[0])
    for col in ["Property Type", "Residential Type"]:
        missing_percentage = df[col].isnull().mean()
        df[col] = df[col].fillna(df[col].mode()[0] if missing_percentage < 0.15 else "Unknown")
    return df

def encode_categorical_columns(df: pd.DataFrame):
    """
    Codifica las variables categóricas en el DataFrame.
    """
    encoders = {}
    categorical_cols = ["Town", "Property Type", "Residential Type"]
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    return df, encoders

def preprocess_data(df: pd.DataFrame):
    """
    Ejecuta todos los pasos de preprocesamiento de datos.
    """
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df["Date Recorded"] = pd.to_datetime(df["Date Recorded"], errors="coerce")
    df, encoders = encode_categorical_columns(df)
    
    return df, encoders
