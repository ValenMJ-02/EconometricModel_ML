import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    
    # Eliminar columnas irrelevantes
    columns_to_drop = ["Non Use Code", "Assessor Remarks", "OPM remarks", "Location", "Address"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Imputar valores faltantes
    df["Date Recorded"] = df["Date Recorded"].fillna(df["Date Recorded"].mode()[0])
    for col in ["Property Type", "Residential Type"]:
        missing_percentage = df[col].isnull().mean()
        df[col] = df[col].fillna(df[col].mode()[0] if missing_percentage < 0.15 else "Unknown")

    # Convertir fecha a datetime
    df["Date Recorded"] = pd.to_datetime(df["Date Recorded"], errors="coerce")

    # Codificar variables categóricas
    encoders = {}
    categorical_cols = ["Town", "Property Type", "Residential Type"]
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    return df, encoders
