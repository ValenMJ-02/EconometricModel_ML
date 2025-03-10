import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data_from_gcs(bucket_name: str, file_name: str) -> pd.DataFrame:
    """Carga el archivo CSV desde Google Cloud Storage en un DataFrame."""
    from google.cloud import storage
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    content = blob.download_as_text()
    return pd.read_csv(pd.compat.StringIO(content))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas irrelevantes y maneja valores nulos."""
    columns_to_drop = ["Non Use Code", "Assessor Remarks", "OPM remarks", "Location", "Address"]
    df.drop(columns=columns_to_drop, errors="ignore", inplace=True)
    
    df["Date Recorded"].fillna(df["Date Recorded"].mode()[0], inplace=True)
    
    for col in ["Property Type", "Residential Type"]:
        missing_percentage = df[col].isnull().mean()
        df[col].fillna(df[col].mode()[0] if missing_percentage < 0.15 else "Unknown", inplace=True)
    
    df["Date Recorded"] = pd.to_datetime(df["Date Recorded"], errors="coerce")
    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Codifica las variables categóricas usando Label Encoding."""
    encoders = {}
    categorical_cols = ["Town", "Property Type", "Residential Type"]
    
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    
    return df, encoders
