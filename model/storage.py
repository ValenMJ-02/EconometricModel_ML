from google.cloud import storage
import pandas as pd
import joblib
import os

class StorageHandler:
    """Handles file operations in Google Cloud Storage."""

    def __init__(self, bucket_name):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, local_path, gcs_path):
        """Uploads a file to Google Cloud Storage."""
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to {gcs_path} in GCS.")

    def download_file(self, gcs_path, local_path):
        """Downloads a file from Google Cloud Storage."""
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        print(f"Downloaded {gcs_path} to {local_path}.")

    def load_csv_as_dataframe(self, gcs_path):
        """Loads a CSV file from Google Cloud Storage into a pandas DataFrame."""
        blob = self.bucket.blob(gcs_path)
        content = blob.download_as_text()
        return pd.read_csv(pd.io.common.StringIO(content))

    def save_model(self, model, gcs_path):
        """Saves a trained model to Google Cloud Storage."""
        local_model_path = "temp_model.pkl"
        joblib.dump(model, local_model_path)
        self.upload_file(local_model_path, gcs_path)
        os.remove(local_model_path)

    def load_model(self, gcs_path):
        """Loads a model from Google Cloud Storage."""
        local_model_path = "temp_model.pkl"
        self.download_file(gcs_path, local_model_path)
        model = joblib.load(local_model_path)
        os.remove(local_model_path)
        return model
