from src.controller.storage import StorageHandler
from src.model.preprocessing import Preprocessor
from src.model.trainer import ModelTrainer

class RealEstateHandler:
    """Coordinates data processing, model training, and storage interactions."""

    def __init__(self, bucket_name, dataset_path, model_path):
        self.storage = StorageHandler(bucket_name)
        self.dataset_path = dataset_path
        self.model_path = model_path

    def load_and_preprocess_data(self):
        """Loads data from Google Cloud Storage and preprocesses it."""
        df = self.storage.load_csv_as_dataframe(self.dataset_path)
        preprocessor = Preprocessor()
        processed_data, encoders = preprocessor.preprocess(df)
        return processed_data, encoders

    def train_and_store_model(self, processed_data):
        """Trains a model and uploads it to Google Cloud Storage."""
        trainer = ModelTrainer()
        model, metrics = trainer.train(processed_data)
        self.storage.save_model(model, self.model_path)
        return metrics

    def load_model(self):
        """Loads a trained model from Google Cloud Storage."""
        return self.storage.load_model(self.model_path)
