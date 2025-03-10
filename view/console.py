from src.controller.handler import RealEstateHandler

def main():
    """Console interface to interact with the real estate model."""
    BUCKET_NAME = "your-gcs-bucket"
    DATASET_PATH = "data/real_estate.csv"
    MODEL_PATH = "models/real_estate_model.pkl"

    handler = RealEstateHandler(BUCKET_NAME, DATASET_PATH, MODEL_PATH)

    print("\n📊 Loading and preprocessing data...")
    data, _ = handler.load_and_preprocess_data()
    print("✅ Data loaded and processed!")

    print("\n🧠 Training the model...")
    metrics = handler.train_and_store_model(data)
    print("✅ Model trained and stored in Google Cloud!")

    print("\n📈 Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
