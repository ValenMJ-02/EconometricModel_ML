import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model.train import train_model
from model.predict import predict_price


if __name__ == "__main__":
    train_model()
    town = input("Ingrese la ciudad para predecir el precio: ")
    predictions = predict_price(town)
    print(predictions)
