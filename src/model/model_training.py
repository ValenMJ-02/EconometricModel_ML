import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sklearn.linear_model import LinearRegression

def train_model(X_train, train_target):
    """Entrena un modelo de regresión lineal."""
    model = LinearRegression().fit(X_train, train_target)
    return model

def evaluate_model(model, X, y, dataset_name):
    """Evalúa el modelo y muestra el puntaje."""
    score = model.score(X, y)
    print(f"{dataset_name} set score: {score:.2f}")