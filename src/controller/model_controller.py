from model.model_training import train_model, evaluate_model

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrena y evalúa el modelo."""
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, y_train, "Training")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")