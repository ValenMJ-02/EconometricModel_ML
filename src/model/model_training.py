import sys
import os
from sklearn.linear_model import LinearRegression
# Add the parent directory to sys.path to allow module imports from the parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

  # Import Linear Regression model from scikit-learn

def train_model(X_train, train_target):
    """
    Trains a Linear Regression model using the provided training data.

    Args:
        X_train (pd.DataFrame or np.array): The feature matrix used for training.
        train_target (pd.Series or np.array): The target variable corresponding to the training data.

    Returns:
        model (LinearRegression): The trained Linear Regression model.
    """
    # Create an instance of the Linear Regression model
    model = LinearRegression()

    # Fit (train) the model using the training data and target values
    model.fit(X_train, train_target)

    return model  # Return the trained model

def evaluate_model(model, X, y, dataset_name):
    """
    Evaluates the trained model using a given dataset and prints the performance score.

    Args:
        model (LinearRegression): The trained model.
        X (pd.DataFrame or np.array): The feature matrix of the dataset to be evaluated.
        y (pd.Series or np.array): The true target values for the dataset.
        dataset_name (str): A string indicating the dataset being evaluated (e.g., "Training", "Validation", "Test").

    Returns:
        None
    """
    # Calculate the model's score (R² coefficient of determination)
    score = model.score(X, y)

    # Print the evaluation score with two decimal places
    print(f"{dataset_name} set score: {score:.2f}")
