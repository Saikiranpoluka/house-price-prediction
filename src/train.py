from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import preprocess_data
from src.utils import read_yaml, save_pickle


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def evaluate_model(y_true, y_pred):
    """
    Evaluate regression model performance.

    Args:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.

    Returns:
        dict: Model evaluation metrics including R2, MAE, MSE, and RMSE.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "r2_score": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }

    return metrics


def train_model():
    """
    Train the house price prediction model.

    Steps:
        1. Read configuration from config.yaml.
        2. Run preprocessing pipeline.
        3. Train Linear Regression model.
        4. Predict on test data.
        5. Evaluate model performance.
        6. Save trained model as model.pkl.

    Returns:
        dict: Evaluation metrics of the trained model.
    """
    config = read_yaml(CONFIG_PATH)

    model_path = ROOT_DIR / config["model"]["model_path"]

    X_train_processed, X_test_processed, y_train, y_test = preprocess_data()

    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)

    metrics = evaluate_model(y_test, y_pred)

    save_pickle(model_path, model)

    print("Model training completed successfully.")
    print(f"Model saved at: {model_path}")
    print(f"R2 Score: {metrics['r2_score']:.4f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")

    return metrics


if __name__ == "__main__":
    train_model()