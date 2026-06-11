from pathlib import Path

import pandas as pd

from src.utils import load_pickle, read_yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def predict_house_price(input_data):
    """
    Predict house price using the trained model and preprocessor.

    Args:
        input_data (dict): Dictionary containing house feature values.

    Returns:
        float: Predicted median house value rounded to two decimal places.
    """
    config = read_yaml(CONFIG_PATH)

    model_path = ROOT_DIR / config["model"]["model_path"]
    preprocessor_path = ROOT_DIR / config["model"]["preprocessor_path"]

    numerical_features = config["features"]["numerical_features"]
    categorical_features = config["features"]["categorical_features"]

    feature_columns = numerical_features + categorical_features

    model = load_pickle(model_path)
    preprocessor = load_pickle(preprocessor_path)

    input_df = pd.DataFrame([input_data], columns=feature_columns)

    processed_input = preprocessor.transform(input_df)

    prediction = model.predict(processed_input)

    final_prediction = round(float(prediction[0]), 2)

    return final_prediction


if __name__ == "__main__":
    sample_input = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }

    result = predict_house_price(sample_input)

    print(f"Predicted House Price: {result}")