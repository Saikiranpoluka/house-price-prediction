from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import read_yaml, save_pickle


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_processed_data(file_path):
    """
    Load the processed housing dataset.

    Args:
        file_path (str or Path): Path of the processed CSV file.

    Returns:
        pd.DataFrame: Loaded processed dataset.
    """
    return pd.read_csv(file_path)


def split_features_target(df, numerical_features, categorical_features, target_column):
    """
    Split the dataset into input features and target variable.

    Args:
        df (pd.DataFrame): Housing dataset.
        numerical_features (list): Numerical feature column names.
        categorical_features (list): Categorical feature column names.
        target_column (str): Target column name.

    Returns:
        tuple: X features DataFrame and y target Series.
    """
    feature_columns = numerical_features + categorical_features

    X = df[feature_columns]
    y = df[target_column]

    return X, y


def build_preprocessor(numerical_features, categorical_features):
    """
    Build a preprocessing pipeline for numerical and categorical features.

    Numerical pipeline:
        - Fill missing values using median.
        - Scale values using StandardScaler.

    Categorical pipeline:
        - Fill missing values using most frequent value.
        - Convert categories into numbers using OneHotEncoder.

    Args:
        numerical_features (list): Numerical feature column names.
        categorical_features (list): Categorical feature column names.

    Returns:
        ColumnTransformer: Complete preprocessing transformer.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def preprocess_data():
    """
    Run the preprocessing pipeline.

    Steps:
        1. Read configuration from config.yaml.
        2. Load processed housing dataset.
        3. Split data into X and y.
        4. Split X and y into train and test sets.
        5. Create preprocessing pipeline.
        6. Fit preprocessor on training data.
        7. Transform training and testing data.
        8. Save fitted preprocessor as preprocessor.pkl.

    Returns:
        tuple: X_train_processed, X_test_processed, y_train, y_test
    """
    config = read_yaml(CONFIG_PATH)

    processed_data_path = ROOT_DIR / config["data"]["processed_data_path"]
    preprocessor_path = ROOT_DIR / config["model"]["preprocessor_path"]

    numerical_features = config["features"]["numerical_features"]
    categorical_features = config["features"]["categorical_features"]
    target_column = config["target"]["name"]

    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]

    df = load_processed_data(processed_data_path)

    X, y = split_features_target(
        df=df,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target_column,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    save_pickle(preprocessor_path, preprocessor)

    print("Preprocessing completed successfully.")
    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    preprocess_data()