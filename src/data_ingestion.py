from pathlib import Path

import pandas as pd

from src.utils import read_yaml, create_directories


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str or Path): Path of the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def validate_data(df):
    """
    Validate the loaded housing dataset.

    This function checks whether the dataset is empty and whether
    all required columns are present.

    Args:
        df (pd.DataFrame): Housing dataset.

    Returns:
        None

    Raises:
        ValueError: If dataset is empty or required columns are missing.
    """
    required_columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",
        "ocean_proximity",
    ]

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")


def ingest_data():
    """
    Run the data ingestion process.

    Steps:
        1. Read configuration from config.yaml.
        2. Load raw housing dataset.
        3. Validate required columns.
        4. Create processed data folder if missing.
        5. Save dataset into processed folder.

    Returns:
        pd.DataFrame: Ingested housing dataset.
    """
    config = read_yaml(CONFIG_PATH)

    raw_data_path = ROOT_DIR / config["data"]["raw_data_path"]
    processed_data_path = ROOT_DIR / config["data"]["processed_data_path"]

    df = load_data(raw_data_path)
    validate_data(df)

    create_directories([processed_data_path.parent])

    df.to_csv(processed_data_path, index=False)

    print("Data ingestion completed successfully.")
    print(f"Raw data path: {raw_data_path}")
    print(f"Processed data saved at: {processed_data_path}")
    print(f"Dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    ingest_data()