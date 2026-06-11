import pickle
from pathlib import Path

import yaml


def read_yaml(file_path):
    """
    Read a YAML configuration file.

    Args:
        file_path (str or Path): Path of the YAML file.

    Returns:
        dict: Configuration data loaded from the YAML file.
    """
    file_path = Path(file_path)

    with open(file_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config


def create_directories(paths):
    """
    Create directories if they do not already exist.

    Args:
        paths (list): List of folder paths to create.

    Returns:
        None
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_pickle(file_path, obj):
    """
    Save a Python object into a pickle file.

    Args:
        file_path (str or Path): Location where the pickle file should be saved.
        obj (object): Python object to save.

    Returns:
        None
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(file_path):
    """
    Load a Python object from a pickle file.

    Args:
        file_path (str or Path): Path of the pickle file.

    Returns:
        object: Loaded Python object.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)