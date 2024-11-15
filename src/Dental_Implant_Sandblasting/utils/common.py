import os
import yaml
import json
import joblib
import pandas as pd
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from Dental_Implant_Sandblasting import logger

# Exception handling for empty or invalid YAML files
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.

    Returns:
        ConfigBox: YAML content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("YAML file is empty.")
            logger.info(f"YAML file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty.")
    except Exception as e:
        logger.error(f"Error loading YAML file: {path_to_yaml} - {e}")
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates directories from a list of paths if they don't already exist.

    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, logs the directory creation.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dictionary as a JSON file.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to be saved in the JSON file.
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file at: {path} with error: {e}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads JSON data from a file and returns it as a ConfigBox.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: JSON content as a ConfigBox object.
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Failed to load JSON file from: {path} with error: {e}")
        return None

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data as a binary file.

    Args:
        data (Any): Data to save as a binary file.
        path (Path): Path to the binary file.
    """
    try:
        joblib.dump(value=data, filename=path)
        logger.info(f"Binary file saved at: {path}")
    except Exception as e:
        logger.error(f"Failed to save binary file at: {path} with error: {e}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads data from a binary file.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Data loaded from the binary file.
    """
    try:
        data = joblib.load(path)
        logger.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load binary file from: {path} with error: {e}")
        return None

@ensure_annotations
def get_size(path: Path) -> str:
    """Gets the size of a file in KB.

    Args:
        path (Path): Path to the file.

    Returns:
        str: File size in KB.
    """
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        logger.info(f"Size of the file at: {path} is ~ {size_in_kb} KB")
        return f"~ {size_in_kb} KB"
    except Exception as e:
        logger.error(f"Failed to get the size of the file at: {path} with error: {e}")
        return None

@ensure_annotations
def load_data(path: Path) -> pd.DataFrame:
    """Loads data from a CSV file and returns it as a pandas DataFrame.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Data loaded as a DataFrame.
    """
    try:
        data = pd.read_csv(path)
        logger.info(f"Data loaded successfully from: {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from: {path} with error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure
