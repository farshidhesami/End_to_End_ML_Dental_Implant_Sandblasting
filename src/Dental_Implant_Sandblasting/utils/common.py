import os
import yaml
import json
import joblib
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from Dental_Implant_Sandblasting import logger

# Exception handling for empty or invalid YAML files
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        e: Other exceptions related to file operations.

    Returns:
        ConfigBox: Contents of the YAML file as a ConfigBox.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty.")
    except Exception as e:
        logger.error(f"Error loading YAML file: {e}")
        raise e
    

# Creates each directory in the list, if it doesn't already exist
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create a list of directories if they don't exist.

    Args:
        path_to_directories (list): List of paths to directories.
        verbose (bool, optional): If True, logs the creation of each directory. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

# Handles exceptions during JSON file writing
@ensure_annotations
def save_json(path: Path, data: dict):
    """Save data to a JSON file.

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

# Exception handling for loading JSON files
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load data from a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data from the JSON file as a ConfigBox.
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Failed to load JSON file from: {path} with error: {e}")
        return None

# Saves any serializable data as a binary file
@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save data as a binary file.

    Args:
        data (Any): Data to be saved as binary.
        path (Path): Path to the binary file.
    """
    try:
        joblib.dump(value=data, filename=path)
        logger.info(f"Binary file saved at: {path}")
    except Exception as e:
        logger.error(f"Failed to save binary file at: {path} with error: {e}")

# Handles exceptions during binary file loading
@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load data from a binary file.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Object stored in the file.
    """
    try:
        data = joblib.load(path)
        logger.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load binary file from: {path} with error: {e}")
        return None

# Exception handling for getting file size
@ensure_annotations
def get_size(path: Path) -> str:
    """Get the size of a file in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size in KB.
    """
    try:
        size_in_kb = round(os.path.getsize(path) / 1024)
        logger.info(f"Size of the file at: {path} is ~ {size_in_kb} KB")
        return f"~ {size_in_kb} KB"
    except Exception as e:
        logger.error(f"Failed to get the size of the file at: {path} with error: {e}")
        return None
