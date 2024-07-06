# Config entity for data ingestion process (data ingestion config)
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)             # from 01_data_ingestion.py
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)              # from 02_data_validation.py
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    transformed_train_dir: Path
    transformed_test_dir: Path
    test_size: float
    random_state: int
    polynomial_features_degree: int
    scaling_method: str


