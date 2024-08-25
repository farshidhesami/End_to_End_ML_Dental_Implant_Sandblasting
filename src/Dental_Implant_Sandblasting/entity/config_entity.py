# Config entity for data ingestion process (data ingestion config)
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path          # Root directory for data ingestion artifacts
    source_URL: str         # URL for downloading the dataset
    local_data_file: Path   # Path to save the downloaded file
    unzip_dir: Path         # Directory to extract the dataset


@dataclass(frozen=True)
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

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float
    random_state: int
    models: dict
    param_grids: dict
    target_column: str
    cv: int
    scoring: str
    poly_features_degree: int
    poly_features_path: Path
    model_path: Path
    imputation_strategy: str
    scaling_method: str
    sa_model_name: str
    cv_model_name: str
