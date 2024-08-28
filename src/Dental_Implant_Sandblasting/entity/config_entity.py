from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path                # Root directory for data ingestion artifacts
    source_URL: str               # URL for downloading the dataset
    local_data_file: Path         # Path to save the downloaded file
    unzip_dir: Path               # Directory to extract the dataset

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path                # Root directory for data validation artifacts
    STATUS_FILE: str              # Path to store the validation status file
    unzip_data_dir: Path          # Directory where the extracted data file is located
    all_schema: dict              # Dictionary holding schema definitions

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path                # Root directory for data transformation artifacts
    data_path: Path               # Path to the cleaned and validated data file
    transformed_train_dir: Path   # Directory to save the transformed training data
    transformed_test_dir: Path    # Directory to save the transformed testing data
    test_size: float              # Proportion of data to be used as the test set
    random_state: int             # Random state for reproducibility
    polynomial_features_degree: int  # Degree for polynomial feature generation
    scaling_method: str           # Method used for scaling features (e.g., 'standard')
    lasso_max_iter: int           # Maximum iterations for Lasso regression during feature selection

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
