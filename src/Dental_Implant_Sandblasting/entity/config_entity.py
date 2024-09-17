from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path                # Root directory for data ingestion artifacts
    source_URL: str               # URL for downloading the dataset
    local_data_file: Path         # Path to save the downloaded file
    unzip_dir: Path               # Directory to extract the dataset
    columns_to_convert: list      # List of columns to convert to numeric
    sa_lower_bound: float         # Lower bound for surface roughness
    sa_upper_bound: float         # Upper bound for surface roughness
    cell_viability_threshold: int # Threshold for cell viability to determine pass/fail
    outlier_capping_method: str   # Method for capping outliers (e.g., 'quantiles')
    outlier_tail: str             # Tail(s) to apply the outlier capping (e.g., 'both')
    outlier_fold: float           # Fold for Winsorizing
    log_transform_variable: str   # Variable to apply log transformation

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path                # Root directory for data validation artifacts
    STATUS_FILE: str              # Path to save validation status
    unzip_data_dir: Path          # Directory containing the extracted dataset
    all_schema: dict              # Schema dictionary for column validation
    columns_to_convert: list      # Columns to be converted to numeric
    knn_n_neighbors: int          # Number of neighbors for KNN imputation
    sa_lower_bound: float         # Lower bound for surface roughness
    sa_upper_bound: float         # Upper bound for surface roughness
    feature_columns: list         # Feature columns for modeling
    target_column_sa: str         # Target column for Surface Roughness (Sa)
    target_column_cv: str         # Target column for Cell Viability (CV)
    test_size: float              # Test size for train/test split
    random_state: int             # Random state for reproducibility

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path                # Root directory for data transformation artifacts
    data_path: Path               # Path to the dataset
    transformed_train_dir: Path   # Directory to save the transformed training data
    transformed_test_dir: Path    # Directory to save the transformed testing data
    test_size: float              # Test size for train-test split
    random_state: int             # Random state for reproducibility
    polynomial_features_degree: int  # Degree for polynomial features
    scaling_method: str           # Method to scale the data (e.g., 'RobustScaler')
    lasso_max_iter: int           # Maximum iterations for Lasso regression
    knn_n_neighbors: int          # Number of neighbors for KNN imputation
