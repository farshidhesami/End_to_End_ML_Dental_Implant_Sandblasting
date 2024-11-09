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
    root_dir: Path                      # Root directory for data transformation artifacts
    data_path: Path                     # Path to the dataset
    transformed_train_dir: Path         # Directory to save the transformed training data
    transformed_test_dir: Path          # Directory to save the transformed testing data
    test_size: float                    # Test size for train-test split
    random_state: int                   # Random state for reproducibility
    polynomial_features_degree: int     # Degree for polynomial feature expansion
    scaling_method: str                 # Scaling method (e.g., 'RobustScaler')
    lasso_max_iter: int                 # Maximum iterations for Lasso regression
    knn_n_neighbors: int                # Number of neighbors for KNN imputation

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path                      # Root directory for model trainer artifacts
    transformed_train_dir: Path         # Directory containing transformed training data
    transformed_test_dir: Path          # Directory containing transformed testing data
    test_size: float                    # Test size for train/test split
    random_state: int                   # Random state for reproducibility
    param_grid_rf: dict                 # Parameter grid for RandomForest
    param_grid_bagging: dict            # Parameter grid for Bagging Regressor
    param_grid_ridge: dict              # Parameter grid for Ridge regression
    models: dict                        # Dictionary containing model names and configurations
    n_iter: int                         # Number of iterations for RandomizedSearchCV
    cv: int                             # Number of cross-validation folds
    verbose: int                        # Verbosity level for RandomizedSearchCV
    n_jobs: int                         # Number of parallel jobs for RandomizedSearchCV

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path                      # Root directory for model evaluation artifacts
    model_dir: Path                     # Directory where models are saved
    test_sa_data: Path                  # Path to test data for Surface Roughness (Sa)
    test_cv_data: Path                  # Path to test data for Cell Viability (CV)
    n_estimators_rf: int                # Number of trees for RandomForest
    max_depth_rf: int                   # Maximum depth for RandomForest
    min_samples_split_rf: int           # Minimum samples required to split for RandomForest
    min_samples_leaf_rf: int            # Minimum samples required at a leaf for RandomForest
    bootstrap_rf: bool                  # Whether to bootstrap samples for RandomForest
    alpha_ridge: float                  # Alpha parameter for Ridge regression
    n_estimators_bagging: int           # Number of estimators for Bagging Regressor
    max_samples_bagging: float          # Maximum samples to use in Bagging Regressor
    max_features_bagging: float         # Maximum features to use in Bagging Regressor
    random_state: int                   # Random state for reproducibility
