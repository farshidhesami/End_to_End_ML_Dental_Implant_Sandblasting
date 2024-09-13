from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import DataValidationConfig


### Class for validating and preprocessing the data:
class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True

            csv_path = str(self.config.unzip_data_dir)  # Convert the Path to a string
            logger.info(f"Reading CSV file from: {csv_path}")

            data = pd.read_csv(csv_path)  # Load the CSV file based on the updated path
            all_cols = list(data.columns)
            all_schema = list(self.config.all_schema.keys())

            logger.info(f"Checking columns in the dataset: {all_cols}")
            logger.info(f"Expected schema columns: {all_schema}")

            # Check if any expected columns are missing
            missing_cols = [col for col in all_schema if col not in all_cols]
            if missing_cols:
                validation_status = False
                logger.error(f"Missing columns in data: {missing_cols}")

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    logger.error(f"Unexpected column {col} not found in schema")
                else:
                    logger.info(f"Column {col} is valid")

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            logger.exception(f"Error during column validation: {e}")
            raise e

    def preprocess_data(self):
        try:
            # Load the dataset
            csv_path = str(self.config.unzip_data_dir)  # Convert the Path to a string
            logger.info(f"Loading dataset from {csv_path}")
            data = pd.read_csv(csv_path)

            # Check for missing values
            missing_values = data.isnull().sum()
            logger.info(f"Missing values:\n{missing_values}")

            # Convert columns to numeric, forcing errors to NaN
            cols_to_convert = self.config.columns_to_convert
            logger.info(f"Converting columns {cols_to_convert} to numeric.")
            data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

            # Advanced Imputation using KNN Imputer
            logger.info(f"Imputing missing values using KNN with {self.config.knn_n_neighbors} neighbors.")
            imputer = KNNImputer(n_neighbors=self.config.knn_n_neighbors)
            data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

            logger.info(f"Data after imputing missing values:\n{data_imputed.info()}")

            if data_imputed.empty:
                raise ValueError("Dataset is empty after imputing missing values.")

            # Filter data according to validation ranges for Surface Roughness (Sa)
            valid_data = data_imputed[
                (data_imputed['sa_surface_roughness_micrometer'] > self.config.sa_lower_bound) & 
                (data_imputed['sa_surface_roughness_micrometer'] < self.config.sa_upper_bound)
            ]

            data_imputed.loc[~data_imputed.index.isin(valid_data.index), 'cell_viability_percent'] = 0

            # Separate features and target variables
            feature_columns = self.config.feature_columns
            target_column_sa = self.config.target_column_sa
            target_column_cv = self.config.target_column_cv

            X = data_imputed[feature_columns]
            y_sa = data_imputed[target_column_sa]
            y_cv = data_imputed[target_column_cv]

            # Normalize or standardize features using RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, data_imputed[[target_column_sa, target_column_cv]], test_size=self.config.test_size, random_state=self.config.random_state)

            y_sa_train = y_train[target_column_sa]
            y_sa_test = y_test[target_column_sa]
            y_cv_train = y_train[target_column_cv]
            y_cv_test = y_test[target_column_cv]

            logger.info(f"Training set size for Surface Roughness (Sa): {X_train.shape}")
            logger.info(f"Testing set size for Surface Roughness (Sa): {X_test.shape}")
            logger.info(f"Training set size for Cell Viability (CV): {y_cv_train.shape}")
            logger.info(f"Testing set size for Cell Viability (CV): {y_cv_test.shape}")

            if X_train.shape[0] != y_sa_train.shape[0] or X_train.shape[0] != y_cv_train.shape[0]:
                raise ValueError("Mismatch in the number of training samples between features and targets.")
            if X_test.shape[0] != y_sa_test.shape[0] or X_test.shape[0] != y_cv_test.shape[0]:
                raise ValueError("Mismatch in the number of testing samples between features and targets.")

        except Exception as e:
            logger.exception(f"Error during data preprocessing: {e}")
            raise e

