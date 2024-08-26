from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True  # Initialize to True assuming validation will pass

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            # Combine schema columns and target columns into a single list
            all_schema = list(self.config.all_schema.keys())

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False  # Set to False if any column is not found in the schema
                    logger.error(f"Column {col} not found in schema")
                    break  # Stop further checks if a mismatch is found
                else:
                    logger.info(f"Column {col} is valid")

            # Write the validation status to the status file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status  # Return the final validation status

        except Exception as e:
            logger.exception(e)
            raise e

    def preprocess_data(self):
        try:
            # Load the dataset
            data = pd.read_csv(self.config.unzip_data_dir)

            # Check for missing values
            missing_values = data.isnull().sum()
            logger.info(f"Missing values:\n{missing_values}")

            # Convert appropriate columns to numeric, forcing errors to NaN
            cols_to_convert = [
                'Angle of Sandblasting', 
                'Pressure of Sandblasting (bar)', 
                'Temperture of Acid Etching',
                'Time of Acid Etching (min)',
                'Voltage of Anodizing (v)', 
                'Time of  Anodizing (min)', 
                '(Sa) Average of Surface roughness (micrometer)', 
                'Cell Viability (%)'
            ]
            data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

            # Impute missing values using the mean for numeric columns
            data_imputed = data.copy()
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_imputed[numeric_cols] = data_imputed[numeric_cols].fillna(data_imputed[numeric_cols].mean())

            # Verify data after imputation
            logger.info(f"Data after imputing missing values:\n{data_imputed.info()}")

            # Ensure that the dataset is not empty after imputation
            if data_imputed.empty:
                raise ValueError("Dataset is empty after imputing missing values.")

            # Filter data according to the given validation ranges for Surface Roughness (Sa)
            valid_data = data_imputed[
                (data_imputed['(Sa) Average of Surface roughness (micrometer)'] > 1.5) & 
                (data_imputed['(Sa) Average of Surface roughness (micrometer)'] < 2.5)
            ]

            # Set "Cell Viability (%)" to 0 where Sa is outside the valid range
            data_imputed.loc[~data_imputed.index.isin(valid_data.index), 'Cell Viability (%)'] = 0

            # Handle outliers in Cell Viability
            def cap_outliers(series):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return np.clip(series, lower_bound, upper_bound)

            y_cv_capped = cap_outliers(data_imputed['Cell Viability (%)'])

            # Standardize the features
            scaler = StandardScaler()
            feature_columns = [
                'Angle of Sandblasting', 
                'Pressure of Sandblasting (bar)', 
                'Temperture of Acid Etching', 
                'Time of Acid Etching (min)', 
                'Voltage of Anodizing (v)', 
                'Time of  Anodizing (min)'
            ]
            X = data_imputed[feature_columns]
            X_scaled = scaler.fit_transform(X)

            # Split the data into training and testing sets for Surface Roughness (Sa) and Cell Viability (CV)
            y_sa = data_imputed['(Sa) Average of Surface roughness (micrometer)']
            X_train, X_test, y_sa_train, y_sa_test = train_test_split(X_scaled, y_sa, test_size=0.2, random_state=42)
            _, _, y_cv_train, y_cv_test = train_test_split(X_scaled, y_cv_capped, test_size=0.2, random_state=42)

            logger.info(f"Training set size for Surface Roughness (Sa): {X_train.shape}")
            logger.info(f"Testing set size for Surface Roughness (Sa): {X_test.shape}")
            logger.info(f"Training set size for Cell Viability (CV): {y_cv_train.shape}")
            logger.info(f"Testing set size for Cell Viability (CV): {y_cv_test.shape}")

            # Check for any inconsistencies in the data split
            if X_train.shape[0] != y_sa_train.shape[0] or X_train.shape[0] != y_cv_train.shape[0]:
                raise ValueError("Mismatch in the number of training samples between features and targets.")
            if X_test.shape[0] != y_sa_test.shape[0] or X_test.shape[0] != y_cv_test.shape[0]:
                raise ValueError("Mismatch in the number of testing samples between features and targets.")

        except Exception as e:
            logger.exception(e)
            raise e
