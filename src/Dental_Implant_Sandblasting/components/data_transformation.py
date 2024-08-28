import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import DataTransformationConfig

### Data Transformation class
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        data = pd.read_csv(self.config.data_path)
        logger.info(f"Data loaded from {self.config.data_path}")

        # Basic Data Exploration
        logger.info(f"Data Head: \n{data.head()}")
        logger.info(f"Data Info: \n{data.info()}")
        logger.info(f"Data Description: \n{data.describe()}")

        return data

    def preprocess_data(self, data):
        # Convert columns to numeric, forcing any errors to NaN
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Handle missing values by imputing
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data_imputed = data.copy()
        data_imputed[numeric_cols] = data_imputed[numeric_cols].fillna(data_imputed[numeric_cols].mean())

        logger.info("Missing values handled")
        return data_imputed

    def feature_engineering(self, data):
        # Define feature and target columns
        feature_columns = [
            'Angle of Sandblasting', 
            'Pressure of Sandblasting (bar)', 
            'Temperture of Acid Etching', 
            'Time of Acid Etching (min)', 
            'Voltage of Anodizing (v)', 
            'Time of  Anodizing (min)'
        ]
        target_column_sa = '(Sa) Average of Surface roughness (micrometer)'
        target_column_cv = 'Cell Viability (%)'

        X = data[feature_columns]
        y_sa = data[target_column_sa]
        y_cv = data[target_column_cv]

        # Apply PolynomialFeatures
        poly = PolynomialFeatures(degree=self.config.polynomial_features_degree, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(X)

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)

        # Feature Selection using Lasso (as Lasso inherently performs feature selection)
        lasso_sa = Lasso(alpha=0.01, max_iter=self.config.lasso_max_iter)
        lasso_sa.fit(X_scaled, y_sa)
        coef_lasso_sa = lasso_sa.coef_

        lasso_cv = Lasso(alpha=0.01, max_iter=self.config.lasso_max_iter)
        lasso_cv.fit(X_scaled, y_cv)
        coef_lasso_cv = lasso_cv.coef_

        # Selecting the top features based on Lasso
        threshold = 0.01  # Adjust this threshold based on model tuning
        selected_features_sa = np.where(np.abs(coef_lasso_sa) > threshold)[0]
        selected_features_cv = np.where(np.abs(coef_lasso_cv) > threshold)[0]

        X_selected_sa = X_scaled[:, selected_features_sa]
        X_selected_cv = X_scaled[:, selected_features_cv]

        logger.info(f"Number of features selected for Sa: {X_selected_sa.shape[1]}")
        logger.info(f"Number of features selected for CV: {X_selected_cv.shape[1]}")

        return X_selected_sa, X_selected_cv, y_sa, y_cv

    def train_test_splitting(self, X_selected_sa, X_selected_cv, y_sa, y_cv):
        # Split the data into training and testing sets for Surface Roughness (Sa) and Cell Viability (CV)
        X_train_sa, X_test_sa, y_sa_train, y_sa_test = train_test_split(X_selected_sa, y_sa, test_size=self.config.test_size, random_state=self.config.random_state)
        X_train_cv, X_test_cv, y_cv_train, y_cv_test = train_test_split(X_selected_cv, y_cv, test_size=self.config.test_size, random_state=self.config.random_state)

        # Ensure directories exist before saving the files
        os.makedirs(self.config.transformed_train_dir, exist_ok=True)
        os.makedirs(self.config.transformed_test_dir, exist_ok=True)

        # Save the transformed datasets
        train_data_sa = pd.DataFrame(X_train_sa)
        train_data_cv = pd.DataFrame(X_train_cv)
        train_data_sa.to_csv(self.config.transformed_train_dir / 'train_sa.csv', index=False)
        train_data_cv.to_csv(self.config.transformed_train_dir / 'train_cv.csv', index=False)
        logger.info(f"Training data saved: Sa - {train_data_sa.shape}, CV - {train_data_cv.shape}")

        test_data_sa = pd.DataFrame(X_test_sa)
        test_data_cv = pd.DataFrame(X_test_cv)
        test_data_sa.to_csv(self.config.transformed_test_dir / 'test_sa.csv', index=False)
        test_data_cv.to_csv(self.config.transformed_test_dir / 'test_cv.csv', index=False)
        logger.info(f"Testing data saved: Sa - {test_data_sa.shape}, CV - {test_data_cv.shape}")

    def execute(self):
        try:
            data = self.load_data()
            preprocessed_data = self.preprocess_data(data)
            X_selected_sa, X_selected_cv, y_sa, y_cv = self.feature_engineering(preprocessed_data)
            self.train_test_splitting(X_selected_sa, X_selected_cv, y_sa, y_cv)

            # Create status file
            with open(self.config.root_dir / "status.txt", "w") as f:
                f.write("Transformation status: True")

            logger.info("Data transformation and splitting completed successfully.")
        except Exception as e:
            # Create status file with failure status
            with open(self.config.root_dir / "status.txt", "w") as f:
                f.write("Transformation status: False")

            logger.exception(e)
            raise e
