import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        data = pd.read_csv(self.config.data_path)
        logger.info(f"Data loaded from {self.config.data_path}")
        logger.info(f"Data Head: \n{data.head()}")
        logger.info(f"Data Info: \n{data.info()}")
        logger.info(f"Data Description: \n{data.describe()}")
        return data

    def preprocess_data(self, data):
        # Convert columns to numeric, handling non-numeric entries as NaN
        data = data.apply(pd.to_numeric, errors='coerce')
        
        # Use KNN Imputer to handle missing values
        imputer = KNNImputer(n_neighbors=self.config.knn_n_neighbors)
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        logger.info("Missing values handled using KNN imputation")
        return data_imputed

    def feature_engineering(self, data):
        # Define feature and target columns
        feature_columns = [
            'angle_sandblasting',
            'pressure_sandblasting_bar',
            'temperature_acid_etching',
            'time_acid_etching_min',
            'voltage_anodizing_v',
            'time_anodizing_min'
        ]
        target_column_sa = 'sa_surface_roughness_micrometer'
        target_column_cv = 'cell_viability_percent'

        X = data[feature_columns]
        y_sa = data[target_column_sa]
        y_cv = data[target_column_cv]

        # Apply Polynomial Features
        poly = PolynomialFeatures(degree=self.config.polynomial_features_degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Standardize the features using RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_poly)

        # Dimensionality Reduction with PCA
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        logger.info(f"Number of components after PCA: {X_pca.shape[1]}")

        # Feature Selection using RFE with Lasso
        lasso_model = Lasso(alpha=0.01, max_iter=self.config.lasso_max_iter)
        
        # RFE for Surface Roughness (Sa)
        rfe_sa = RFE(lasso_model, n_features_to_select=10)
        X_sa_rfe = rfe_sa.fit_transform(X_pca, y_sa)

        # RFE for Cell Viability (CV)
        rfe_cv = RFE(lasso_model, n_features_to_select=10)
        X_cv_rfe = rfe_cv.fit_transform(X_pca, y_cv)

        logger.info(f"Number of features selected for Sa after RFE: {X_sa_rfe.shape[1]}")
        logger.info(f"Number of features selected for CV after RFE: {X_cv_rfe.shape[1]}")

        return X_sa_rfe, X_cv_rfe, y_sa, y_cv

    def train_test_splitting(self, X_sa_rfe, X_cv_rfe, y_sa, y_cv):
        # Split data into train and test sets for both Surface Roughness (Sa) and Cell Viability (CV)
        X_train_sa, X_test_sa, y_train_sa, y_test_sa = train_test_split(
            X_sa_rfe, y_sa, test_size=self.config.test_size, random_state=self.config.random_state
        )
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
            X_cv_rfe, y_cv, test_size=self.config.test_size, random_state=self.config.random_state
        )

        # Create directories if they don't exist
        os.makedirs(self.config.transformed_train_dir, exist_ok=True)
        os.makedirs(self.config.transformed_test_dir, exist_ok=True)

        # Save transformed datasets
        pd.DataFrame(X_train_sa).to_csv(self.config.transformed_train_dir / 'train_sa_target.csv', index=False)
        pd.DataFrame(X_train_cv).to_csv(self.config.transformed_train_dir / 'train_cv_target.csv', index=False)
        pd.DataFrame(X_test_sa).to_csv(self.config.transformed_test_dir / 'test_sa_target.csv', index=False)
        pd.DataFrame(X_test_cv).to_csv(self.config.transformed_test_dir / 'test_cv_target.csv', index=False)

        logger.info(f"Data saved to {self.config.transformed_train_dir} and {self.config.transformed_test_dir}")

    def execute(self):
        try:
            data = self.load_data()
            preprocessed_data = self.preprocess_data(data)
            X_sa_rfe, X_cv_rfe, y_sa, y_cv = self.feature_engineering(preprocessed_data)
            self.train_test_splitting(X_sa_rfe, X_cv_rfe, y_sa, y_cv)

            # Create status file indicating success
            with open(self.config.root_dir / "status.txt", "w") as f:
                f.write("Transformation status: True")

            logger.info("Data transformation and train-test splitting completed successfully.")

        except Exception as e:
            # Create status file indicating failure
            with open(self.config.root_dir / "status.txt", "w") as f:
                f.write("Transformation status: False")
            logger.exception(e)
            raise e
