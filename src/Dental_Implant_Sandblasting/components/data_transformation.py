import os
import pandas as pd
import numpy as np
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        data = pd.read_csv(self.config.data_path)
        logger.info(f"Data loaded from {self.config.data_path}")
        return data

    def preprocess_data(self, data):
        # Convert columns to numeric, forcing any errors to NaN
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Handle missing values by imputing
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data))
        data_imputed.columns = data.columns
        
        logger.info("Missing values handled")
        return data_imputed

    def feature_engineering(self, data):
        # Generating polynomial features
        poly = PolynomialFeatures(degree=self.config.polynomial_features_degree, include_bias=False)
        poly_features = poly.fit_transform(data.drop(columns=['(Sa) Average of Surface roughness (micrometer)', 'Cell Viability (%)', 'Result (1=Passed, 0=Failed)']))
        
        # Scaling features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(poly_features)
        
        # Create a DataFrame with the new features
        feature_columns = poly.get_feature_names_out(data.columns[:-3])
        engineered_data = pd.DataFrame(scaled_features, columns=feature_columns)
        
        # Add target columns back to the DataFrame
        engineered_data['(Sa) Average of Surface roughness (micrometer)'] = data['(Sa) Average of Surface roughness (micrometer)'].values
        engineered_data['Cell Viability (%)'] = data['Cell Viability (%)'].values
        engineered_data['Result (1=Passed, 0=Failed)'] = data['Result (1=Passed, 0=Failed)'].values
        
        logger.info("Feature engineering completed")
        return engineered_data

    def train_test_splitting(self, data):
        train, test = train_test_split(data, test_size=self.config.test_size, random_state=self.config.random_state)

        train.to_csv(self.config.transformed_train_dir, index=False)
        test.to_csv(self.config.transformed_test_dir, index=False)

        logger.info(f"Train-test split completed with train shape: {train.shape} and test shape: {test.shape}")
        print(train.shape)
        print(test.shape)
