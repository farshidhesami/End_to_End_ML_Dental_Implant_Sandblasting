import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import DataTransformationConfig

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

        # Check for multicollinearity using VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = engineered_data.columns
        vif_data["VIF"] = [variance_inflation_factor(engineered_data.values, i) for i in range(len(engineered_data.columns))]
        logger.info(f"\nVIF before feature selection:\n{vif_data}")

        # Feature selection using Lasso with increased iterations
        lasso = LassoCV(max_iter=10000)
        lasso.fit(scaled_features, data['(Sa) Average of Surface roughness (micrometer)'])
        model = SelectFromModel(lasso, prefit=True)
        X_selected = model.transform(scaled_features)

        selected_features = np.array(feature_columns)[model.get_support()]
        logger.info(f"Selected Features:\n{selected_features}")

        return pd.DataFrame(X_selected, columns=selected_features), data[['(Sa) Average of Surface roughness (micrometer)', 'Cell Viability (%)', 'Result (1=Passed, 0=Failed)']]

    def train_test_splitting(self, features, targets):
        train_features, test_features, train_targets, test_targets = train_test_split(
            features, targets, test_size=self.config.test_size, random_state=self.config.random_state)

        train_data = pd.concat([train_features, train_targets.reset_index(drop=True)], axis=1)
        test_data = pd.concat([test_features, test_targets.reset_index(drop=True)], axis=1)

        train_data.to_csv(self.config.transformed_train_dir, index=False)
        test_data.to_csv(self.config.transformed_test_dir, index=False)

        logger.info(f"Train-test split completed with train shape: {train_data.shape} and test shape: {test_data.shape}")
        print(train_data.shape)
        print(test_data.shape)

    def execute(self):
        data = self.load_data()
        preprocessed_data = self.preprocess_data(data)
        features, targets = self.feature_engineering(preprocessed_data)
        self.train_test_splitting(features, targets)
