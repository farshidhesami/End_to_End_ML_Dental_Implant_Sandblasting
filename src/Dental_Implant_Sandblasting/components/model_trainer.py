import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    # Custom evaluation metrics
    def smape(self, y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-10))

    def mape(self, y_true, y_pred):
        return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

    def load_data(self):
        try:
            # Define paths to the training and testing data
            train_sa_path = self.config.transformed_train_dir / 'train_sa_target.csv'
            test_sa_path = self.config.transformed_test_dir / 'test_sa_target.csv'
            train_cv_path = self.config.transformed_train_dir / 'train_cv_target.csv'
            test_cv_path = self.config.transformed_test_dir / 'test_cv_target.csv'

            # Load data
            train_sa_data = pd.read_csv(train_sa_path)
            test_sa_data = pd.read_csv(test_sa_path)
            train_cv_data = pd.read_csv(train_cv_path)
            test_cv_data = pd.read_csv(test_cv_path)

            # Extracting feature columns
            X_train_sa = train_sa_data.iloc[:, :-1]  # Assuming the last column is the target
            y_train_sa = train_sa_data.iloc[:, -1]
            X_test_sa = test_sa_data.iloc[:, :-1]
            y_test_sa = test_sa_data.iloc[:, -1]

            X_train_cv = train_cv_data.iloc[:, :-1]  # Assuming the last column is the target
            y_train_cv = train_cv_data.iloc[:, -1]
            X_test_cv = test_cv_data.iloc[:, :-1]
            y_test_cv = test_cv_data.iloc[:, -1]

            return (X_train_sa, y_train_sa, X_test_sa, y_test_sa), (X_train_cv, y_train_cv, X_test_cv, y_test_cv)
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            raise e

    def train_models(self, X_train, y_train, X_test, y_test, model_name):
        try:
            models = {
                "RandomForest": RandomForestRegressor(random_state=self.config.random_state),
                "BaggingRF": BaggingRegressor(estimator=RandomForestRegressor(random_state=self.config.random_state), random_state=self.config.random_state),
                "Ridge": Ridge()
            }

            model_performance = {}

            # Ensure the directory for models exists
            model_trainer_dir = self.config.root_dir
            os.makedirs(model_trainer_dir, exist_ok=True)

            for name, model in models.items():
                logger.info(f"Training {name} model for {model_name}...")
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Calculate evaluation metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape_val = self.mape(y_test, y_pred)
                smape_val = self.smape(y_test, y_pred)

                model_performance[name] = {
                    "MAE": mae,
                    "MSE": mse,
                    "R²": r2,
                    "MAPE": mape_val,
                    "SMAPE": smape_val
                }

                logger.info(f"{name} - MAE: {mae:.4f}, R²: {r2:.4f}")

                # Save the model
                joblib.dump(model, model_trainer_dir / f"{name}_{model_name}.joblib")

            return model_performance
        except Exception as e:
            logger.exception(f"Error training models: {e}")
            raise e

    def hyperparameter_tuning(self, X_train, y_train, model_name):
        try:
            if model_name == "RandomForest":
                param_grid = self.config.param_grid_rf
                model = RandomForestRegressor(random_state=self.config.random_state)
            elif model_name == "BaggingRF":
                param_grid = self.config.param_grid_bagging
                model = BaggingRegressor(estimator=RandomForestRegressor(random_state=self.config.random_state), random_state=self.config.random_state)
            elif model_name == "Ridge":
                param_grid = self.config.param_grid_ridge
                model = Ridge()

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.config.n_iter,
                cv=self.config.cv,
                verbose=self.config.verbose,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_
            logger.info(f"Best parameters for {model_name}: {random_search.best_params_}")

            return best_model
        except Exception as e:
            logger.exception(f"Error during hyperparameter tuning for {model_name}: {e}")
            raise e

    def execute(self):
        try:
            # Load data
            (X_train_sa, y_train_sa, X_test_sa, y_test_sa), (X_train_cv, y_train_cv, X_test_cv, y_test_cv) = self.load_data()

            # Train models
            sa_performance = self.train_models(X_train_sa, y_train_sa, X_test_sa, y_test_sa, "Sa")
            cv_performance = self.train_models(X_train_cv, y_train_cv, X_test_cv, y_test_cv, "CV")

            logger.info(f"Surface Roughness (Sa) Model Performance: {sa_performance}")
            logger.info(f"Cell Viability (CV) Model Performance: {cv_performance}")

            # Hyperparameter tuning
            best_rf_model_sa = self.hyperparameter_tuning(X_train_sa, y_train_sa, "RandomForest")
            joblib.dump(best_rf_model_sa, self.config.root_dir / "best_rf_model_sa.joblib")

            best_rf_model_cv = self.hyperparameter_tuning(X_train_cv, y_train_cv, "BaggingRF")
            joblib.dump(best_rf_model_cv, self.config.root_dir / "best_rf_model_cv.joblib")

            logger.info("Model training and tuning completed successfully.")
        except Exception as e:
            logger.exception(f"Error during model training execution: {e}")
            raise e
