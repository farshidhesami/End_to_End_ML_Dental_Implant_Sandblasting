import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import ModelTrainerConfig

# Define ModelTrainer class
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.poly = None  # Initialize the poly attribute

    def load_data(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            X_train = train_data.drop(columns=[self.config.target_column])
            y_train = train_data[self.config.target_column]

            X_test = test_data.drop(columns=[self.config.target_column])
            y_test = test_data[self.config.target_column]

            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def preprocess_data(self, X_train, y_train, X_test, y_test):
        try:
            # Imputation
            imputer = SimpleImputer(strategy=self.config.imputation_strategy)
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            y_imputer = SimpleImputer(strategy="most_frequent")
            y_train = y_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            y_test = y_imputer.transform(y_test.values.reshape(-1, 1)).ravel()

            # Create polynomial features
            poly = PolynomialFeatures(degree=self.config.poly_features_degree)
            self.poly = poly  # Assign the poly attribute
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            # Ensure the directory exists
            poly_features_path = self.config.poly_features_path
            poly_features_path.parent.mkdir(parents=True, exist_ok=True)

            # Save polynomial features for later use
            joblib.dump(poly, poly_features_path)

            # Scaling
            if self.config.scaling_method == "StandardScaler":
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")

            X_train_scaled = scaler.fit_transform(X_train_poly)
            X_test_scaled = scaler.transform(X_test_poly)

            return X_train_scaled, y_train, X_test_scaled, y_test
        except PermissionError as e:
            logger.error(f"Permission denied: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise e

    def evaluate_models(self, X_train, y_train):
        try:
            models = self.config.models
            performance_metrics = {}

            for model_name, model_params in models.items():
                logger.info(f"Training {model_name}...")
                if model_name == "ridge":
                    model = Ridge(**model_params)
                elif model_name == "elasticnet":
                    model = ElasticNet(**model_params)
                elif model_name == "bayesian_ridge":
                    model = BayesianRidge(**model_params)
                elif model_name == "huber_regressor":
                    model = HuberRegressor(**model_params)
                elif model_name == "random_forest":
                    model = RandomForestRegressor(**model_params)
                elif model_name == "gradient_boosting":
                    model = GradientBoostingRegressor(**model_params)
                elif model_name == "svr":
                    model = SVR(**model_params)
                elif model_name == "xgboost":
                    model = XGBRegressor(**model_params)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)

                mae = mean_absolute_error(y_train, y_pred_train)
                rmse = mean_squared_error(y_train, y_pred_train, squared=False)
                r2 = r2_score(y_train, y_pred_train)
                mape = mean_absolute_percentage_error(y_train, y_pred_train)
                medae = median_absolute_error(y_train, y_pred_train)

                performance_metrics[model_name] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "MAPE": mape,
                    "MedAE": medae
                }
                logger.info(f"{model_name} - MAE: {mae}")

            # Visualization: Performance Metrics
            self.visualize_performance(performance_metrics, test=False)
            return performance_metrics
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            raise e

    def hyperparameter_tuning(self, X_train, y_train):
        try:
            best_models = {}

            for model_name, param_grid in self.config.param_grids.items():
                if model_name == "ridge":
                    model = Ridge()
                elif model_name == "elasticnet":
                    model = ElasticNet()
                elif model_name == "huber_regressor":
                    model = HuberRegressor()
                elif model_name == "svr":
                    model = SVR()
                elif model_name == "random_forest":
                    model = RandomForestRegressor()
                elif model_name == "gradient_boosting":
                    model = GradientBoostingRegressor()
                elif model_name == "xgboost":
                    model = XGBRegressor()
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=self.config.cv, scoring=self.config.scoring, n_jobs=-1)
                logger.info(f"Tuning {model_name}...")
                grid_search.fit(X_train, y_train)
                best_models[model_name] = grid_search.best_estimator_
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

            return best_models
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise e

    def evaluate_best_models(self, best_models, X_test, y_test):
        try:
            performance_metrics = {}

            for model_name, model in best_models.items():
                y_pred_test = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred_test)
                rmse = mean_squared_error(y_test, y_pred_test, squared=False)
                r2 = r2_score(y_test, y_pred_test)
                mape = mean_absolute_percentage_error(y_test, y_pred_test)
                medae = median_absolute_error(y_test, y_pred_test)

                performance_metrics[model_name] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "MAPE": mape,
                    "MedAE": medae
                }
                logger.info(f"{model_name} - Test MAE: {mae}, RMSE: {rmse}, R2: {r2}, MAPE: {mape}, MedAE: {medae}")

            # Visualization: Best Model Performance
            self.visualize_performance(performance_metrics, test=True)
            return performance_metrics
        except Exception as e:
            logger.error(f"Error evaluating best models: {e}")
            raise e

    def save_models(self, best_models):
        try:
            for model_name, model in best_models.items():
                model_save_path = self.config.model_path / f"{model_name}.joblib"
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_save_path)
                logger.info(f"Saved {model_name} model at: {model_save_path}")

            # Save the best models for 'sa' and 'cv' with the expected filenames
            best_model_sa = best_models.get(self.config.sa_model_name)
            best_model_cv = best_models.get(self.config.cv_model_name)

            if best_model_sa:
                joblib.dump(best_model_sa, self.config.model_path / 'best_model_sa.joblib')
                logger.info("Saved best model for Surface Roughness (Sa) at: artifacts/model_trainer/models/best_model_sa.joblib")

            if best_model_cv:
                joblib.dump(best_model_cv, self.config.model_path / 'best_model_cv.joblib')
                logger.info("Saved best model for Cell Viability (CV) at: artifacts/model_trainer/models/best_model_cv.joblib")

            # Save polynomial features
            joblib.dump(self.poly, self.config.poly_features_path)
            logger.info("Saved polynomial features at: artifacts/model_trainer/poly_features.joblib")

        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            raise e

    def visualize_performance(self, performance_metrics, test=False):
        try:
            metric_df = pd.DataFrame(performance_metrics).T
            metric_df = metric_df[['MAE', 'RMSE', 'R2', 'MAPE', 'MedAE']]

            if test:
                title = 'Test Set Performance'
            else:
                title = 'Training Set Performance'

            plt.figure(figsize=(12, 8))
            sns.barplot(data=metric_df.reset_index().melt(id_vars='index'), x='index', y='value', hue='variable', palette="Set2")
            plt.title(title)
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.legend(loc='upper right')
            plt.show()
        except Exception as e:
            logger.error(f"Error visualizing performance: {e}")
            raise e

    def execute(self):
        try:
            X_train, y_train, X_test, y_test = self.load_data()
            X_train, y_train, X_test, y_test = self.preprocess_data(X_train, y_train, X_test, y_test)
            model_performance = self.evaluate_models(X_train, y_train)
            performance_df = pd.DataFrame(model_performance).T
            print("\nModel Performance:\n", performance_df)

            best_models = self.hyperparameter_tuning(X_train, y_train)
            performance_metrics = self.evaluate_best_models(best_models, X_test, y_test)

            best_hyperparameters = {model_name: model.get_params() for model_name, model in best_models.items()}
            print("\nBest Hyperparameters:\n", best_hyperparameters)
            print("\nPerformance Metrics:\n", performance_metrics)

            self.save_models(best_models)
        except Exception as e:
            logger.exception(e)
            raise e