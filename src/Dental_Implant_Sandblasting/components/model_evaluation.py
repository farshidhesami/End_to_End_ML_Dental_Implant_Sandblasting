from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import ModelEvaluationConfig
from Dental_Implant_Sandblasting.utils.common import save_json

# ModelEvaluation class
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_data(self):
        # Load test datasets for Surface Roughness (Sa) and Cell Viability (CV)
        X_test_sa = pd.read_csv(self.config.test_sa_data).iloc[:, :-1]
        y_test_sa = pd.read_csv(self.config.test_sa_data).iloc[:, -1]
        X_test_cv = pd.read_csv(self.config.test_cv_data).iloc[:, :-1]
        y_test_cv = pd.read_csv(self.config.test_cv_data).iloc[:, -1]
        
        return (X_test_sa, y_test_sa), (X_test_cv, y_test_cv)

    def load_models(self):
        # Load the best models saved from previous stages
        rf_model_sa = joblib.load(self.config.model_dir / "best_rf_model_sa.joblib")
        ridge_model_sa = joblib.load(self.config.model_dir / "best_ridge_model_sa.joblib")
        bagging_model_cv = joblib.load(self.config.model_dir / "best_rf_model_cv.joblib")

        return rf_model_sa, ridge_model_sa, bagging_model_cv

    # Evaluation function
    def evaluate_model(self, y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-10))

        logger.info(f"\nEvaluation metrics for {model_name}:")
        logger.info(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}%, SMAPE: {smape:.4f}%")

        return {"MAE": mae, "MSE": mse, "R²": r2, "MAPE": mape, "SMAPE": smape}

    # Visualizations function
    def visualize_results(self, y_test_sa, y_sa_pred_rf, y_sa_pred_ridge, y_test_cv, y_cv_pred_bagging):
        # Predicted vs Actual Values Plot
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_test_sa, y_sa_pred_rf, alpha=0.6, label="RandomForest Predictions (Sa)", color="blue", edgecolors="k")
        plt.scatter(y_test_sa, y_sa_pred_ridge, alpha=0.6, label="Ridge Predictions (Sa)", color="green", edgecolors="k")
        plt.plot([y_test_sa.min(), y_test_sa.max()], [y_test_sa.min(), y_test_sa.max()], color="red", linestyle="--", label="Perfect Fit")
        plt.title("Actual vs Predicted Surface Roughness (Sa)")
        plt.xlabel("Actual Sa")
        plt.ylabel("Predicted Sa")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(y_test_cv, y_cv_pred_bagging, alpha=0.6, label="BaggingRF Predictions (CV)", color="orange", edgecolors="k")
        plt.plot([y_test_cv.min(), y_test_cv.max()], [y_test_cv.min(), y_test_cv.max()], color="red", linestyle="--", label="Perfect Fit")
        plt.title("Actual vs Predicted Cell Viability (CV)")
        plt.xlabel("Actual CV")
        plt.ylabel("Predicted CV")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Execute the evaluation
    def execute(self):
        try:
            # Load data
            (X_test_sa, y_test_sa), (X_test_cv, y_test_cv) = self.load_data()

            # Load models
            rf_model_sa, ridge_model_sa, bagging_model_cv = self.load_models()

            # Make predictions
            y_sa_pred_rf = rf_model_sa.predict(X_test_sa)
            y_sa_pred_ridge = ridge_model_sa.predict(X_test_sa)
            y_cv_pred_bagging = bagging_model_cv.predict(X_test_cv)

            # Evaluate models
            rf_metrics_sa = self.evaluate_model(y_test_sa, y_sa_pred_rf, "RandomForest (Sa)")
            ridge_metrics_sa = self.evaluate_model(y_test_sa, y_sa_pred_ridge, "Ridge (Sa)")
            bagging_metrics_cv = self.evaluate_model(y_test_cv, y_cv_pred_bagging, "BaggingRF (CV)")

            # Visualize the results
            self.visualize_results(y_test_sa, y_sa_pred_rf, y_sa_pred_ridge, y_test_cv, y_cv_pred_bagging)

            # Saving predictions and metrics in JSON format (optional for web interface)
            results = {
                "predictions": {
                    "y_sa_pred_rf": y_sa_pred_rf.tolist(),
                    "y_sa_pred_ridge": y_sa_pred_ridge.tolist(),
                    "y_cv_pred_bagging": y_cv_pred_bagging.tolist()
                },
                "metrics": {
                    "RandomForest (Sa)": rf_metrics_sa,
                    "Ridge (Sa)": ridge_metrics_sa,
                    "BaggingRF (CV)": bagging_metrics_cv
                }
            }
            save_json(self.config.root_dir / "evaluation_results.json", results)

            logger.info("Model evaluation completed successfully.")

        except Exception as e:
            logger.exception(f"Error during model evaluation execution: {e}")
            raise e
