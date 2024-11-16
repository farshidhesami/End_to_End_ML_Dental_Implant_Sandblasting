from Dental_Implant_Sandblasting.config.configuration_manager import ConfigurationManager
from Dental_Implant_Sandblasting.utils.common import save_json, load_data
from Dental_Implant_Sandblasting import logger
import joblib
import pandas as pd
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        """Initialize the PredictionPipeline with configuration and model loading."""
        try:
            config_manager = ConfigurationManager()
            self.config = config_manager.get_model_evaluation_config()
            self.load_models()
        except Exception as e:
            logger.exception("Error during initialization.")
            raise e

    def load_models(self):
        """Load pre-trained models for Sa and CV predictions."""
        try:
            self.rf_model_sa = joblib.load(self.config.model_dir / "best_rf_model_sa.joblib")
            self.ridge_model_sa = joblib.load(self.config.model_dir / "best_ridge_model_sa.joblib")
            self.bagging_model_cv = joblib.load(self.config.model_dir / "best_rf_model_cv.joblib")
            logger.info("Models loaded successfully.")
        except FileNotFoundError as fnf_error:
            logger.error("One or more model files are missing. Ensure all models are saved in the specified directory.")
            raise fnf_error
        except Exception as e:
            logger.exception("Error loading models.")
            raise e

    def preprocess_input_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Apply necessary preprocessing to input data (e.g., scaling, feature engineering)."""
        try:
            # Define the expected features
            expected_features = [
                'angle_sandblasting',
                'pressure_sandblasting_bar',
                'temperature_acid_etching',
                'time_acid_etching_min',
                'voltage_anodizing_v'
            ]

            # Validate and select only the expected features
            missing_features = set(expected_features) - set(input_data.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Ensure the input only contains the expected features
            input_data = input_data[expected_features]

            # Rename columns to match the model's expected format (e.g., unnamed)
            input_data.columns = range(input_data.shape[1])

            logger.info("Input data preprocessed successfully.")
            return input_data
        except KeyError as key_error:
            logger.error(f"Input data missing required columns: {key_error}")
            raise key_error
        except Exception as e:
            logger.exception("Error during data preprocessing.")
            raise e

    def predict(self, input_data: pd.DataFrame) -> dict:
        """Generate predictions using the pre-trained models for Sa and CV."""
        try:
            # Preprocess the input data
            processed_data = self.preprocess_input_data(input_data)

            # Generate predictions
            y_sa_pred_rf = self.rf_model_sa.predict(processed_data)
            y_sa_pred_ridge = self.ridge_model_sa.predict(processed_data)
            y_cv_pred_bagging = self.bagging_model_cv.predict(processed_data)

            # Create a dictionary for predictions
            predictions = {
                "sa_predictions_rf": y_sa_pred_rf.tolist(),
                "sa_predictions_ridge": y_sa_pred_ridge.tolist(),
                "cv_predictions_bagging": y_cv_pred_bagging.tolist()
            }

            logger.info("Predictions generated successfully.")
            return predictions
        except ValueError as value_error:
            logger.error(f"Prediction error due to data validation: {value_error}")
            raise value_error
        except Exception as e:
            logger.exception("Error during prediction.")
            raise e

    def save_predictions(self, predictions: dict, output_path: Path):
        """Save predictions to a JSON file."""
        try:
            save_json(output_path, predictions)
            logger.info(f"Predictions saved to {output_path}")
        except PermissionError as perm_error:
            logger.error(f"Permission denied while saving predictions: {perm_error}")
            raise perm_error
        except Exception as e:
            logger.exception("Error saving predictions.")
            raise e

    def execute(self, input_data_path: Path, output_path: Path):
        """Main method to execute the prediction pipeline."""
        try:
            # Load input data
            input_data = load_data(input_data_path)
            logger.info(f"Input data loaded from {input_data_path}")

            # Generate predictions
            predictions = self.predict(input_data)

            # Save predictions
            self.save_predictions(predictions, output_path)
        except FileNotFoundError as fnf_error:
            logger.error(f"Input file not found: {fnf_error}")
            raise fnf_error
        except Exception as e:
            logger.exception("Error executing prediction pipeline.")
            raise e


if __name__ == "__main__":
    try:
        STAGE_NAME = "Prediction Stage"
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

        # Example usage
        input_data_path = Path("path/to/your/input_data.csv")  # Update this path
        output_path = Path("artifacts/predictions/prediction_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prediction_pipeline = PredictionPipeline()
        prediction_pipeline.execute(input_data_path, output_path)

        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception("Unhandled exception in the main execution block.")
        raise e
