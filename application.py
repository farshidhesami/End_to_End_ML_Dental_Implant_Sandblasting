from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from Dental_Implant_Sandblasting.pipeline.prediction import PredictionPipeline

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the home page route
@app.route('/', methods=['GET'])
def home_page():
    """Render the main index page with the form for prediction input."""
    return render_template('index.html')

# Route for training the model
@app.route('/train', methods=['POST'])
def train_model():
    """Trigger model training by running main.py as a subprocess."""
    try:
        logger.info("Starting model training...")
        # Run main.py for training and capture output
        result = subprocess.run(['python', 'main.py'], capture_output=True, text=True, check=True)
        logger.info("Model training completed successfully.")
        return jsonify({"message": "Training successful!", "details": result.stdout}), 200
    except subprocess.CalledProcessError as e:
        logger.error(f"Model training failed: {e.stderr}")
        return jsonify({"message": "Training failed", "error": e.stderr}), 500

# Route to display the prediction form
@app.route('/predict', methods=['GET'])
def predict_form():
    """Render the prediction form on index.html."""
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    """Handle form data for prediction and display results."""
    try:
        # Define the expected input features for sandblasting prediction
        input_features = [
            'angle_sandblasting', 'pressure_sandblasting_bar', 'temperature_acid_etching',
            'time_acid_etching_min', 'voltage_anodizing_v', 'time_anodizing_min'
        ]

        # Collect and validate form data
        data = []
        for feature in input_features:
            try:
                # Get the form value and convert to float; default to 0 if value is missing
                value = float(request.form.get(feature, 0))
            except ValueError:
                logger.error(f"Invalid input for feature: {feature}")
                return render_template(
                    'index.html',
                    error_message=f"Invalid input for feature: {feature}. Please provide a valid number."
                )
            data.append(value)

        # Convert the data into a DataFrame with column names
        data_df = pd.DataFrame([data], columns=input_features)

        # Initialize the prediction pipeline and make a prediction
        prediction_pipeline = PredictionPipeline()
        predictions = prediction_pipeline.predict(data_df)

        logger.info(f"Prediction results: {predictions}")

        # Create a correlation heatmap
        corr_matrix = data_df.corr()
        corr_file_path = "static/img/correlation_heatmap.png"
        os.makedirs("static/img", exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Heatmap")
        plt.savefig(corr_file_path)
        plt.close()

        # Render the results page with predictions, passing individual values for visualization
        return render_template(
            'results.html',
            sa_rf=predictions['sa_predictions_rf'][0],
            sa_ridge=predictions['sa_predictions_ridge'][0],
            cv_bagging=predictions['cv_predictions_bagging'][0],
            benchmark_sa=0.5,  # Example benchmark
            benchmark_cv=0.7,  # Example benchmark
            correlation_heatmap_path=corr_file_path,
            distribution_sa=[0.2, 0.4, 0.6],  # Placeholder distribution data
            distribution_cv=[0.1, 0.3, 0.5]  # Placeholder distribution data
        )
    except Exception as e:
        logger.exception("An error occurred during prediction.")
        return render_template(
            'index.html',
            error_message="An error occurred during prediction. Please try again or contact support."
        )

# Route for downloading reports
@app.route('/download_report', methods=['GET'])
def download_report():
    """Generate and serve a downloadable report."""
    try:
        # Example report data
        report_data = {
            "Metric": ["Surface Roughness (Sa)", "Cell Viability (CV)"],
            "RF Prediction": [0.24, 0.039],  # Example placeholders
            "Ridge Prediction (Sa)": [0.95, None],
            "Benchmark": [0.5, 0.7]
        }
        report_df = pd.DataFrame(report_data)

        # Save report to an Excel file
        report_path = "artifacts/prediction_report.xlsx"
        os.makedirs("artifacts", exist_ok=True)  # Ensure the directory exists
        report_df.to_excel(report_path, index=False)

        # Send file to user
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        logger.exception("Failed to generate report.")
        return jsonify({"message": "Failed to generate report", "error": str(e)}), 500

if __name__ == "__main__":
    # Start the Flask app on port 8080, with debug mode enabled for development
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)  # Set debug=False in production
