from flask import Flask, render_template, request, jsonify
import subprocess
import pandas as pd
from Dental_Implant_Sandblasting.pipeline.prediction import PredictionPipeline
import os
import logging

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
                return jsonify({"message": f"Invalid input for feature: {feature}"}), 400
            data.append(value)

        # Convert the data into a DataFrame with column names
        data_df = pd.DataFrame([data], columns=input_features)

        # Initialize the prediction pipeline and make a prediction
        prediction_pipeline = PredictionPipeline()
        predictions = prediction_pipeline.predict(data_df)

        logger.info(f"Prediction results: {predictions}")

        # Render the results page with predictions
        return render_template('results.html', predictions=predictions)
    except Exception as e:
        logger.exception("An error occurred during prediction.")
        return jsonify({"message": "An error occurred during prediction", "error": str(e)}), 500

if __name__ == "__main__":
    # Start the Flask app on port 8080, with debug mode enabled for development
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)  # Set debug=False in production
