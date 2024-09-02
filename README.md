# End-to-End Machine Learning for Predicting Optimal Sandblasting Conditions and Performance Metrics for Dental Implants

## Project Overview

This project aims to build an end-to-end machine learning pipeline that predicts the optimal sandblasting conditions for dental implants based on several factors. The goal is to achieve accurate predictions of surface roughness (Sa) and cell viability (%) based on various input conditions, ensuring that the implants meet the required performance standards.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Components](#key-components)
  - [Data Inputs](#data-inputs)
  - [Prediction Metrics](#prediction-metrics)
  - [Procedure for Predictions](#procedure-for-predictions)
- [Project Structure](#project-structure)
- [Research Stages](#research-stages)
  - [Stage 01: Data Ingestion](#stage-01-data-ingestion)
  - [Stage 02: Data Validation](#stage-02-data-validation)
  - [Stage 03: Data Transformation](#stage-03-data-transformation)
  - [Stage 04: Model Trainer](#stage-04-model-trainer)
  - [Stage 05: Model Evaluation](#stage-05-model-evaluation)
- [GitHub Repository](#github-repository)
- [Contact Information](#contact-information)


## Key Components

### Data Inputs:

The model uses specific input conditions related to sandblasting, acid etching, and anodizing processes to predict the quality metrics of the implant surface.

**Sandblasting Conditions:**

- **Angle of Sandblasting (degrees) (`angle_sandblasting`)**: 
  - The angle at which sandblasting is performed on the implant surface.
  - This angle influences the texture and roughness of the implant surface.

- **Pressure of Sandblasting (bar) (`pressure_sandblasting_bar`)**: 
  - The pressure applied during the sandblasting process.
  - Higher pressures typically result in more aggressive surface modifications.

**Acid Etching Conditions:**

- **Temperature of Acid Etching (°C) (`temperature_acid_etching`)**: 
  - The temperature at which acid etching is conducted to create micro-textures on the implant surface.
  - Temperature affects the depth and uniformity of the etching.

- **Time of Acid Etching (minutes) (`time_acid_etching_min`)**: 
  - The duration for which acid etching is performed.
  - Longer etching times can increase surface roughness but may also affect other properties like cell viability.

**Anodizing Conditions:**

- **Voltage of Anodizing (V) (`voltage_anodizing_v`)**: 
  - The voltage used during anodizing to form a protective oxide layer on the implant surface.
  - The voltage influences the thickness and structure of the oxide layer.

- **Time of Anodizing (minutes) (`time_anodizing_min`)**: 
  - The duration of the anodizing process.
  - Time influences the formation and quality of the oxide layer.

### Prediction Metrics:

The model predicts two key metrics: **Surface Roughness (Sa)** and **Cell Viability (%)**, both critical for the performance and biocompatibility of implants.

**Average Surface Roughness (Sa) (`sa_surface_roughness_micrometer`)**:
- **Definition**: A critical metric for implant performance, representing the average roughness of the surface, measured in micrometers (µm).
- **Importance**: Surface roughness affects the implant’s interaction with biological tissues and its ability to integrate with bone (osseointegration).
- **Validation Range**: The predicted Sa value must fall within the range 1.5μm < Sa < 2.5μm to be considered valid.
- **Action**: If the predicted Sa falls outside this range, the "Cell Viability (%)" prediction is automatically set to 0 (indicating failure).

**Cell Viability (%) (`cell_viability_percent`)**:
- **Definition**: Represents the percentage of viable cells on the implant surface, indicating its biocompatibility.
- **Importance**: High cell viability is crucial for the successful integration of the implant into biological tissue.
- **Threshold for Validity**: Predictions are valid only if `cell_viability_percent > 90`.
- **Binary Indicator (`Result_Passed_1_Failed_0`)**:
  - **1 (Passed)**: If `cell_viability_percent > 90`, the implant is considered to have passed.
  - **0 (Failed)**: If `cell_viability_percent ≤ 90`, the implant is considered to have failed.

### Procedure for Predictions:

This section outlines the steps the model follows to predict and validate the metrics for implant surface quality.

**Predict Surface Roughness (Sa):**
1. **Input Conditions**: The sandblasting, acid etching, and anodizing conditions are input into the model.
2. **Prediction**: The model predicts the `sa_surface_roughness_micrometer` value based on the input conditions.

**Evaluate Surface Roughness (Sa):**
1. **Check Validity**: The predicted Sa value is evaluated to ensure it falls within the range 1.5μm < Sa < 2.5μm.
   - **If Valid**: The model proceeds to predict `cell_viability_percent`.
   - **If Invalid**: The model sets `cell_viability_percent` to 0, indicating failure.

**Predict Cell Viability (%):**
1. **Prediction**: If the predicted Sa value is valid, the model uses the input conditions to predict `cell_viability_percent`.

**Evaluate Cell Viability (%):**
1. **Check Validity**: The predicted `cell_viability_percent` is evaluated to ensure it exceeds 90%.
2. **Binary Indicator (`Result_Passed_1_Failed_0`)**: The result is recorded as:
   - **1 (Passed)**: If `cell_viability_percent > 90`.
   - **0 (Failed)**: If `cell_viability_percent ≤ 90`.

## Project Structure

The project is structured into various stages, each representing a critical step in the machine learning pipeline. Below is an outline of the project structure, with a focus on how each component fits into the research process.

### General Code for Whole Project:

1. `setup.py`
2. `requirements.txt`
3. `params.yaml`
4. `src/Dental_Implant_Sandblasting/utils/common.py`
5. `src/Dental_Implant_Sandblasting/constants/__init__.py`
6. `template.py`
7. `src/Dental_Implant_Sandblasting/constants/__init__.py`

## Research Stages

### Research Stage: Stage 01: Data Ingestion

#### Dental_Implant_Sandblasting Project Structure and Flow:

**Data Loading and Exploration (Step 2):**

- **Tasks:**
  - Load the dataset (`Sandblasting-Condition.csv`).
  - Perform Exploratory Data Analysis (EDA) to check for missing values, analyze basic statistics, and explore relationships between variables.
  - Output a loaded dataset and initial insights from EDA.

#### Rules for Implementation:

- **Import Necessary Libraries:** Import all libraries required for data loading and exploration.
- **Make Data Class:** Define or update the data class in `src/Dental_Implant_Sandblasting/entity/config_entity.py` to handle paths and configurations related to data ingestion.
- **Define ConfigurationManager Class:** Update the configuration manager in `src/Dental_Implant_Sandblasting/config/configuration_manager.py` to include methods for loading data ingestion configurations.
- **Define DataIngestion Class:** Create or update the `DataIngestion` class in `src/Dental_Implant_Sandblasting/components/data_ingestion.py` to handle data loading and basic exploration.
- **Pipeline Execution:** Implement the pipeline execution in `src/Dental_Implant_Sandblasting/pipeline/stage_01_data_ingestion.py`.

#### General Updates:

- **Update the config/config.yaml:** Include data ingestion paths.
- **Update main.py:** Include the execution of this stage.
- **Update the schema.yaml:** If needed, include schema definitions for the data.

### Research Stage: Stage 02: Data Validation

#### Dental_Implant_Sandblasting Project Structure and Flow:

**Data Preprocessing (Step 3):**

- **Tasks:**
  - Validate data integrity, ensure correct data types, and handle missing values or outliers.
  - Output a cleaned and validated dataset ready for transformation.

#### Rules for Implementation:

- **Import Necessary Libraries:** Import libraries required for data validation, including those for data integrity checks.
- **Make Data Class:** Define or update the data validation class in `src/Dental_Implant_Sandblasting/entity/config_entity.py` to handle validation configurations.
- **Define ConfigurationManager Class:** Update the configuration manager to include methods for loading data validation configurations.
- **Define DataValidation Class:** Create or update the `DataValidation` class in `src/Dental_Implant_Sandblasting/components/data_validation.py` to perform data validation tasks.
- **Pipeline Execution:** Implement the pipeline execution in `src/Dental_Implant_Sandblasting/pipeline/stage_02_data_validation.py`.

#### General Updates:

- **Update the config/config.yaml:** Include data validation configurations.
- **Ensure main.py:** Executes the data validation stage after data ingestion.
- **Update the schema.yaml:** To reflect any validation rules.


### Research Stage: Stage 03: Data Transformation

#### Dental_Implant_Sandblasting Project Structure and Flow:

**Feature Engineering (Step 4):**

- **Tasks:**
  - Apply transformations (e.g., scaling, normalization), encode categorical variables, and perform feature selection or creation.
  - Output a transformed dataset with features ready for modeling.

#### Rules for Implementation:

- **Import Necessary Libraries:** Import libraries for feature engineering and transformation.
- **Make Data Class:** Define or update the data transformation class in `src/Dental_Implant_Sandblasting/entity/config_entity.py` to handle transformation configurations.
- **Define ConfigurationManager Class:** Update the configuration manager to include methods for loading data transformation configurations.
- **Define DataTransformation Class:** Create or update the `DataTransformation` class in `src/Dental_Implant_Sandblasting/components/data_transformation.py` to apply necessary transformations.
- **Pipeline Execution:** Implement the pipeline execution in `src/Dental_Implant_Sandblasting/pipeline/stage_03_data_transformation.py`.

#### General Updates:

- **Update the config/config.yaml:** With transformation parameters.
- **Ensure main.py:** Includes this stage in the pipeline.
- **Adjust schema.yaml:** As needed to include transformation-related schema.

### Research Stage: Stage 04: Model Trainer

#### Dental_Implant_Sandblasting Project Structure and Flow:

**Model Training (Step 5):**

- **Tasks:**
  - Train multiple models and evaluate them using cross-validation.
  - Assess model performance and select the best models.

**Hyperparameter Tuning (Step 6):**

- **Tasks:**
  - Optimize model hyperparameters using techniques like GridSearchCV or RandomizedSearchCV.
  - Output optimized models ready for final evaluation.

#### Rules for Implementation:

- **Import Necessary Libraries:** Import libraries for model training and hyperparameter tuning, such as scikit-learn, xgboost, etc.
- **Make Data Class:** Define or update the model training class in `src/Dental_Implant_Sandblasting/entity/config_entity.py` to handle model training configurations.
- **Define ConfigurationManager Class:** Update the configuration manager to include methods for loading model training and hyperparameter tuning configurations.
- **Define ModelTrainer Class:** Create or update the `ModelTrainer` class in `src/Dental_Implant_Sandblasting/components/model_trainer.py` to handle model training and tuning.
- **Pipeline Execution:** Implement the pipeline execution in `src/Dental_Implant_Sandblasting/pipeline/stage_04_model_trainer.py`.

#### General Updates:

- **Update the config/config.yaml:** With model training and hyperparameter tuning parameters.
- **Ensure main.py:** Runs the model training stage after data transformation.
- **Modify schema.yaml:** If required to include model-related schema.

### Research Stage: Stage 05: Model Evaluation

#### Dental_Implant_Sandblasting Project Structure and Flow:

**Model Evaluation (Step 7):**

- **Tasks:**
  - Evaluate the best models on a separate test set to ensure they generalize well.
  - Generate a comprehensive report on model performance.

#### Rules for Implementation:

- **Import Necessary Libraries:** Import libraries necessary for model evaluation, such as metrics and plotting libraries.
- **Make Data Class:** Define or update the model evaluation class in `src/Dental_Implant_Sandblasting/entity/config_entity.py` to handle evaluation configurations.
- **Define ConfigurationManager Class:** Update the configuration manager to include methods for loading model evaluation configurations.
- **Define ModelEvaluation Class:** Create or update the `ModelEvaluation` class in `src/Dental_Implant_Sandblasting/components/model_evaluation.py` to evaluate model performance.
- **Pipeline Execution:** Implement the pipeline execution in `src/Dental_Implant_Sandblasting/pipeline/stage_05_model_evaluation.py`.

#### General Updates:

- **Update config/config.yaml:** With evaluation parameters.
- **Ensure main.py:** Includes this final stage in the pipeline.
- **Revise schema.yaml:** If necessary to include evaluation metrics or thresholds.

## Model Choices

The project explores several models to predict the outcomes accurately. The primary models used include:

- **RandomForestRegressor:** A versatile and powerful ensemble learning method that builds multiple decision trees and merges them together to get more accurate and stable predictions.
- **BaggingRegressor:** An ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregates their individual predictions to form a final prediction.

Hyperparameter tuning was performed to optimize these models to achieve the best possible performance.

# GitHub Repository

For more details on the project, please visit the GitHub repository:
- GitHub: [https://github.com/farshidhesami](https://github.com/farshidhesami)

## Contact Information

For any questions or suggestions, please feel free to connect:
- LinkedIn: [https://www.linkedin.com/in/farshid-hesami-33a09529/](https://www.linkedin.com/in/farshid-hesami-33a09529/)
