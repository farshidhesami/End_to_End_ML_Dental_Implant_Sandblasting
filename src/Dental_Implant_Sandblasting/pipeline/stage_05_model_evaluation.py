from Dental_Implant_Sandblasting.config.configuration_manager import ConfigurationManager
from Dental_Implant_Sandblasting.components.model_evaluation import ModelEvaluation
from Dental_Implant_Sandblasting import logger

# Define the stage name for logging purposes
STAGE_NAME = "Model Evaluation Stage"

# Define the Model Evaluation Pipeline class
class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        # Initialize configuration manager and get model evaluation configuration
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        # Initialize the ModelEvaluation component with the configuration
        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        # Execute the model evaluation process
        model_evaluation.execute()

# Execute the pipeline if this script is run as the main program
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"An error occurred in the {STAGE_NAME}: {e}")
        raise e
