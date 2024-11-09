from Dental_Implant_Sandblasting.config.configuration_manager import ConfigurationManager
from Dental_Implant_Sandblasting.components.model_trainer import ModelTrainer
from Dental_Implant_Sandblasting import logger

# Define stage name for logging purposes
STAGE_NAME = "Model Training Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        # Initialize the configuration manager and get the model trainer configuration
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        
        # Initialize the ModelTrainer component with the configuration
        model_trainer = ModelTrainer(config=model_trainer_config)
        
        # Execute the model training, evaluation, and hyperparameter tuning
        model_trainer.execute()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        pipeline = ModelTrainerPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
