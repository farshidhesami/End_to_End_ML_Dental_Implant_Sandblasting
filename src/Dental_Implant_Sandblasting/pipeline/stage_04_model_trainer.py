from Dental_Implant_Sandblasting.config.configuration_manager import ConfigurationManager
from Dental_Implant_Sandblasting.components.model_trainer import ModelTrainer
from Dental_Implant_Sandblasting import logger
from pathlib import Path

STAGE_NAME = "Model Trainer stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_transformation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1].strip()

                if status == "True":
                    config = ConfigurationManager()
                    model_trainer_config = config.get_model_trainer_config()
                    model_trainer = ModelTrainer(config=model_trainer_config)
                    model_trainer.execute()
                else:
                    raise Exception("Your data transformation is not completed")

        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
