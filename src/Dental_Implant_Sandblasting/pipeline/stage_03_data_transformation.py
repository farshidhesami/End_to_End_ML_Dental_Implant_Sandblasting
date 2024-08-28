from Dental_Implant_Sandblasting.config.configuration_manager import ConfigurationManager
from Dental_Implant_Sandblasting.components.data_transformation import DataTransformation
from Dental_Implant_Sandblasting import logger
from pathlib import Path

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Check the status file from data validation
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1].strip()

                # Proceed only if validation status is True
                if status == "True":
                    # Get configuration for data transformation
                    config = ConfigurationManager()
                    data_transformation_config = config.get_data_transformation_config()
                    
                    # Initialize DataTransformation component
                    data_transformation = DataTransformation(config=data_transformation_config)
                    
                    # Execute the transformation process
                    data_transformation.execute()
                else:
                    raise Exception("Data validation failed: Your data schema is not valid")

        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # Run the data transformation pipeline
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
