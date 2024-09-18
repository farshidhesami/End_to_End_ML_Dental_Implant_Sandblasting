from Dental_Implant_Sandblasting.config.configuration_manager import ConfigurationManager
from Dental_Implant_Sandblasting.components.data_transformation import DataTransformation
from Dental_Implant_Sandblasting import logger

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        
        # Initialize DataTransformation component
        data_transformation = DataTransformation(config=data_transformation_config)

        # Execute the transformation process
        data_transformation.execute()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
