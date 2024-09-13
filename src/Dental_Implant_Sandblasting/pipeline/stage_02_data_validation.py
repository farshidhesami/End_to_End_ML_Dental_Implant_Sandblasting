from Dental_Implant_Sandblasting.config.configuration_manager import ConfigurationManager
from Dental_Implant_Sandblasting.components.data_validation import DataValidation
from Dental_Implant_Sandblasting import logger

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)

        # Step 1: Validate all columns
        validation_status = data_validation.validate_all_columns()
        logger.info(f"Validation status: {validation_status}")

        # Step 2: If validation passes, preprocess the data
        if validation_status:
            logger.info("Validation successful. Proceeding to data preprocessing...")
            data_validation.preprocess_data()
        else:
            logger.error("Validation failed. Preprocessing will not proceed.")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
