from Dental_Implant_Sandblasting.config.configuration_manager import ConfigurationManager   # Import the ConfigurationManager
from Dental_Implant_Sandblasting.components.data_ingestion import DataIngestion             # Import the DataIngestion component
from Dental_Implant_Sandblasting import logger                                              # Import the logger

# Define the stage name for logging purposes
STAGE_NAME = "Data Ingestion stage"

# Define the Data Ingestion Training Pipeline class
class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Initialize the configuration manager and get the data ingestion configuration
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        # Initialize the DataIngestion component with the configuration
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # Execute the data ingestion steps
        data_ingestion.download_file()          # Download the data file
        data_ingestion.extract_zip_file()       # Extract the zip file
        data_ingestion.load_and_explore_data()  # Load and explore the data as part of the EDA

# Execute the data ingestion stage if this script is run as the main program
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
