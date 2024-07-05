import os
import pandas as pd
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True  # Initialize to True assuming validation will pass

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            # Combine schema columns and target columns into a single list
            all_schema = list(self.config.all_schema.keys())

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False  # Set to False if any column is not found in the schema
                    logger.error(f"Column {col} not found in schema")
                    break  # Stop further checks if a mismatch is found
                else:
                    logger.info(f"Column {col} is valid")

            # Write the validation status to the status file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status  # Return the final validation status

        except Exception as e:
            logger.exception(e)
            raise e


