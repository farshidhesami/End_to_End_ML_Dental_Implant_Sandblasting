from pathlib import Path
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from Dental_Implant_Sandblasting.utils.common import read_yaml, create_directories
from Dental_Implant_Sandblasting.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig
)

# Define Configuration Manager
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH
    ):
        # Reading configurations, parameters, and schema files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Creating the necessary directories defined in config.yaml
        create_directories([self.config['artifacts_root']])

    # Fetch Data Ingestion Config
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Extracting ingestion configurations
        config = self.config['data_ingestion']
        params = self.params['data_ingestion']

        # Create the root directory for data ingestion artifacts
        create_directories([config['root_dir']])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            source_URL=config['source_URL'],
            local_data_file=Path(config['local_data_file']),
            unzip_dir=Path(config['unzip_dir']),
            columns_to_convert=config["columns_to_convert"],
            sa_lower_bound=params["sa_lower_bound"],
            sa_upper_bound=params["sa_upper_bound"],
            cell_viability_threshold=params["cell_viability_threshold"],
            outlier_capping_method=params["outlier_capping_method"],
            outlier_tail=params["outlier_tail"],
            outlier_fold=params["outlier_fold"],
            log_transform_variable=params["log_transform_variable"]
        )

        return data_ingestion_config

    # Fetch Data Validation Config
    def get_data_validation_config(self) -> DataValidationConfig:
        # Extracting validation configurations and parameters
        config = self.config['data_validation']
        params = self.params['data_validation']                               # Fetching validation-related params
        schema = {**self.schema['COLUMNS'], **self.schema['TARGET_COLUMNS']}  # Combining schema columns

        # Create the root directory for data validation artifacts
        create_directories([config['root_dir']])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config['root_dir']),
            unzip_data_dir=Path(config['unzip_data_dir']),                    # Updated to include the CSV path
            STATUS_FILE=Path(config['STATUS_FILE']),
            all_schema=schema,
            columns_to_convert=config['columns_to_convert'],     # Fetch from config.yaml
            knn_n_neighbors=params['knn_n_neighbors'],           # Fetch from params.yaml
            sa_lower_bound=params['sa_lower_bound'],             # Fetch from params.yaml
            sa_upper_bound=params['sa_upper_bound'],             # Fetch from params.yaml
            feature_columns=config['feature_columns'],           # Fetch from config.yaml
            target_column_sa=config['target_column_sa'],         # Fetch from config.yaml
            target_column_cv=config['target_column_cv'],         # Fetch from config.yaml
            test_size=params['test_size'],                       # Fetch from params.yaml
            random_state=params['random_state']                  # Fetch from params.yaml
        )

        return data_validation_config
