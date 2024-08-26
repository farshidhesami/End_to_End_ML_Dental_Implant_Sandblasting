from pathlib import Path
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from Dental_Implant_Sandblasting.utils.common import read_yaml, create_directories
from Dental_Implant_Sandblasting.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH, schema_filepath=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config['artifacts_root']])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']
        create_directories([config['root_dir']])
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            source_URL=config['source_URL'],
            local_data_file=Path(config['local_data_file']),
            unzip_dir=Path(config['unzip_dir'])
        )
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config['data_validation']
        schema = {**self.schema['COLUMNS'], **self.schema['TARGET_COLUMNS']}
        create_directories([config['root_dir']])
        data_validation_config = DataValidationConfig(
            root_dir=Path(config['root_dir']),
            unzip_data_dir=Path(config['unzip_data_dir']),
            STATUS_FILE=config['STATUS_FILE'],
            all_schema=schema
        )
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']
        params = self.params['data_transformation']
        create_directories([config['root_dir']])
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config['root_dir']),
            data_path=Path(config['data_path']),
            transformed_train_dir=Path(config['transformed_train_path']),
            transformed_test_dir=Path(config['transformed_test_path']),
            test_size=params['test_size'],
            random_state=params['random_state'],
            polynomial_features_degree=params['polynomial_features_degree'],
            scaling_method=params['scaling_method']
        )
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config['model_trainer']
        params = self.params

        def convert_to_dict(d):
            return {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in d.items()}

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config['root_dir']),
            train_data_path=Path(config['train_data_path']),
            test_data_path=Path(config['test_data_path']),
            test_size=params['data_transformation']['test_size'],
            random_state=params['data_transformation']['random_state'],
            models=params['model_training']['models'],
            param_grids={key: convert_to_dict(value['param_grid']) for key, value in params['hyperparameter_tuning'].items() if isinstance(value, dict) and 'param_grid' in value},
            target_column=params['model_training']['target_column'],
            cv=params['hyperparameter_tuning']['cv'],
            scoring=params['hyperparameter_tuning']['scoring'],
            poly_features_degree=params['data_transformation']['polynomial_features_degree'],
            poly_features_path=Path(config['poly_features_path']),
            model_path=Path(config['model_path']),
            imputation_strategy=params['data_preprocessing']['imputation_strategy'],
            scaling_method=params['data_transformation']['scaling_method'],
            sa_model_name=config['sa_model_name'],
            cv_model_name=config['cv_model_name']
        )
        return model_trainer_config
