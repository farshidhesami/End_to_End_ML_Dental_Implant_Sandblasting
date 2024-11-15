from pathlib import Path
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from Dental_Implant_Sandblasting.utils.common import read_yaml, create_directories
from Dental_Implant_Sandblasting.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig  # Added ModelEvaluationConfig import
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
            sa_lower_bound=params.get("sa_lower_bound"),
            sa_upper_bound=params.get("sa_upper_bound"),
            cell_viability_threshold=params.get("cell_viability_threshold"),
            outlier_capping_method=params.get("outlier_capping_method"),
            outlier_tail=params.get("outlier_tail"),
            outlier_fold=params.get("outlier_fold"),
            log_transform_variable=params.get("log_transform_variable")
        )

        return data_ingestion_config

    # Fetch Data Validation Config
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config['data_validation']
        schema = {**self.schema['COLUMNS'], **self.schema['TARGET_COLUMNS']}
        create_directories([config['root_dir']])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config['root_dir']),
            unzip_data_dir=Path(config['unzip_data_dir']),
            STATUS_FILE=Path(config['STATUS_FILE']),
            all_schema=schema,
            columns_to_convert=self.config['data_ingestion']['columns_to_convert'],
            knn_n_neighbors=self.params.get('knn_n_neighbors', 5),  # Default to 5 if not present
            sa_lower_bound=self.params.get('sa_lower_bound'),
            sa_upper_bound=self.params.get('sa_upper_bound'),
            feature_columns=config['feature_columns'],
            target_column_sa=config['target_column_sa'],
            target_column_cv=config['target_column_cv'],
            test_size=self.params['data_validation']['test_size'],
            random_state=self.params['data_validation']['random_state']
        )
        return data_validation_config

    # Fetch Data Transformation Config
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']
        params = self.params['data_transformation']

        # Create the root directory for data transformation artifacts
        create_directories([config['root_dir']])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config['root_dir']),
            data_path=Path(config['data_path']),
            transformed_train_dir=Path(config['transformed_train_path']),
            transformed_test_dir=Path(config['transformed_test_path']),
            test_size=params['test_size'],
            random_state=params['random_state'],
            polynomial_features_degree=params['polynomial_features_degree'],
            scaling_method=params['scaling_method'],
            lasso_max_iter=params['lasso_max_iter'],
            knn_n_neighbors=params.get('knn_n_neighbors', 5)  # Default to 5 if not present
        )

        return data_transformation_config

    # Fetch Model Trainer Config
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config['model_trainer']
        params = self.params['model_training']

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config['root_dir']),
            transformed_train_dir=Path(config['transformed_train_dir']),
            transformed_test_dir=Path(config['transformed_test_dir']),
            test_size=params['test_size'],
            random_state=params['random_state'],
            param_grid_rf=params['param_grid_rf'],
            param_grid_bagging=params['param_grid_bagging'],
            param_grid_ridge=params['param_grid_ridge'],
            models=params['models'],
            n_iter=params['n_iter'],
            cv=params['cv'],
            verbose=params['verbose'],
            n_jobs=params['n_jobs']
        )
        return model_trainer_config

    # Fetch Model Evaluation Config
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config['model_evaluation']
        params = self.params['model_evaluation']
        
        create_directories([config['root_dir']])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config['root_dir']),
            model_dir=Path(config['model_dir']),
            test_sa_data=Path(config['test_sa_data']),
            test_cv_data=Path(config['test_cv_data']),
            n_estimators_rf=params['n_estimators_rf'],
            max_depth_rf=params['max_depth_rf'],
            min_samples_split_rf=params['min_samples_split_rf'],
            min_samples_leaf_rf=params['min_samples_leaf_rf'],
            bootstrap_rf=params['bootstrap_rf'],
            alpha_ridge=params['alpha_ridge'],
            n_estimators_bagging=params['n_estimators_bagging'],
            max_samples_bagging=params['max_samples_bagging'],
            max_features_bagging=params['max_features_bagging'],
            random_state=params['random_state']
        )
        return model_evaluation_config
