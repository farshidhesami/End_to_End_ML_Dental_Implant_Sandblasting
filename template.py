# template.py

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

project_name = 'Dental_Implant_Sandblasting'

# List of files to be created
list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/predictor.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration_manager.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/prediction.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_data_validation.py",
    f"src/{project_name}/pipeline/stage_03_data_transformation.py",
    f"src/{project_name}/pipeline/stage_04_model_trainer.py",
    f"src/{project_name}/pipeline/stage_05_model_evaluation.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "artifacts/data_ingestion",
    "artifacts/data_transformation",
    "artifacts/data_validation",
    "artifacts/model_evaluation",
    "artifacts/model_trainer",
    "data/Sandblasting-Condition.csv",
    "deployment-steps",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "application.py",
    "requirements.txt",
    "setup.py",
    "research/01_data_ingestion.ipynb",
    "research/02_data_validation.ipynb",
    "research/03_data_transformation.ipynb",
    "research/04_model_trainer.ipynb",
    "research/05_model_evaluation.ipynb",
    "research/Experiment.ipynb",
    "research/trials.ipynb",
    "research/predictions",       # The "predictions" folder under the research directory is intended to store prediction results generated in the "Experiment.ipynb" notebook as part of the "Predictions (Step 8)" process.  .
    "templates/index.html",
    "templates/results.html"
]

for filepath_str in list_of_files:
    filepath = Path(filepath_str)

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created or verified directory: {filepath.parent}")

    # Create or touch the file
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Created or verified file: {filepath}")
    else:
        logging.info(f"{filepath.name} already exists")
