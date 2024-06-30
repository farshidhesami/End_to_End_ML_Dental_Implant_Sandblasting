# Config entity for data ingestion process (data ingestion config)
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)             # This code copy from 01_data_ingestion.py
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    

@dataclass(frozen=True)              # This code copy from 02_data_validation.py
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict
