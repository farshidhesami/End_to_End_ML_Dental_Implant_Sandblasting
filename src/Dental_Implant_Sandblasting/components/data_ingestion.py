import os
import urllib.request as request
import zipfile
from pathlib import Path
from Dental_Implant_Sandblasting import logger
from Dental_Implant_Sandblasting.utils.common import get_size
from Dental_Implant_Sandblasting.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # Method to download a file from a URL and save it locally
    def download_file(self):
        local_data_file_path = Path(self.config.local_data_file)
        if not local_data_file_path.exists():
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=local_data_file_path
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(local_data_file_path)}")

    # Method to extract the downloaded zip file into a specified directory
    def extract_zip_file(self):
        unzip_path = Path(self.config.unzip_dir)
        unzip_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            logger.info(f"Extracted {self.config.local_data_file} to {unzip_path}")
