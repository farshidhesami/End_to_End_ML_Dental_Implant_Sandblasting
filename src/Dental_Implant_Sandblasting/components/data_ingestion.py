import os
import urllib.request as request
import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

    # Method to load the dataset and perform basic EDA
    def load_and_explore_data(self):
        # Load the dataset
        data_file_path = Path(self.config.unzip_dir) / "Sandblasting-Condition.csv"
        if not data_file_path.exists():
            raise FileNotFoundError(f"{data_file_path} not found. Ensure the data file is correctly extracted.")
        
        data = pd.read_csv(data_file_path)
        logger.info(f"Loaded data from {data_file_path} with shape {data.shape}")

        # Display first few rows of the dataset
        logger.info(f"First few rows of the dataset:\n{data.head()}")

        # Data shape
        logger.info(f"Data shape: {data.shape}")

        # Data types and basic info
        logger.info(f"Data info: {data.info()}")

        # Display data types of each column
        logger.info(f"Data types:\n{data.dtypes}")

        # Summary statistics
        logger.info(f"Summary statistics:\n{data.describe(include='all')}")

        # Check for missing values
        missing_values = data.isnull().sum()
        logger.info(f"Missing values by column:\n{missing_values}")

        # Basic EDA: Plotting
        self.basic_eda_plots(data)

    # Method to perform basic EDA plots
    def basic_eda_plots(self, data):
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Result (1=Passed, 0=Failed)', data=data)
        plt.title("Counts of Result (1=Passed, 0=Failed)")
        plt.xlabel("Result")
        plt.ylabel("Count")
        plt.show()

        result_counts = data['Result (1=Passed, 0=Failed)'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Distribution of Results (1=Passed, 0=Failed)")
        plt.axis('equal')
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(y='(Sa) Average of Surface roughness (micrometer)', data=data)
        plt.title("Distribution of Surface Roughness (Sa)")
        plt.ylabel("Surface Roughness (Sa) (µm)")

        plt.subplot(1, 2, 2)
        sns.boxplot(y='Cell Viability (%)', data=data)
        plt.title("Distribution of Cell Viability (%)")
        plt.ylabel("Cell Viability (%)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data['(Sa) Average of Surface roughness (micrometer)'], 
                        y=data['Cell Viability (%)'], hue=data['Result (1=Passed, 0=Failed)'])
        plt.title("Surface Roughness (Sa) vs Cell Viability (%)")
        plt.xlabel("Surface Roughness (Sa) (µm)")
        plt.ylabel("Cell Viability (%)")
        plt.legend(title='Result')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()
