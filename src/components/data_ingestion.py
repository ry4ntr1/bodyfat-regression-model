import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(os.getcwd(), 'artifacts', 'train.csv')
    test_data_path: str = os.path.join(os.getcwd(), 'artifacts', 'test.csv')
    raw_data_path: str = os.path.join(os.getcwd(), 'artifacts', 'data.csv')


@dataclass
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            # Read the data whether raw or get it from database
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read Dataset As DataFrame")

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            logging.info("Train Test Split Initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion Complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()