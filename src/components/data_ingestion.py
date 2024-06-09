import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

def load_data(train_data_path, test_data_path):
    try:
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        return train_data, test_data
    except FileNotFoundError:
        print("One or both of the files could not be found.")
        return None, None

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("ENTERED DATA INGESTION METHOD")
        try:
            df = pd.read_csv("notebook/data/Obesity Estimation Cleaned.csv")
            logging.info("READ DATASET AS DATAFRAME")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("TRAIN TEST SPLIT INITIATED")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("INGESTION OF DATA IS COMPLETED!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    input_feature_train_df, input_feature_test_df, target_feature_train_df, target_feature_test_df = data_transformation.initiate_data_transformation(train_data, test_data)

    config = DataIngestionConfig()

    train_data_path = config.train_data_path
    test_data_path = config.test_data_path

    train_data, test_data = load_data(train_data_path, test_data_path)
    if train_data is not None and test_data is not None:
        print("Train data loaded successfully!")
        print("Test data loaded successfully!")
    else:
        print("Failed to load train and test data.")