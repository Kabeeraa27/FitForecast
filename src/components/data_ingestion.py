import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

<<<<<<< HEAD
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


=======
>>>>>>> 66e6f1154ac29fc8979ada33cf599ec0656515bd


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def Initiate_Data_Ingestion(self):
        logging.info("ENTERED DATA INGESTION METHOD")
        try:
            df = pd.read_csv("notebook\data\Obesity Estimation.csv")
            logging.info("READ DATASET AS DATAFRAME")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("TRAIN TEST SPLIT INITIATED")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("INGESTION OF DATA IS COMPLETED!")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":

    obj=DataIngestion()
<<<<<<< HEAD
    train_data, test_data = obj.Initiate_Data_Ingestion()
  
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.Initiate_Data_Transformation(train_data,test_data)

    
=======
    obj.Initiate_Data_Ingestion()
  
>>>>>>> 66e6f1154ac29fc8979ada33cf599ec0656515bd
