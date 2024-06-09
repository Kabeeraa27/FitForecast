import os
import sys
from dataclasses import dataclass
import pandas as pd
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            target_column_name = "Obesity"  # Update target column name
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)

            # Print column names for debugging
            print("Columns in the train dataset:", train_df.columns)
            print("Columns in the test dataset:", test_df.columns)

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            # Further processing...

            return input_feature_train_df, input_feature_test_df, target_feature_train_df, target_feature_test_df

        except Exception as e:
            raise CustomException(e, sys)
