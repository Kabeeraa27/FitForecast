# BASIC IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestRegressor, RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.components.data_ingestion import DataIngestionConfig, load_data

# PREPROCESSING
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# MODEL SELECTION
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# METRICS
from sklearn.metrics import (classification_report, accuracy_score, 
                            r2_score, mean_absolute_error, mean_squared_error)

# WARNINGS
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array, task_type):
        try:
            logging.info("SPLITTING TRAIN AND TEST INPUT DATA")
            X_train, y_train, X_test, y_test = (
                train_array.iloc[:, :-1],
                train_array.iloc[:, -1],
                test_array.iloc[:, :-1],
                test_array.iloc[:, -1],
            )


            if task_type == 'regression':
                models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoost Regressor": CatBoostRegressor(verbose=0),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                    "SVR": SVR(),
                    "K-Nearest Neighbors": KNeighborsRegressor(),
                }

                params = {
                    "Random Forest": {"n_estimators": [50, 100, 200]},
                    "Decision Tree": {"max_depth": [3, 5, 7]},
                    "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                    "Linear Regression": {},
                    "XGBRegressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                    "CatBoost Regressor": {"iterations": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                    "AdaBoost Regressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                    "SVR": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7]},
                }

                target_metric = "r2_score"

            elif task_type == 'classification':
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "SVM": SVC(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "AdaBoost": AdaBoostClassifier(),
                    "XGBoost": XGBClassifier(),
                    "CatBoost": CatBoostClassifier(verbose=0),
                }

                params = {
                    "Logistic Regression": {"solver": ["liblinear", "saga"], "C": [0.1, 1, 10]},
                    "Decision Tree": {"criterion": ["gini", "entropy"], "max_depth": [3, 5, 7]},
                    "Random Forest": {"n_estimators": [50, 100, 200], "max_features": ["auto", "sqrt"]},
                    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                    "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                    "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                    "CatBoost": {"iterations": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                }

                target_metric = "accuracy_score"

            else:
                raise ValueError("Invalid task type. Supported types are 'regression' and 'classification'.")

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"BEST MODEL: {best_model_name}, SCORE: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    config = DataIngestionConfig()

    train_data_path = config.train_data_path
    test_data_path = config.test_data_path

    train_data, test_data = load_data(train_data_path, test_data_path)

    print("Type of train_data:", type(train_data))
    print("Shape of train_data:", train_data.shape)
    print("Type of test_data:", type(test_data))
    print("Shape of test_data:", test_data.shape)

    if train_data is not None and test_data is not None:
        print("Train data loaded successfully!")
        print("Test data loaded successfully!")
    else:
        print("Failed to load train and test data.")

    obj = ModelTrainer()
    task_type = 'classification'  # or 'regression'
    best_model_score = obj.initiate_model_training(train_data, test_data, task_type)
    print(f"Best model score: {best_model_score}")
