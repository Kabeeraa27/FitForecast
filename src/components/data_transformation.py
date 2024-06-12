import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
import os

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Read dataset as DataFrame from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"Error in loading data from {file_path}: {str(e)}")
        raise

def split_data(df, target_variable, test_size=0.2, random_state=42):
    try:
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info("Train-test split initiated.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in splitting data: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    file_path = "C:\\Users\\kabee\\OneDrive\\Desktop\\DS_PROJECT\\notebook\\data\\Obesity Estimation Cleaned.csv"
    target_variable = "Obesity"
    test_size = 0.2
    random_state = 42

    # Load data
    df = load_data(file_path)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_variable, test_size, random_state)
