#main_r.py
import pandas as pd
from data_ingestion import load_data
from data_transformation import preprocess_data
from regression_trainer import train_regression_model
from sklearn.model_selection import train_test_split
from src.logger import log_info, log_error
import os
import joblib

def main():
    # Load data
    data_file_path = "C:\\Users\\kabee\\OneDrive\\Desktop\\DS_PROJECT\\notebook\\data\\Obesity Estimation Cleaned.csv"
    data = load_data(data_file_path)
    
    if data is None:
        log_error("DATA LOADING FAILED. EXITING...")
        return

    # Define target column
    target_column = 'BMI'

    if target_column not in data.columns:
        log_error(f"TARGET COLUMN '{target_column}' NOT FOUND IN THE DATASET. EXITING...")
        return

    # Drop the target column
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ensure the artifacts directory exists
    os.makedirs('artifacts', exist_ok=True)

    # Preprocess the data
    X_train_transformed, regression_preprocessor = preprocess_data(X_train)
    if X_train_transformed is None:
        log_error("DATA PREPROCESSING FAILED. EXITING...")
        return
    
    X_test_transformed = regression_preprocessor.transform(X_test)
    
    # Train regression model
    regression_results = train_regression_model(X_train_transformed, X_test_transformed, y_train, y_test)
    print("Regression Results:\n", regression_results)

    # Save the preprocessor
    joblib.dump(regression_preprocessor, 'artifacts/regression_preprocessor.pkl')

if __name__ == "__main__":
    main()
