#main_c.py
import pandas as pd
from data_ingestion import load_data
from classification_trainer import train_classification_model
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
    target_column = 'Obesity'

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

    # Train classification model
    classification_results = train_classification_model(X_train, X_test, y_train, y_test)
    print("Classification Results:\n", classification_results)

if __name__ == "__main__":
    main()
