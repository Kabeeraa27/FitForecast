from classification_trainer import train_models_and_evaluate
from data_ingestion import load_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
import os

logging.basicConfig(level=logging.INFO)

def main():

    os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set to desired number of cores

    # Load data
    file_path = 'path_to_your_data.csv'  # Adjust path to your data
    df = load_data(file_path)

    if df is None:
        logging.error("Failed to load data. Exiting.")
        return

    # Assuming 'Obesity' is your target column
    X = df.drop(columns=['Obesity'])
    y = df['Obesity']

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = label_encoder.fit_transform(X[col])

    # Train models and evaluate
    best_model_name, evaluation_results = train_models_and_evaluate(X, y)

    # Log best model and evaluation results
    logging.info(f"Best Model: {best_model_name}")
    logging.info("Evaluation Results:")
    for result in evaluation_results:
        logging.info(result)

    logging.info("Saving pickle files...")
    # Log saving of pickle files
    logging.info("Pickle files saved in artifacts folder.")

    # Return best model and evaluation results
    return best_model_name, evaluation_results

if __name__ == "__main__":
    main()
