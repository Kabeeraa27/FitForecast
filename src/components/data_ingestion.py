# data_ingestion.py

import pandas as pd
from src.logger import log_info, log_error

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)  # Adjust based on your actual data format
        log_info(f"Data loaded successfully from {file_path}")
        
        # Save the loaded data to artifacts folder
        df.to_csv('artifacts/data.csv', index=False)
        log_info("Data saved as data.csv in artifacts folder")

        return df
    except Exception as e:
        log_error(f"Error loading data from {file_path}: {str(e)}")
        return None
