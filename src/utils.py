import joblib
import os

def save_object(obj, file_path):
    try:
        joblib.dump(obj, file_path)
    except Exception as e:
        raise ValueError(f"Error saving object to {file_path}: {str(e)}")

def load_object(file_path):
    try:
        if os.path.exists(file_path):
            return joblib.load(file_path)
        else:
            raise FileNotFoundError(f"File {file_path} does not exist.")
    except Exception as e:
        raise ValueError(f"Error loading object from {file_path}: {str(e)}")
