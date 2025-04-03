import pandas as pd
from joblib import load

class PredictPipeline:
    def __init__(self, model_path='best_model_pipeline.joblib'):
        self.pipeline = load(model_path)

    def predict(self, data):
        # Convert data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Convert numerical columns to correct types if necessary
        numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']
        for col in numerical_cols:
            input_data[col] = pd.to_numeric(input_data[col])
        
        # Predict
        prediction = self.pipeline.predict(input_data)
        return prediction[0]
