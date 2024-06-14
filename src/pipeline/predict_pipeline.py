# prediction_pipeline.py

import pickle
import numpy as np
import pandas as pd

def load_model_and_preprocessor():
    with open('artifacts/classification_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('artifacts/classification_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

def predict(features):
    model, preprocessor = load_model_and_preprocessor()
    features_df = pd.DataFrame([features])
    preprocessed_features = preprocessor.transform(features_df)
    prediction = model.predict(preprocessed_features)
    return prediction[0]
