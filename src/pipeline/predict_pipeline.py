#predict_pipeline.py
import pickle
import joblib

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def load_models():
    regression_model = joblib.load('artifacts/regression_model.pkl')
    regression_preprocessor = joblib.load('artifacts/regression_preprocessor.pkl')
    classification_model = joblib.load('artifacts/classification_model.pkl')
    classification_preprocessor = joblib.load('artifacts/classification_preprocessor.pkl')
    return regression_preprocessor, regression_model, classification_preprocessor, classification_model

def predict_regression(input_data, preprocessor, model):
    # Transform input data
    input_transformed = preprocessor.transform([input_data])
    # Predict
    prediction = model.predict(input_transformed)
    return prediction

def predict_classification(input_data, preprocessor, model):
    # Transform input data
    input_transformed = preprocessor.transform([input_data])
    # Predict
    prediction = model.predict(input_transformed)
    return prediction
