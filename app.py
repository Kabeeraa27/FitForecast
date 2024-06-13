from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='template')  


def load_model(filename):
    model_path = os.path.join('artifacts', filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"File {filename} not found in artifacts directory.")

def load_models():
    regression_preprocessor = load_model('regression_preprocessor.pkl')
    regression_model = load_model('regression_model.pkl')
    classification_preprocessor = load_model('classification_preprocessor.pkl')
    classification_model = load_model('classification_model.pkl')
    return regression_preprocessor, regression_model, classification_preprocessor, classification_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_regression', methods=['POST'])
def predict_regression():
    try:
        data = request.json
        input_data = np.array(data['input']).reshape(1, -1)

        # Load the models
        regression_preprocessor, regression_model, _, _ = load_models()

        # Preprocess the input data
        input_transformed = regression_preprocessor.transform(input_data)

        # Predict using the regression model
        prediction = regression_model.predict(input_transformed)
        
        # Return the prediction as JSON response
        return jsonify({'prediction': float(prediction[0])})  # Ensure prediction is converted to a float

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_classification', methods=['POST'])
def predict_classification():
    try:
        data = request.json
        input_data = np.array(data['input']).reshape(1, -1)

        # Load the models
        _, _, classification_preprocessor, classification_model = load_models()

        # Preprocess the input data
        input_transformed = classification_preprocessor.transform(input_data)

        # Predict using the classification model
        prediction = classification_model.predict(input_transformed)
        
        # Return the prediction as JSON response
        return jsonify({'prediction': int(prediction[0])})  # Ensure prediction is converted to an integer

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
