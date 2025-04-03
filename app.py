from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__, template_folder='template')

# Load the trained model and label encoder
model_path = 'notebook/trained_model'
label_encoder_path = 'notebook/label_encoder'

model = None
label_encoder = None

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print(f"Model path {model_path} does not exist.")
    
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
    else:
        print(f"Label encoder path {label_encoder_path} does not exist.")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model or label encoder not found.'})

    # Get form data
    data = {
        'Gender': request.form['gender'],
        'Age': int(request.form['age']),
        'Height': float(request.form['height']),
        'Weight': float(request.form['weight']),
        'FamOverweightHist': request.form['famOverweightHist'],
        'FreqHighCalFood': request.form['freqHighCalFood'],
        'FoodBtwMeals': request.form['foodBtwMeals'],
        'Smoke': request.form['smoke'],
        'CalorieMonitor': request.form['calorieMonitor'],
        'AlcoholConsump': request.form['alcoholConsump'],
        'Transport': request.form['transport'],
        'FreqVeg': int(request.form['freqVeg']),
        'MainMeals': int(request.form['mainMeals']),
        'WaterIntake': float(request.form['waterIntake']),
        'FreqPhyAct': int(request.form['freqPhyAct']),
        'TechUse': int(request.form['techUse']),
    }

    df = pd.DataFrame([data])

    # Make prediction
    try:
        prediction = model.predict(df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({'error': 'Error making prediction'})

if __name__ == '__main__':
    app.run(debug=True)
