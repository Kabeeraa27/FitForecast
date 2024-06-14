from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import predict

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_obesity():
    features = {
        'Gender': request.form['gender'],
        'Age': float(request.form['age']),
        'Height': float(request.form['height']),
        'Weight': float(request.form['weight']),
        'FamOverweightHist': request.form['famOverweightHist'],
        'FreqHighCalFood': float(request.form['freqHighCalFood']),
        'FoodBtwMeals': request.form['foodBtwMeals'],
        'Smoke': request.form['smoke'],
        'CalorieMonitor': request.form['calorieMonitor'],
        'AlcoholConsump': request.form['alcoholConsump'],
        'Transport': request.form['transport'],
        'FreqVeg': float(request.form['freqVeg']),
        'MainMeals': float(request.form['mainMeals']),
        'WaterIntake': float(request.form['waterIntake']),
        'FreqPhyAct': float(request.form['freqPhyAct']),
        'TechUse': float(request.form['techUse'])
    }

    prediction = predict(features)
    return render_template('index.html', result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
