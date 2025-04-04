# FitForecast 🧠🏃‍♂️

**FitForecast** is an AI-driven tool that predicts obesity levels and BMI using health and lifestyle indicators. It uses **regression** for BMI prediction and **classification algorithms** for obesity category prediction. The project also includes a simple, user-friendly **Tkinter GUI** for real-time predictions.

---

## 🗂️ Dataset Overview

The dataset contains demographic, lifestyle, and behavioral features to help predict:

- **BMI (Body Mass Index)** → *Regression*
- **Obesity Category** → *Classification*

### 📊 Features:
| Feature               | Description                                         |
|-----------------------|-----------------------------------------------------|
| Gender                | Male / Female                                       |
| Age                   | Age in years                                        |
| Height                | In meters                                           |
| Weight                | In kilograms                                        |
| FamOverweightHist     | Family history of overweight (yes/no)              |
| FreqHighCalFood       | Frequency of high-calorie food intake (yes/no)     |
| FreqVeg               | Frequency of vegetable consumption (scale)         |
| MainMeals             | Average number of meals per day                     |
| FoodBtwMeals          | Eating habits between meals                         |
| Smoke                 | Smoking habits (yes/no)                             |
| WaterIntake           | Daily water intake (liters)                         |
| CalorieMonitor        | Monitors daily calorie intake (yes/no)             |
| FreqPhyAct            | Frequency of physical activity (scale)             |
| TechUse               | Time spent using technology (hours/day)            |
| AlcoholConsump        | Alcohol consumption (yes/no)                        |
| Transport             | Mode of transportation                              |
| Obesity               | Target variable for classification                 |
| BMI                   | Target variable for regression                      |

---

## 🤖 Models Used

1. **Regression**
   - `Linear Regression` to predict **BMI**
2. **Classification**
   - `Random Forest Classifier`
   - `Logistic Regression`

---

## 🚀 Running the App
Launch the Tkinter-based GUI:

```bash
python gui_app.py
```

✅ The GUI will open a form where you can input health and lifestyle data to receive:

- Predicted BMI
- Predicted Obesity Category

---

## 🗃️ Project Structure
```bash
FitForecast/
│
├── notebook/               # Jupyter notebooks for EDA & training
├── src/                    # Scripts for data processing & modeling
├── artifacts/              # Saved models, encoders, scalers, etc.
├── gui_app.py              # Tkinter-based GUI app
├── requirements.txt        # Python dependencies
└── README.md               # You're reading it!
```
---

## 🧰 Installation

```bash
git clone https://github.com/Kabeeraa27/FitForecast.git
cd FitForecast
pip install -r requirements.txt
```
---
