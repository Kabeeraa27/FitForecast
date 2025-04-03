# BASIC IMPORTS
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as ss
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
import joblib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from sklearn.pipeline import Pipeline

# MODELLING
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier

# PREPROCESSING
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# MODEL SELECTION
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# METRICS
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             classification_report, confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve, precision_score, recall_score, f1_score, log_loss,
                             matthews_corrcoef, cohen_kappa_score, accuracy_score, auc)

# SUPRESSING SOME WARNINGS
warnings.filterwarnings("ignore", message="Precision is ill-defined and being set to 0.0 in labels with no predicted samples.")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*loky.backend.context.*")
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')
sys.stderr = open(os.devnull, 'w')

# DATASET: OBESITY ESTIMATION CLEANED
df = pd.read_csv('C:/Users/kabee/OneDrive/Desktop/DS_PROJECT/notebook/data/Obesity Estimation Cleaned.csv')

numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

if len(numerical_features) > len(categorical_features):
    categorical_features.extend([None] * (len(numerical_features) - len(categorical_features)))
elif len(numerical_features) < len(categorical_features):
    numerical_features.extend([None] * (len(categorical_features) - len(numerical_features)))
else:
    features = pd.DataFrame({
        'Numerical': numerical_features,
        'Categorical': categorical_features
    })

categorical_features = ['Gender', 'FamOverweightHist', 'FreqHighCalFood', 'FoodBtwMeals', 'Smoke', 'CalorieMonitor', 'AlcoholConsump', 'Transport']
numerical_features = ['Age', 'Height', 'Weight', 'FreqVeg', 'MainMeals', 'WaterIntake', 'FreqPhyAct', 'TechUse']

X = df.drop(columns=['Obesity', 'BMI'])
y = df['Obesity']

# ENCODING TARGET VARIABLE
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PREPROCESSING PIPELINE
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# DEFINING CLASSIFICATION MODELS
Model_cls = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "K-Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), 
    "CatBoost Classifier": CatBoostClassifier(verbose=False),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
}

results = []
best_model_name = None
best_model_obj = None
best_accuracy_score = -float('inf')

for model_name, model in Model_cls.items():
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Fit the model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

    result = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
    }
    
    results.append(result)

    if accuracy > best_accuracy_score:
        best_accuracy_score = accuracy
        best_model_name = model_name
        best_model_obj = clf  

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy', ascending=False)

print(f"BEST MODEL IS: {best_model_name}\n")
results_df

y_pred = best_model_obj.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

new_data = pd.DataFrame({
    'Age': [85],
    'Height': [2.85],
    'Weight': [100],
    'FreqVeg': [1],
    'MainMeals': [2],
    'WaterIntake': [1],
    'FreqPhyAct': [2],
    'TechUse': [4],
    'Gender': ['Female'],
    'FamOverweightHist': ['yes'],
    'FreqHighCalFood': ['yes'],
    'FoodBtwMeals': ['Sometimes'],
    'Smoke': ['yes'],
    'CalorieMonitor': ['no'],
    'AlcoholConsump': ['Frequently'],
    'Transport': ['Public Transportation'],
})

new_predictions = best_model_obj.predict(new_data)
predicted_labels = label_encoder.inverse_transform(new_predictions)
print("Predicted Obesity Levels:", predicted_labels[0])

joblib.dump(best_model_obj, 'trained_model')
joblib.dump(label_encoder, 'label_encoder')

label_encoder.classes_ = pd.read_pickle('label_encoder')
best_model = joblib.load('trained_model')

# Initialize tkinter
root = tk.Tk()
root.title("Obesity Prediction")

# Initialize tkinter variables
gender_var = tk.StringVar()
age_var = tk.IntVar()
height_var = tk.DoubleVar()
weight_var = tk.DoubleVar()
fam_overweight_hist_var = tk.StringVar()
freq_high_cal_food_var = tk.StringVar()
freq_veg_var = tk.IntVar()
main_meals_var = tk.IntVar()
food_btw_meals_var = tk.StringVar()
smoke_var = tk.StringVar()
water_intake_var = tk.IntVar()
calorie_monitor_var = tk.StringVar()
freq_phy_act_var = tk.IntVar()
tech_use_var = tk.IntVar()
alcohol_consump_var = tk.StringVar()
transport_var = tk.StringVar()

# Set default values (can be adjusted as needed)
gender_var.set('Female')
age_var.set(21)
height_var.set(1.62)
weight_var.set(64)
fam_overweight_hist_var.set('yes')
freq_high_cal_food_var.set('no')
freq_veg_var.set(2)
main_meals_var.set(3)
food_btw_meals_var.set('Sometimes')
smoke_var.set('no')
water_intake_var.set(2)
calorie_monitor_var.set('no')
freq_phy_act_var.set(0)
tech_use_var.set(1)
alcohol_consump_var.set('no')
transport_var.set('Public Transportation')

# Function to create DataFrame and predict obesity level
def predict_obesity():
    try:
        best_model = joblib.load('trained_model')  # Adjust filename as necessary
        label_encoder = joblib.load('label_encoder')  # Adjust filename as necessary
    except FileNotFoundError:
        messagebox.showerror("Error", "Model or label encoder not found. Please train a model first.")
        return
    
    data = pd.DataFrame({
        'Gender': [gender_var.get()],
        'Age': [age_var.get()],
        'Height': [height_var.get()],
        'Weight': [weight_var.get()],
        'FamOverweightHist': [fam_overweight_hist_var.get()],
        'FreqHighCalFood': [freq_high_cal_food_var.get()],
        'FreqVeg': [freq_veg_var.get()],
        'MainMeals': [main_meals_var.get()],
        'FoodBtwMeals': [food_btw_meals_var.get()],
        'Smoke': [smoke_var.get()],
        'WaterIntake': [water_intake_var.get()],
        'CalorieMonitor': [calorie_monitor_var.get()],
        'FreqPhyAct': [freq_phy_act_var.get()],
        'TechUse': [tech_use_var.get()],
        'AlcoholConsump': [alcohol_consump_var.get()],
        'Transport': [transport_var.get()],
        'Obesity': [0],
        'BMI': [0]
    })
    
    prediction = best_model.predict(data)
    result_label.config(text="Predicted obesity level: {}".format(label_encoder.inverse_transform(prediction)[0]))

# GUI Layout
mainframe = ttk.Frame(root, padding="20")
mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# Labels and Entry/Combobox Widgets
labels = [
    'Gender', 'Age', 'Height', 'Weight', 'FamOverweightHist',
    'FreqHighCalFood', 'FreqVeg', 'MainMeals', 'FoodBtwMeals',
    'Smoke', 'WaterIntake', 'CalorieMonitor', 'FreqPhyAct',
    'TechUse', 'AlcoholConsump', 'Transport'
]
entries = [
    ('Gender', gender_var, ['Male', 'Female']),
    ('Age', age_var, None),
    ('Height', height_var, None),
    ('Weight', weight_var, None),
    ('FamOverweightHist', fam_overweight_hist_var, ['yes', 'no']),
    ('FreqHighCalFood', freq_high_cal_food_var, ['yes', 'no']),
    ('FreqVeg', freq_veg_var, None),
    ('MainMeals', main_meals_var, None),
    ('FoodBtwMeals', food_btw_meals_var, ['Sometimes', 'Frequently', 'Always', 'no']),
    ('Smoke', smoke_var, ['yes', 'no']),
    ('WaterIntake', water_intake_var, None),
    ('CalorieMonitor', calorie_monitor_var, ['yes', 'no']),
    ('FreqPhyAct', freq_phy_act_var, None),
    ('TechUse', tech_use_var, None),
    ('AlcoholConsump', alcohol_consump_var, ['Frequently', 'Sometimes', 'no']),
    ('Transport', transport_var, ['Automobile', 'Public_Transportation', 'Walking', 'Motorbike'])
]

row = 1
for label, var, values in entries:
    ttk.Label(mainframe, text=label + ":").grid(column=1, row=row, sticky=tk.W)
    if values:
        widget = ttk.Combobox(mainframe, width=15, textvariable=var, values=values)
    else:
        widget = ttk.Entry(mainframe, width=15, textvariable=var)
    widget.grid(column=2, row=row, sticky=tk.W)
    row += 1

# Button to make prediction
ttk.Button(mainframe, text="Predict Obesity", command=predict_obesity).grid(column=2, row=row, sticky=tk.W, pady=10)

# Add label to display result
result_label = ttk.Label(mainframe, text="")
result_label.grid(column=1, row=18, columnspan=2, sticky=tk.W)

root.mainloop()

# Function to load the trained model and perform prediction
def predict_and_save():
    try:
        trained_model = joblib.load('trained_model')
    except FileNotFoundError:
        messagebox.showerror("Error", "Model not found. Please train a model first.")
        return
    
    file_path = filedialog.askopenfilename()
    if not file_path:
        messagebox.showerror("Error", "Please select a file.")
        return
    
    try:
        test_data = pd.read_csv(file_path)
        
        test_predictions = trained_model.predict(test_data)
        
        label_encoder = joblib.load('label_encoder')
        
        submission_df = pd.DataFrame({'id': test_data['id'], 
                'NObeyesdad': label_encoder.inverse_transform(test_predictions)})
        
        submission_df.to_csv('submission.csv', index=False)
        
        messagebox.showinfo("Predictions", "Predictions have been made and saved as 'submission.csv'.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# GUI Creation
window = tk.Tk()
window.title("Obesity Prediction")
window.geometry("300x100")

predict_button = tk.Button(window, text="Predict and Save", command=predict_and_save)
predict_button.pack()

window.mainloop()

# CLASSIFICATION REPORT FOR BEST MODEL
print(f"BEST MODEL FOR GENERAL CLASSIFICATION: {best_model_name} \nACCURACY = {(accuracy*100):.4f}%\n") 
print("   --------------| CLASSIFICATION REPORT |----------------\n")
report = classification_report(y_test, y_pred)
print(report)
