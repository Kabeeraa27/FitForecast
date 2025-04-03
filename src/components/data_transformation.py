import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Separate features and target
    X = data.drop(['Obesity', 'BMI'], axis=1)
    y = data['Obesity']

    return X, y

def preprocess_data(X, y):
    # Identify categorical and numeric columns
    categorical_cols = ['Gender', 'FamOverweightHist', 'FreqHighCalFood', 'FoodBtwMeals', 
                        'Smoke', 'CalorieMonitor', 'TechUse', 'AlcoholConsump', 'Transport']
    numeric_cols = ['Age', 'Height', 'Weight', 'FreqVeg', 'MainMeals', 'WaterIntake', 'FreqPhyAct']

    # Pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    return preprocessor, X_processed, y
