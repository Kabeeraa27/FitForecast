from src.components.data_transformation import load_data, preprocess_data
from src.components.classification_trainer import models_c
from sklearn.pipeline import Pipeline
import joblib

# Load and preprocess data
data = load_data(r'C:\Users\kabee\OneDrive\Desktop\DS_PROJECT\notebook\data\Obesity Estimation Cleaned.csv')
preprocessor, X, y = preprocess_data(data)

# Train models
for model_name, model in models_c.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X, y)  # Assuming full data training for demonstration
    print(f"{model_name} trained successfully.")

# Example of saving preprocessor if needed
joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
