import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.components.data_transformation import load_data, preprocess_data
from sklearn.pipeline import Pipeline

# Load and preprocess data
file_path = r'C:\Users\kabee\OneDrive\Desktop\DS_PROJECT\notebook\data\Obesity Estimation Cleaned.csv'
X, y = load_data(file_path)
preprocessor, X_processed, y = preprocess_data(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define models to train
models_c = {
    'RandomForest': RandomForestClassifier(),
    'KNeighbors': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'AdaBoost': AdaBoostClassifier(),
    'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'CatBoostClassifier': CatBoostClassifier(verbose=0)
}

# Train and evaluate models
for model_name, model in models_c.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
    
    # Save trained model
    joblib.dump(pipeline, f"artifacts/{model_name}.pkl")
