import pandas as pd
from src.components.data_ingestion import load_data, split_data
from src.components.data_transformation import preprocess_data
from src.logger import logging
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

def main():
    # Load and split the data
    file_path = "path/to/your/data.csv"
    target_variable = "target_column"
    test_size = 0.2
    random_state = 42

    df = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df, target_variable, test_size, random_state)

    # Preprocess the data
    categorical_features = ["categorical_feature_1", "categorical_feature_2"]
    strategy = "mean"
    X_train, label_encoders, preprocessor = preprocess_data(X_train, categorical_features, strategy)
    X_test = transform_data(X_test, label_encoders)

    # Define regression models and their hyperparameters
    regression_models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge()
    }
    regression_params = {
        "Linear Regression": {},
        "Lasso": {"alpha": [0.1, 1.0, 10.0]},
        "Ridge": {"alpha": [0.1, 1.0, 10.0]}
    }

    # Train regression models
    best_model, best_score = train_regression_models(X_train, y_train, X_test, y_test, regression_models, regression_params)

    logging.info(f"Best regression model: {best_model} with R2 score: {best_score}")

def train_regression_models(X_train, y_train, X_test, y_test, models, params):
    best_model = None
    best_score = -float('inf')

    for model_name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        grid_search = GridSearchCV(pipeline, params[model_name], scoring='r2', cv=5)
        grid_search.fit(X_train, y_train)

        y_pred = grid_search.predict(X_test)
        score = r2_score(y_test, y_pred)

        if score > best_score:
            best_score = score
            best_model = model_name

    # Save the best model
    joblib.dump(grid_search.best_estimator_, 'model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

    return best_model, best_score

if __name__ == "__main__":
    main()
