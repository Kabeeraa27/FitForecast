# regression_training.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

def train_regression_model(X_train_transformed, X_test_transformed, y_train, y_test):
    # Initialize the regressor
    reg = RandomForestRegressor(random_state=42)
    
    # Fit the regressor
    reg.fit(X_train_transformed, y_train)
    
    # Predict on the test set
    y_pred = reg.predict(X_test_transformed)
    
    # Calculate RMSE
    mse = mean_squared_error(y_test, y_pred, squared=False)
    rmse = np.sqrt(mse)
    # Save the model
    joblib.dump(reg, 'artifacts/regression_model.pkl')
    
    return f"MSE: {mse}\n RMSE{rmse}"
