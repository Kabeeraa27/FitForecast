import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from data_transformation import preprocess_data

Model_cls = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "K-Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "CatBoost Classifier": CatBoostClassifier(verbose=False),
    "AdaBoost Classifier": AdaBoostClassifier(n_estimators=100, learning_rate=1.0),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
}

def train_models_and_evaluate(X, y):
    results = []

    # Label encoding for y if it's categorical strings
    if isinstance(y.iloc[0], str):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print("Label Mapping:", label_mapping)

    for model_name, model in Model_cls.items():
        model_pipeline = Pipeline(steps=[
            ('preprocessor', None),  # Placeholder for actual preprocessor
            ('classifier', model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)

        # Adjusted evaluation metrics for multiclass classification
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)  # Use 'weighted' for multiclass
        recall = recall_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
        f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass

        # Check if y contains string labels and convert to numeric for ROC-AUC calculation
        if isinstance(y_test[0], str):
            y_test_numeric = label_encoder.transform(y_test)
            y_pred_numeric = label_encoder.transform(y_pred)
        else:
            y_test_numeric = y_test
            y_pred_numeric = y_pred

        try:
            roc_auc = roc_auc_score(y_test_numeric, model_pipeline.predict_proba(X_test), multi_class='ovr')  # Use 'ovr' for multiclass
        except ValueError:
            roc_auc = 0.0

        result = {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
        }

        results.append(result)

    results_df = pd.DataFrame(results)
    best_model_name = results_df.loc[results_df['F1 Score'].idxmax()]['Model']

    best_model = Model_cls[best_model_name]
    preprocessor = preprocess_data(X)[1]
    best_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])
    best_model_pipeline.fit(X, y)

    with open('artifacts/classification_model.pkl', 'wb') as f:
        pickle.dump(best_model_pipeline, f)

    with open('artifacts/classification_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    return best_model_name, results
