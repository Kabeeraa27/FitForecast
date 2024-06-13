#classification_training
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def preprocess_data(X):
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_transformed = preprocessor.fit_transform(X)
    
    return X_transformed, preprocessor

def train_classification_model(X_train, X_test, y_train, y_test):
    # Preprocess data
    X_train_transformed, preprocessor = preprocess_data(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Initialize the classifier
    clf = RandomForestClassifier(random_state=42)
    
    # Fit the classifier
    clf.fit(X_train_transformed, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test_transformed)
    
    # Print classification report
    classification_results = classification_report(y_test, y_pred)
    
    # Save the model and preprocessor
    joblib.dump(clf, 'artifacts/classification_model.pkl')
    joblib.dump(preprocessor, 'artifacts/classification_preprocessor.pkl')
    
    return classification_results



categorical_features = ['Gender', 'FamOverweightHist', 'FreqHighCalFood', 'FoodBtwMeals', 'Smoke', 'CalorieMonitor', 'AlcoholConsump', 'Transport']
numerical_features = ['Age', 'Height', 'Weight', 'FreqVeg', 'MainMeals', 'WaterIntake', 'FreqPhyAct', 'TechUse']

X = df.drop(columns=['Obesity', 'BMI'])
y = df['Obesity']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

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
        "Accuracy": accuracy,          #ACURRACY
        "Precision": precision,        #PRECISION
        "Recall": recall,              #RECALL
        "F1 Score": f1,                #F1 SCORE
        "ROC-AUC": roc_auc,            #ROC-AUC
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
