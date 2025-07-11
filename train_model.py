import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
import time

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Store test ids for submission
test_ids = test_df['id']

# Separate target variable
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']
test_df = test_df.drop('id', axis=1)

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['float64']).columns

# Preprocessing pipelines
# Impute missing values
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
test_df[numerical_features] = numerical_imputer.transform(test_df[numerical_features])
test_df[categorical_features] = categorical_imputer.transform(test_df[categorical_features])

# Encode categorical features
for col in categorical_features:
    le_cat = LabelEncoder()
    X[col] = le_cat.fit_transform(X[col])
    test_df[col] = le_cat.transform(test_df[col])

# Feature Scaling
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])

# Create some new features
X['social_score'] = X['Social_event_attendance'] * X['Going_outside']
X['friend_post_ratio'] = X['Friends_circle_size'] / (X['Post_frequency'] + 1)
test_df['social_score'] = test_df['Social_event_attendance'] * test_df['Going_outside']
test_df['friend_post_ratio'] = test_df['Friends_circle_size'] / (test_df['Post_frequency'] + 1)

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models and hyperparameter spaces
models = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [200, 300, 400],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False),
        'params': {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
}

# Perform Randomized Search for each model
best_models = {}
for name, config in models.items():
    print(f"\nPerforming Randomized Search for {name}")
    start_time = time.time()
    n_iter = 10
    random_search = RandomizedSearchCV(
        estimator=config['model'],
        param_distributions=config['params'],
        cv=3,
        n_iter=n_iter,
        random_state=42,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    best_models[name] = random_search.best_estimator_
    print(f"Best Parameters for {name}: {random_search.best_params_}")
    print(f"Best Accuracy for {name}: {random_search.best_score_}")
    print(f"Time taken: {time.time() - start_time} seconds")

# Validate each best model on validation set
validation_results = {}
for name, model in best_models.items():
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    validation_results[name] = accuracy
    print(f"Validation Accuracy for {name}: {accuracy}")

# Select the best performing model
best_model_name = max(validation_results, key=validation_results.get)
best_model = best_models[best_model_name]

# Train the best model on full data
best_model.fit(X, y)

# Make predictions on test data
test_predictions = best_model.predict(test_df)

# Inverse transform predictions to original labels
test_predictions = le.inverse_transform(test_predictions)

# Create submission file
submission_df = pd.DataFrame({'id': test_ids, 'Personality': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")