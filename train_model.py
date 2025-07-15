import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

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

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Training with Gradient Boosting and Hyperparameter Tuning
param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}

gb_model = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Validation
y_pred_val = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
print(f'Validation Accuracy: {accuracy}')

# Train on full data and predict on test data
best_model.fit(X, y)
test_predictions = best_model.predict(test_df)

# Inverse transform predictions to original labels
test_predictions = le.inverse_transform(test_predictions)

# Create submission file
submission_df = pd.DataFrame({'id': test_ids, 'Personality': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")