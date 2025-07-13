import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=== SIMPLE Personality Prediction Model (Maximum Generalization) ===\n")

# Load data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Store test ids for submission
test_ids = test_df['id']

# Separate target variable
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']
test_df = test_df.drop('id', axis=1)

print(f"Training data shape: {X.shape}")
print(f"Class distribution: {y.value_counts()}")

# Minimal preprocessing
print("\n1. Minimal Preprocessing...")

# Simple imputation
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_imputer = SimpleImputer(strategy='median')
X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])
test_df[numerical_features] = numerical_imputer.transform(test_df[numerical_features])

if len(categorical_features) > 0:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    test_df[categorical_features] = categorical_imputer.transform(test_df[categorical_features])

# Encode categorical variables
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

X = pd.DataFrame(X_scaled, columns=X.columns)
test_df = pd.DataFrame(test_scaled, columns=test_df.columns)

# Keep original class distribution - NO BALANCING
print("2. Using Original Distribution...")

# Use the most conservative RandomForest possible
print("\n3. Training Ultra-Conservative RandomForest...")

# Very conservative RandomForest
model = RandomForestClassifier(
    n_estimators=200,      # Modest number of trees
    max_depth=6,           # Very shallow trees  
    min_samples_split=20,  # Very conservative splits
    min_samples_leaf=10,   # Large leaves
    max_features='sqrt',   # Limited features per tree
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Robust evaluation with large validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                test_size=0.3,  # Large validation set
                                                random_state=42, 
                                                stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Extensive cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f}")
print(f"CV Std: {cv_scores.std():.4f}")

# Train and validate
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)

print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"CV-Validation gap: {abs(cv_scores.mean() - val_accuracy):.4f}")

# Final training on full dataset
print("\n4. Final Training...")
model.fit(X, y)

# Make predictions
test_predictions = model.predict(test_df)

# Create submission
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': test_predictions
})

submission_df.to_csv('submission_simple.csv', index=False)

print("\n" + "="*50)
print("SIMPLE MODEL RESULTS")
print("="*50)
print(f"CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"CV-Val Gap: {abs(cv_scores.mean() - val_accuracy):.4f}")
print(f"Predicted Test Accuracy: ~{cv_scores.mean():.4f}")
print(f"Submission file: submission_simple.csv")

if abs(cv_scores.mean() - val_accuracy) < 0.01:
    print("✅ Excellent generalization!")
elif abs(cv_scores.mean() - val_accuracy) < 0.02:
    print("✅ Good generalization")
else:
    print("⚠️ May be overfitting")

print("="*50)