import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

print("=== Advanced Personality Prediction Model ===\n")

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
print(f"Class distribution (%): {y.value_counts(normalize=True)}")

# Advanced preprocessing
print("\n1. Advanced Data Preprocessing...")

# Identify feature types
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Numerical features: {list(numerical_features)}")
print(f"Categorical features: {list(categorical_features)}")

# Advanced imputation using KNN for numerical features
print("   - Advanced KNN imputation...")
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
X[numerical_features] = knn_imputer.fit_transform(X[numerical_features])
test_df[numerical_features] = knn_imputer.transform(test_df[numerical_features])

# Mode imputation for categorical features
if len(categorical_features) > 0:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    test_df[categorical_features] = categorical_imputer.transform(test_df[categorical_features])

# Advanced feature engineering
print("   - Advanced feature engineering...")

# Create meaningful interaction features based on psychology
X['social_engagement_index'] = X['Social_event_attendance'] * X['Going_outside'] * X['Friends_circle_size']
X['introversion_index'] = X['Time_spent_Alone'] / (X['Social_event_attendance'] + 1)
X['social_anxiety_index'] = X['Time_spent_Alone'] * (X['Stage_fear'] == 'Yes').astype(int)
X['social_battery_ratio'] = X['Social_event_attendance'] / (X['Time_spent_Alone'] + 1)
X['communication_preference'] = X['Post_frequency'] * X['Friends_circle_size']
X['social_comfort'] = X['Going_outside'] * (X['Stage_fear'] == 'No').astype(int)

# Apply same features to test set
test_df['social_engagement_index'] = test_df['Social_event_attendance'] * test_df['Going_outside'] * test_df['Friends_circle_size']
test_df['introversion_index'] = test_df['Time_spent_Alone'] / (test_df['Social_event_attendance'] + 1)
test_df['social_anxiety_index'] = test_df['Time_spent_Alone'] * (test_df['Stage_fear'] == 'Yes').astype(int)
test_df['social_battery_ratio'] = test_df['Social_event_attendance'] / (test_df['Time_spent_Alone'] + 1)
test_df['communication_preference'] = test_df['Post_frequency'] * test_df['Friends_circle_size']
test_df['social_comfort'] = test_df['Going_outside'] * (test_df['Stage_fear'] == 'No').astype(int)

# Encode categorical variables
print("   - Encoding categorical variables...")
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])

# Scale features
print("   - Scaling features...")
scaler = StandardScaler()
feature_columns = X.columns
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

X = pd.DataFrame(X_scaled, columns=feature_columns)
test_df = pd.DataFrame(test_scaled, columns=feature_columns)

# Advanced class balancing
print("\n2. Advanced Class Balancing...")
print(f"Original class distribution: {y.value_counts()}")

# Create balanced dataset using upsampling
class_counts = y.value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

# Split data by class
minority_data = X[y == minority_class]
majority_data = X[y == majority_class]
minority_labels = y[y == minority_class]
majority_labels = y[y == majority_class]

# Upsample minority class
minority_upsampled = resample(minority_data, 
                            replace=True, 
                            n_samples=len(majority_data), 
                            random_state=42)
minority_labels_upsampled = [minority_class] * len(minority_upsampled)

# Combine datasets
X_balanced = pd.concat([majority_data, minority_upsampled])
y_balanced = pd.Series(list(majority_labels) + minority_labels_upsampled)

print(f"Balanced class distribution: {y_balanced.value_counts()}")

# Feature selection
print("\n3. Feature Selection...")
selector = SelectKBest(score_func=f_classif, k=min(15, len(X_balanced.columns)))
X_selected = selector.fit_transform(X_balanced, y_balanced)
test_selected = selector.transform(test_df)

selected_features = X_balanced.columns[selector.get_support()]
print(f"Selected features ({len(selected_features)}): {list(selected_features)}")

X_final = pd.DataFrame(X_selected, columns=selected_features)
test_final = pd.DataFrame(test_selected, columns=selected_features)

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_final, y_balanced, 
                                                test_size=0.2, 
                                                random_state=42, 
                                                stratify=y_balanced)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# Advanced model ensemble
print("\n4. Advanced Model Training...")

# Define optimized models
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=3,
        min_samples_leaf=1,
        subsample=0.9,
        random_state=42
    ),
    'SVM': SVC(
        C=2.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(200, 100),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        C=1.0,
        solver='liblinear',
        random_state=42
    )
}

# Train individual models
trained_models = {}
model_scores = {}

for name, model in models.items():
    print(f"   Training {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv_score = cv_scores.mean()
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    
    trained_models[name] = model
    model_scores[name] = val_accuracy
    
    print(f"     CV Score: {mean_cv_score:.4f} (±{cv_scores.std()*2:.4f})")
    print(f"     Val Accuracy: {val_accuracy:.4f}")

# Select best models for ensemble
print("\n5. Creating Advanced Ensemble...")
top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"Top 3 models for ensemble: {[name for name, score in top_models]}")

# Create ensemble
ensemble_models = [(name, trained_models[name]) for name, score in top_models]
ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
ensemble.fit(X_train, y_train)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_val)
ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)

print(f"Ensemble validation accuracy: {ensemble_accuracy:.4f}")

# Hyperparameter tuning for the best individual model
print("\n6. Hyperparameter Tuning...")
best_model_name = max(model_scores, key=model_scores.get)
print(f"Best individual model: {best_model_name}")

if best_model_name == 'RandomForest':
    param_grid = {
        'n_estimators': [400, 500, 600],
        'max_depth': [10, 12, 15],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    grid_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        n_iter=15,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_val)
    tuned_accuracy = accuracy_score(y_val, y_pred_tuned)
    
    print(f"Tuned {best_model_name} accuracy: {tuned_accuracy:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Choose best model
    if tuned_accuracy > ensemble_accuracy:
        final_model = tuned_model
        final_accuracy = tuned_accuracy
        print("Using tuned model as final model")
    else:
        final_model = ensemble
        final_accuracy = ensemble_accuracy
        print("Using ensemble as final model")
else:
    final_model = ensemble
    final_accuracy = ensemble_accuracy
    print("Using ensemble as final model")

# Final training on full balanced dataset
print("\n7. Final Training...")
final_model.fit(X_final, y_balanced)

# Make predictions on test set
print("Making predictions on test set...")
test_predictions = final_model.predict(test_final)

# Create submission file
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': test_predictions
})

submission_df.to_csv('submission_improved.csv', index=False)

# Results summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"Final Model Validation Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"Submission file: submission_improved.csv")

print("\nIndividual Model Performance:")
for name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {score:.4f} ({score*100:.2f}%)")

print(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")

# Feature importance analysis
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)