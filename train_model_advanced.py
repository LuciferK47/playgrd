import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

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

# Advanced data preprocessing
print("\nAdvanced preprocessing...")

# 1. Advanced imputation using KNN
print("- Advanced imputation...")
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# KNN imputation for numerical features
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
X[numerical_features] = knn_imputer.fit_transform(X[numerical_features])
test_df[numerical_features] = knn_imputer.transform(test_df[numerical_features])

# Mode imputation for categorical features
categorical_imputer = SimpleImputer(strategy='most_frequent')
if len(categorical_features) > 0:
    X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    test_df[categorical_features] = categorical_imputer.transform(test_df[categorical_features])

# 2. Advanced feature engineering
print("- Advanced feature engineering...")

# Create interaction features
X['social_engagement'] = X['Social_event_attendance'] * X['Going_outside'] * X['Friends_circle_size']
X['social_ratio'] = X['Social_event_attendance'] / (X['Time_spent_Alone'] + 1)
X['energy_drain_ratio'] = X['Time_spent_Alone'] / (X['Social_event_attendance'] + 1)
X['communication_score'] = X['Post_frequency'] * X['Friends_circle_size']

# Create the same features for test set
test_df['social_engagement'] = test_df['Social_event_attendance'] * test_df['Going_outside'] * test_df['Friends_circle_size']
test_df['social_ratio'] = test_df['Social_event_attendance'] / (test_df['Time_spent_Alone'] + 1)
test_df['energy_drain_ratio'] = test_df['Time_spent_Alone'] / (test_df['Social_event_attendance'] + 1)
test_df['communication_score'] = test_df['Post_frequency'] * test_df['Friends_circle_size']

# Create polynomial features for key personality indicators
key_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size']
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
poly_features = poly.fit_transform(X[key_features])
poly_feature_names = poly.get_feature_names_out(key_features)

# Add polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
X = pd.concat([X, poly_df], axis=1)

poly_test_features = poly.transform(test_df[key_features])
poly_test_df = pd.DataFrame(poly_test_features, columns=poly_feature_names, index=test_df.index)
test_df = pd.concat([test_df, poly_test_df], axis=1)

# 3. Handle categorical variables with advanced encoding
print("- Advanced categorical encoding...")
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])

# 4. Advanced scaling using RobustScaler (less sensitive to outliers)
print("- Advanced scaling...")
scaler = RobustScaler()
feature_columns = X.columns
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

X = pd.DataFrame(X_scaled, columns=feature_columns)
test_df = pd.DataFrame(test_scaled, columns=feature_columns)

# 5. Feature selection using multiple methods
print("- Feature selection...")
# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# SelectKBest for feature selection
selector = SelectKBest(score_func=chi2, k=min(20, len(X.columns)))
X_selected = selector.fit_transform(X.abs(), y_encoded)  # Use abs() for chi2
test_selected = selector.transform(test_df.abs())

selected_features = X.columns[selector.get_support()]
print(f"Selected features: {len(selected_features)}")

X = pd.DataFrame(X_selected, columns=selected_features)
test_df = pd.DataFrame(test_selected, columns=selected_features)

# 6. Handle class imbalance with advanced sampling
print("- Handling class imbalance...")
class_counts = y.value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

# Combine minority and majority class data
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

# Combine upsampled minority class with majority class
X_balanced = pd.concat([majority_data, minority_upsampled])
y_balanced = pd.Series(list(majority_labels) + minority_labels_upsampled)

print(f"Balanced class distribution: {y_balanced.value_counts()}")

# 7. Advanced model training with ensemble methods
print("\nTraining advanced models...")

# Encode target variable for model training
y_balanced_encoded = le_target.transform(y_balanced)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced_encoded, 
                                                test_size=0.2, 
                                                random_state=42, 
                                                stratify=y_balanced_encoded)

# Define advanced models
models = {
    'XGBoost': XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    ),
    'SVM': SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),
        activation='relu',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
}

# Train and evaluate individual models
trained_models = {}
model_scores = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    trained_models[name] = model
    model_scores[name] = accuracy
    
    print(f"{name} Validation Accuracy: {accuracy:.4f}")

# 8. Create advanced ensemble model
print("\nCreating advanced ensemble...")

# Select top performing models for ensemble
top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:4]
print(f"Top models for ensemble: {[name for name, score in top_models]}")

# Create voting ensemble
ensemble_models = [(name, trained_models[name]) for name, score in top_models]
voting_clf = VotingClassifier(estimators=ensemble_models, voting='soft')
voting_clf.fit(X_train, y_train)

# Evaluate ensemble
y_pred_ensemble = voting_clf.predict(X_val)
ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)
print(f"Ensemble Validation Accuracy: {ensemble_accuracy:.4f}")

# 9. Advanced hyperparameter tuning for best model
print("\nAdvanced hyperparameter tuning...")

# Select best individual model
best_model_name = max(model_scores, key=model_scores.get)
best_model = trained_models[best_model_name]

print(f"Best individual model: {best_model_name} with accuracy: {model_scores[best_model_name]:.4f}")

# Hyperparameter tuning for XGBoost (if it's the best model)
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [4, 6, 8],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    grid_search = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    y_pred_tuned = best_xgb.predict(X_val)
    tuned_accuracy = accuracy_score(y_val, y_pred_tuned)
    
    print(f"Tuned XGBoost Validation Accuracy: {tuned_accuracy:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    if tuned_accuracy > ensemble_accuracy:
        final_model = best_xgb
        final_accuracy = tuned_accuracy
        print("Using tuned XGBoost as final model")
    else:
        final_model = voting_clf
        final_accuracy = ensemble_accuracy
        print("Using ensemble as final model")
else:
    final_model = voting_clf
    final_accuracy = ensemble_accuracy
    print("Using ensemble as final model")

# 10. Final training on full dataset
print(f"\nFinal model training...")
final_model.fit(X_balanced, y_balanced_encoded)

# Make predictions on test set
print("Making predictions on test set...")
test_predictions_encoded = final_model.predict(test_df)

# Decode predictions back to original labels
test_predictions = le_target.inverse_transform(test_predictions_encoded)

# Create submission file
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': test_predictions
})

submission_df.to_csv('submission_advanced.csv', index=False)

print(f"\nFinal Results:")
print(f"Final Model Validation Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"Submission file created: submission_advanced.csv")

# Additional insights
print("\nModel Performance Summary:")
for name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.4f} ({score*100:.2f}%)")

# Feature importance (if available)
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
elif hasattr(final_model, 'estimators_'):
    # For ensemble models, try to get feature importance from first estimator
    if hasattr(final_model.estimators_[0][1], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': final_model.estimators_[0][1].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features (from first estimator):")
        print(feature_importance.head(10))