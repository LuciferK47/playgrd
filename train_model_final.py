import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

# Custom feature engineering transformer
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Advanced psychological feature engineering
        X_new['social_engagement_index'] = X_new['Social_event_attendance'] * X_new['Going_outside'] * X_new['Friends_circle_size']
        X_new['introversion_index'] = X_new['Time_spent_Alone'] / (X_new['Social_event_attendance'] + 1)
        X_new['social_anxiety_index'] = X_new['Time_spent_Alone'] * (X_new['Stage_fear'] == 'Yes').astype(int)
        X_new['social_battery_ratio'] = X_new['Social_event_attendance'] / (X_new['Time_spent_Alone'] + 1)
        X_new['communication_preference'] = X_new['Post_frequency'] * X_new['Friends_circle_size']
        X_new['social_comfort'] = X_new['Going_outside'] * (X_new['Stage_fear'] == 'No').astype(int)
        
        # Advanced interaction features
        X_new['energy_management'] = X_new['Time_spent_Alone'] * (X_new['Drained_after_socializing'] == 'Yes').astype(int)
        X_new['social_confidence'] = X_new['Social_event_attendance'] * (X_new['Stage_fear'] == 'No').astype(int)
        X_new['extroversion_score'] = (X_new['Social_event_attendance'] + X_new['Going_outside'] + X_new['Friends_circle_size']) / 3
        X_new['introversion_score'] = (X_new['Time_spent_Alone'] + (X_new['Stage_fear'] == 'Yes').astype(int) + (X_new['Drained_after_socializing'] == 'Yes').astype(int)) / 3
        
        # Ratio features
        X_new['social_to_alone_ratio'] = X_new['Social_event_attendance'] / (X_new['Time_spent_Alone'] + 0.1)
        X_new['friends_to_posts_ratio'] = X_new['Friends_circle_size'] / (X_new['Post_frequency'] + 0.1)
        X_new['activity_balance'] = X_new['Going_outside'] / (X_new['Time_spent_Alone'] + 0.1)
        
        # Composite scores
        X_new['social_activity_score'] = (X_new['Social_event_attendance'] + X_new['Going_outside']) / 2
        X_new['communication_score'] = (X_new['Post_frequency'] + X_new['Friends_circle_size']) / 2
        
        return X_new

print("=== ULTIMATE Personality Prediction Model ===\n")

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
print("\n1. ULTIMATE Data Preprocessing...")

# Identify feature types
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Numerical features: {list(numerical_features)}")
print(f"Categorical features: {list(categorical_features)}")

# Advanced imputation using KNN for numerical features
print("   - Advanced KNN imputation...")
knn_imputer = KNNImputer(n_neighbors=7, weights='distance')
X[numerical_features] = knn_imputer.fit_transform(X[numerical_features])
test_df[numerical_features] = knn_imputer.transform(test_df[numerical_features])

# Mode imputation for categorical features
if len(categorical_features) > 0:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    test_df[categorical_features] = categorical_imputer.transform(test_df[categorical_features])

# Apply advanced feature engineering
print("   - ULTIMATE feature engineering...")
feature_engineer = AdvancedFeatureEngineer()
X = feature_engineer.transform(X)
test_df = feature_engineer.transform(test_df)

# Encode categorical variables
print("   - Encoding categorical variables...")
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])

# Advanced scaling using RobustScaler
print("   - Advanced scaling...")
scaler = RobustScaler()
feature_columns = X.columns
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

X = pd.DataFrame(X_scaled, columns=feature_columns)
test_df = pd.DataFrame(test_scaled, columns=feature_columns)

# Advanced class balancing with stratified sampling
print("\n2. ULTIMATE Class Balancing...")
print(f"Original class distribution: {y.value_counts()}")

# Create balanced dataset using both upsampling and downsampling
class_counts = y.value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

# Calculate target size (between minority and majority)
target_size = int(len(y) * 0.4)  # 40% of total for each class

# Split data by class
minority_data = X[y == minority_class]
majority_data = X[y == majority_class]
minority_labels = y[y == minority_class]
majority_labels = y[y == majority_class]

# Upsample minority class
minority_upsampled = resample(minority_data, 
                            replace=True, 
                            n_samples=target_size, 
                            random_state=42)
minority_labels_upsampled = [minority_class] * target_size

# Downsample majority class
majority_downsampled = resample(majority_data, 
                               replace=False, 
                               n_samples=target_size, 
                               random_state=42)
majority_labels_downsampled = [majority_class] * target_size

# Combine datasets
X_balanced = pd.concat([majority_downsampled, minority_upsampled])
y_balanced = pd.Series(majority_labels_downsampled + minority_labels_upsampled)

print(f"Balanced class distribution: {y_balanced.value_counts()}")

# Advanced feature selection using multiple methods
print("\n3. ULTIMATE Feature Selection...")

# Use mutual information for feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=min(18, len(X_balanced.columns)))
X_selected = selector.fit_transform(X_balanced, y_balanced)
test_selected = selector.transform(test_df)

selected_features = X_balanced.columns[selector.get_support()]
print(f"Selected features ({len(selected_features)}): {list(selected_features)}")

X_final = pd.DataFrame(X_selected, columns=selected_features)
test_final = pd.DataFrame(test_selected, columns=selected_features)

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_final, y_balanced, 
                                                test_size=0.15, 
                                                random_state=42, 
                                                stratify=y_balanced)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# ULTIMATE model ensemble
print("\n4. ULTIMATE Model Training...")

# Define highly optimized models
models = {
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.85,
        max_features='sqrt',
        random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=700,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=600,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.1,
        algorithm='SAMME',
        random_state=42
    ),
    'SVM': SVC(
        C=3.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(300, 150, 75),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=2000,
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        C=2.0,
        solver='liblinear',
        random_state=42
    )
}

# Train individual models with cross-validation
trained_models = {}
model_scores = {}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"   Training {name}...")
    
    # Extended cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
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
print("\n5. ULTIMATE Ensemble Creation...")
top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:4]
print(f"Top 4 models for ensemble: {[name for name, score in top_models]}")

# Create weighted ensemble based on performance
ensemble_models = [(name, trained_models[name]) for name, score in top_models]
ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
ensemble.fit(X_train, y_train)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_val)
ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)
print(f"Ensemble validation accuracy: {ensemble_accuracy:.4f}")

# Advanced hyperparameter tuning for the best model
print("\n6. ULTIMATE Hyperparameter Tuning...")
best_model_name = max(model_scores, key=model_scores.get)
print(f"Best individual model: {best_model_name}")

if best_model_name == 'GradientBoosting':
    param_grid = {
        'n_estimators': [500, 600, 700],
        'learning_rate': [0.03, 0.05, 0.07],
        'max_depth': [6, 7, 8],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.85, 0.9],
        'max_features': ['sqrt', 'log2']
    }
    
    grid_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        n_iter=25,
        cv=5,
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
print("\n7. ULTIMATE Final Training...")
final_model.fit(X_final, y_balanced)

# Make predictions on test set
print("Making predictions on test set...")
test_predictions = final_model.predict(test_final)

# Create submission file
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': test_predictions
})

submission_df.to_csv('submission_final.csv', index=False)

# Results summary
print("\n" + "="*60)
print("ULTIMATE RESULTS SUMMARY")
print("="*60)
print(f"Final Model Validation Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"Submission file: submission_final.csv")

if final_accuracy >= 0.98:
    print("🎉 ACHIEVEMENT UNLOCKED: 98%+ ACCURACY! 🎉")
else:
    print(f"Close to target! Need {(0.98 - final_accuracy)*100:.2f}% more to reach 98%")

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
print("ULTIMATE ANALYSIS COMPLETE")
print("="*60)