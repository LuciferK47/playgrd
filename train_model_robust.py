import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

print("=== ROBUST Personality Prediction Model (Focus: Generalization) ===\n")

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

# Conservative preprocessing to avoid overfitting
print("\n1. Conservative Data Preprocessing...")

# Identify feature types
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

print(f"Numerical features: {list(numerical_features)}")
print(f"Categorical features: {list(categorical_features)}")

# Simple but robust imputation
print("   - Simple imputation...")
numerical_imputer = SimpleImputer(strategy='median')
X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])
test_df[numerical_features] = numerical_imputer.transform(test_df[numerical_features])

if len(categorical_features) > 0:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    test_df[categorical_features] = categorical_imputer.transform(test_df[categorical_features])

# Conservative feature engineering (fewer features to reduce overfitting)
print("   - Conservative feature engineering...")

# Only create the most reliable features
X['social_activity'] = (X['Social_event_attendance'] + X['Going_outside']) / 2
X['communication_level'] = (X['Post_frequency'] + X['Friends_circle_size']) / 2
X['social_ratio'] = X['Social_event_attendance'] / (X['Time_spent_Alone'] + 1)

# Apply same to test set
test_df['social_activity'] = (test_df['Social_event_attendance'] + test_df['Going_outside']) / 2
test_df['communication_level'] = (test_df['Post_frequency'] + test_df['Friends_circle_size']) / 2
test_df['social_ratio'] = test_df['Social_event_attendance'] / (test_df['Time_spent_Alone'] + 1)

# Encode categorical variables
print("   - Encoding categorical variables...")
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])

# Conservative scaling
print("   - Standard scaling...")
scaler = StandardScaler()
feature_columns = X.columns
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

X = pd.DataFrame(X_scaled, columns=feature_columns)
test_df = pd.DataFrame(test_scaled, columns=feature_columns)

# NO class balancing - use original distribution for better generalization
print("\n2. Using Original Class Distribution (Better Generalization)...")
print(f"Keeping original class distribution: {y.value_counts()}")

# Conservative feature selection
print("\n3. Conservative Feature Selection...")
selector = SelectKBest(score_func=f_classif, k=min(10, len(X.columns)))
X_selected = selector.fit_transform(X, y)
test_selected = selector.transform(test_df)

selected_features = X.columns[selector.get_support()]
print(f"Selected features ({len(selected_features)}): {list(selected_features)}")

X_final = pd.DataFrame(X_selected, columns=selected_features)
test_final = pd.DataFrame(test_selected, columns=selected_features)

# More robust train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_final, y, 
                                                test_size=0.25,  # Larger validation set
                                                random_state=42, 
                                                stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# Conservative model training with regularization
print("\n4. Conservative Model Training (High Regularization)...")

# Focus on models that generalize well with strong regularization
models = {
    'RandomForest_Conservative': RandomForestClassifier(
        n_estimators=300,  # Fewer trees
        max_depth=8,       # Shallower trees
        min_samples_split=10,  # More conservative splits
        min_samples_leaf=5,    # Larger leaves
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting_Conservative': GradientBoostingClassifier(
        n_estimators=200,  # Fewer estimators
        learning_rate=0.05,  # Lower learning rate
        max_depth=4,       # Shallower trees
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,     # More regularization
        random_state=42
    ),
    'LogisticRegression_Regularized': LogisticRegression(
        C=0.5,  # Strong regularization
        solver='liblinear',
        random_state=42,
        max_iter=1000
    ),
    'SVM_Conservative': SVC(
        C=0.5,  # Strong regularization
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    ),
    'MLP_Regularized': MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Smaller network
        activation='relu',
        solver='adam',
        alpha=0.01,  # Strong regularization
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    )
}

# Train models with aggressive cross-validation
trained_models = {}
model_scores = {}
cv_scores_all = {}

# Use more aggressive cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"   Training {name}...")
    
    # Extensive cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    mean_cv_score = cv_scores.mean()
    
    # Train on training set
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    
    trained_models[name] = model
    model_scores[name] = val_accuracy
    cv_scores_all[name] = cv_scores
    
    print(f"     CV Score: {mean_cv_score:.4f} (±{cv_scores.std()*2:.4f})")
    print(f"     Val Accuracy: {val_accuracy:.4f}")
    print(f"     CV-Val Gap: {abs(mean_cv_score - val_accuracy):.4f}")

# Select models with smallest CV-validation gap (best generalization)
print("\n5. Model Selection Based on Generalization...")

generalization_scores = {}
for name in model_scores:
    cv_mean = cv_scores_all[name].mean()
    val_acc = model_scores[name]
    gap = abs(cv_mean - val_acc)
    generalization_scores[name] = cv_mean - gap  # Penalize large gaps
    print(f"{name}: CV={cv_mean:.4f}, Val={val_acc:.4f}, Gap={gap:.4f}, Score={generalization_scores[name]:.4f}")

# Choose best generalizing models
best_models = sorted(generalization_scores.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"\nBest generalizing models: {[name for name, score in best_models]}")

# Create conservative ensemble
ensemble_models = [(name, trained_models[name]) for name, score in best_models]
ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
ensemble.fit(X_train, y_train)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_val)
ensemble_accuracy = accuracy_score(y_val, y_pred_ensemble)
ensemble_cv = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy').mean()

print(f"\nEnsemble CV accuracy: {ensemble_cv:.4f}")
print(f"Ensemble validation accuracy: {ensemble_accuracy:.4f}")
print(f"Ensemble CV-Val gap: {abs(ensemble_cv - ensemble_accuracy):.4f}")

# Conservative hyperparameter tuning
print("\n6. Conservative Hyperparameter Tuning...")
best_model_name = max(generalization_scores, key=generalization_scores.get)
print(f"Best generalizing model: {best_model_name}")

if 'RandomForest' in best_model_name:
    # Conservative parameter ranges
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [6, 8, 10],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [5, 8, 10],
        'max_features': ['sqrt', 'log2']
    }
    
    grid_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        n_iter=15,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    tuned_model = grid_search.best_estimator_
    
    # Test generalization
    tuned_cv = cross_val_score(tuned_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    y_pred_tuned = tuned_model.predict(X_val)
    tuned_accuracy = accuracy_score(y_val, y_pred_tuned)
    tuned_gap = abs(tuned_cv - tuned_accuracy)
    
    print(f"Tuned model CV: {tuned_cv:.4f}")
    print(f"Tuned model Val: {tuned_accuracy:.4f}")
    print(f"Tuned model Gap: {tuned_gap:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Choose model with best generalization
    if tuned_gap < abs(ensemble_cv - ensemble_accuracy) and tuned_cv > ensemble_cv:
        final_model = tuned_model
        final_accuracy = tuned_accuracy
        final_cv = tuned_cv
        print("Using tuned model as final model")
    else:
        final_model = ensemble
        final_accuracy = ensemble_accuracy
        final_cv = ensemble_cv
        print("Using ensemble as final model")
else:
    final_model = ensemble
    final_accuracy = ensemble_accuracy
    final_cv = ensemble_cv
    print("Using ensemble as final model")

# Final training on full dataset
print("\n7. Final Training on Full Dataset...")
final_model.fit(X_final, y)

# Make predictions on test set
print("Making predictions on test set...")
test_predictions = final_model.predict(test_final)

# Create submission file
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': test_predictions
})

submission_df.to_csv('submission_robust.csv', index=False)

# Results summary
print("\n" + "="*60)
print("ROBUST MODEL RESULTS (GENERALIZATION FOCUSED)")
print("="*60)
print(f"Final Model CV Accuracy: {final_cv:.4f}")
print(f"Final Model Validation Accuracy: {final_accuracy:.4f}")
print(f"CV-Validation Gap: {abs(final_cv - final_accuracy):.4f}")
print(f"Submission file: submission_robust.csv")

print("\nModel Generalization Analysis:")
for name in model_scores:
    cv_mean = cv_scores_all[name].mean()
    val_acc = model_scores[name]
    gap = abs(cv_mean - val_acc)
    print(f"  {name}: CV={cv_mean:.4f}, Val={val_acc:.4f}, Gap={gap:.4f}")

print(f"\nPredicted test accuracy: {final_cv:.4f} (±{abs(final_cv - final_accuracy):.4f})")

if abs(final_cv - final_accuracy) < 0.01:
    print("✅ Model shows excellent generalization (gap < 1%)")
elif abs(final_cv - final_accuracy) < 0.02:
    print("✅ Model shows good generalization (gap < 2%)")
else:
    print("⚠️  Model may still be overfitting (gap >= 2%)")

print("\n" + "="*60)
print("ROBUST ANALYSIS COMPLETE")
print("="*60)