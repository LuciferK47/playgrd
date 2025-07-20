import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, 
                             BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                             HistGradientBoostingClassifier)
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_predict
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

print("=== ULTIMATE Personality Prediction Model (Target: 97.73% Accuracy) ===\n")

# Advanced Feature Engineering with Psychological Insights
class UltimateFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Core psychological features
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
        
        # Ratio and balance features
        X_new['social_to_alone_ratio'] = X_new['Social_event_attendance'] / (X_new['Time_spent_Alone'] + 0.1)
        X_new['friends_to_posts_ratio'] = X_new['Friends_circle_size'] / (X_new['Post_frequency'] + 0.1)
        X_new['activity_balance'] = X_new['Going_outside'] / (X_new['Time_spent_Alone'] + 0.1)
        X_new['social_activity_score'] = (X_new['Social_event_attendance'] + X_new['Going_outside']) / 2
        X_new['communication_score'] = (X_new['Post_frequency'] + X_new['Friends_circle_size']) / 2
        
        # Advanced psychological constructs
        X_new['social_dominance'] = X_new['Social_event_attendance'] * (X_new['Stage_fear'] == 'No').astype(int) * X_new['Going_outside']
        X_new['anxiety_cluster'] = (X_new['Stage_fear'] == 'Yes').astype(int) * (X_new['Drained_after_socializing'] == 'Yes').astype(int) * X_new['Time_spent_Alone']
        X_new['social_adaptability'] = X_new['Friends_circle_size'] * X_new['Post_frequency'] / (X_new['Time_spent_Alone'] + 1)
        X_new['energy_preservation'] = X_new['Time_spent_Alone'] / ((X_new['Social_event_attendance'] + X_new['Going_outside']) + 1)
        
        # Personality type indicators
        X_new['extreme_introvert_indicator'] = ((X_new['Time_spent_Alone'] > X_new['Time_spent_Alone'].quantile(0.75)) & 
                                               (X_new['Social_event_attendance'] < X_new['Social_event_attendance'].quantile(0.25))).astype(int)
        X_new['extreme_extrovert_indicator'] = ((X_new['Social_event_attendance'] > X_new['Social_event_attendance'].quantile(0.75)) & 
                                               (X_new['Time_spent_Alone'] < X_new['Time_spent_Alone'].quantile(0.25))).astype(int)
        
        # Behavioral consistency features
        X_new['social_consistency'] = np.abs(X_new['Social_event_attendance'] - X_new['Going_outside'])
        X_new['communication_consistency'] = np.abs(X_new['Post_frequency'] - X_new['Friends_circle_size'])
        
        # Polynomial features for key interactions
        key_features = ['Social_event_attendance', 'Time_spent_Alone', 'Friends_circle_size']
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                X_new[f'{feat1}_x_{feat2}'] = X_new[feat1] * X_new[feat2]
                X_new[f'{feat1}_squared'] = X_new[feat1] ** 2
        
        return X_new

# Advanced Stacking Ensemble with Multiple Levels
class DeepStackingClassifier(BaseEstimator):
    def __init__(self, level_0_models, level_1_models, final_estimator, cv=5):
        self.level_0_models = level_0_models
        self.level_1_models = level_1_models
        self.final_estimator = final_estimator
        self.cv = cv
        
    def fit(self, X, y):
        self.fitted_level_0_ = []
        self.fitted_level_1_ = []
        
        # Level 0: Base models
        cv_predictions_level_0 = np.zeros((X.shape[0], len(self.level_0_models)))
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for i, model in enumerate(self.level_0_models):
            cv_preds = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
            cv_predictions_level_0[:, i] = cv_preds
            
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_level_0_.append(fitted_model)
        
        # Level 1: Meta models
        cv_predictions_level_1 = np.zeros((X.shape[0], len(self.level_1_models)))
        
        for i, model in enumerate(self.level_1_models):
            cv_preds = cross_val_predict(model, cv_predictions_level_0, y, cv=skf, method='predict_proba')[:, 1]
            cv_predictions_level_1[:, i] = cv_preds
            
            fitted_model = clone(model)
            fitted_model.fit(cv_predictions_level_0, y)
            self.fitted_level_1_.append(fitted_model)
        
        # Final estimator
        self.final_estimator.fit(cv_predictions_level_1, y)
        
        return self
    
    def predict(self, X):
        # Level 0 predictions
        level_0_preds = np.zeros((X.shape[0], len(self.fitted_level_0_)))
        for i, model in enumerate(self.fitted_level_0_):
            level_0_preds[:, i] = model.predict_proba(X)[:, 1]
        
        # Level 1 predictions
        level_1_preds = np.zeros((X.shape[0], len(self.fitted_level_1_)))
        for i, model in enumerate(self.fitted_level_1_):
            level_1_preds[:, i] = model.predict_proba(level_0_preds)[:, 1]
        
        # Final prediction
        return self.final_estimator.predict(level_1_preds)
    
    def predict_proba(self, X):
        # Level 0 predictions
        level_0_preds = np.zeros((X.shape[0], len(self.fitted_level_0_)))
        for i, model in enumerate(self.fitted_level_0_):
            level_0_preds[:, i] = model.predict_proba(X)[:, 1]
        
        # Level 1 predictions
        level_1_preds = np.zeros((X.shape[0], len(self.fitted_level_1_)))
        for i, model in enumerate(self.fitted_level_1_):
            level_1_preds[:, i] = model.predict_proba(level_0_preds)[:, 1]
        
        # Final prediction
        return self.final_estimator.predict_proba(level_1_preds)

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
knn_imputer = KNNImputer(n_neighbors=9, weights='distance')
X[numerical_features] = knn_imputer.fit_transform(X[numerical_features])
test_df[numerical_features] = knn_imputer.transform(test_df[numerical_features])

# Mode imputation for categorical features
if len(categorical_features) > 0:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
    test_df[categorical_features] = categorical_imputer.transform(test_df[categorical_features])

# Apply ultimate feature engineering
print("   - ULTIMATE feature engineering...")
feature_engineer = UltimateFeatureEngineer()
X = feature_engineer.transform(X)
test_df = feature_engineer.transform(test_df)

# Encode categorical variables
print("   - Encoding categorical variables...")
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])

# Advanced scaling using multiple scalers
print("   - Advanced multi-scale preprocessing...")
# Use different scalers for different feature types
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()
quantile_scaler = QuantileTransformer(n_quantiles=1000, random_state=42)

feature_columns = X.columns
X_standard = standard_scaler.fit_transform(X)
X_robust = robust_scaler.fit_transform(X)
X_quantile = quantile_scaler.fit_transform(X)

test_standard = standard_scaler.transform(test_df)
test_robust = robust_scaler.transform(test_df)
test_quantile = quantile_scaler.transform(test_df)

# Advanced class balancing with sophisticated sampling
print("\n2. ULTIMATE Class Balancing...")
print(f"Original class distribution: {y.value_counts()}")

# Create balanced dataset using hybrid approach
class_counts = y.value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

# Calculate target size
target_size = int(len(y) * 0.42)  # Optimal balance

# Split data by class
minority_data = X[y == minority_class]
majority_data = X[y == majority_class]
minority_labels = y[y == minority_class]
majority_labels = y[y == majority_class]

# Advanced oversampling with noise injection
from sklearn.utils import shuffle
minority_upsampled = resample(minority_data, 
                            replace=True, 
                            n_samples=target_size, 
                            random_state=42)

# Add slight noise to prevent overfitting
noise_factor = 0.01
numerical_cols = minority_data.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    noise = np.random.normal(0, noise_factor * minority_upsampled[col].std(), len(minority_upsampled))
    minority_upsampled[col] += noise

minority_labels_upsampled = [minority_class] * target_size

# Smart majority class downsampling
majority_downsampled = resample(majority_data, 
                               replace=False, 
                               n_samples=target_size, 
                               random_state=42)
majority_labels_downsampled = [majority_class] * target_size

# Combine datasets
X_balanced = pd.concat([majority_downsampled, minority_upsampled])
y_balanced = pd.Series(majority_labels_downsampled + minority_labels_upsampled)

print(f"Balanced class distribution: {y_balanced.value_counts()}")

# Advanced feature selection using ensemble methods
print("\n3. ULTIMATE Feature Selection...")

# Multiple feature selection methods
selectors = {
    'mutual_info': SelectKBest(score_func=mutual_info_classif, k=20),
    'f_classif': SelectKBest(score_func=f_classif, k=20),
    'rfe_rf': RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=20)
}

# Apply each selector and combine results
selected_features_sets = {}
for name, selector in selectors.items():
    selector.fit(X_balanced, y_balanced)
    if hasattr(selector, 'get_support'):
        selected_features_sets[name] = X_balanced.columns[selector.get_support()]
    else:
        selected_features_sets[name] = X_balanced.columns[selector.ranking_ <= 20]

# Consensus feature selection
feature_votes = {}
for features in selected_features_sets.values():
    for feature in features:
        feature_votes[feature] = feature_votes.get(feature, 0) + 1

# Select features that appear in at least 2 out of 3 methods
consensus_features = [f for f, votes in feature_votes.items() if votes >= 2]
print(f"Consensus features selected ({len(consensus_features)}): {consensus_features[:10]}...")

X_selected = X_balanced[consensus_features]
test_selected = test_df[consensus_features]

# Create multiple dataset versions with different scalers
datasets = {
    'standard': (standard_scaler.fit_transform(X_selected), standard_scaler.transform(test_selected)),
    'robust': (robust_scaler.fit_transform(X_selected), robust_scaler.transform(test_selected)),
    'quantile': (quantile_scaler.fit_transform(X_selected), quantile_scaler.transform(test_selected))
}

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_selected, y_balanced, 
                                                test_size=0.15, 
                                                random_state=42, 
                                                stratify=y_balanced)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# ULTIMATE model ensemble with deep stacking
print("\n4. ULTIMATE Deep Stacking Model...")

# Level 0: Diverse base models with optimized hyperparameters
level_0_models = [
    # Gradient Boosting variants
    GradientBoostingClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.85,
        max_features='sqrt',
        random_state=42
    ),
    HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    ),
    
    # Random Forest variants
    RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    ExtraTreesClassifier(
        n_estimators=800,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    
    # Neural Networks
    MLPClassifier(
        hidden_layer_sizes=(400, 200, 100),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=3000,
        random_state=42
    ),
    
    # SVM with different kernels
    SVC(
        C=5.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    ),
    
    # Advanced ensemble
    AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.08,
        algorithm='SAMME',
        random_state=42
    ),
    
    # Gaussian Process
    GaussianProcessClassifier(random_state=42),
    
    # K-Nearest Neighbors
    KNeighborsClassifier(n_neighbors=15, weights='distance'),
    
    # Linear models
    LogisticRegression(C=3.0, solver='liblinear', random_state=42),
    
    # Discriminant Analysis
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]

# Level 1: Meta models
level_1_models = [
    LogisticRegression(C=1.0, solver='liblinear', random_state=42),
    RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
]

# Final estimator
final_estimator = LogisticRegression(C=2.0, solver='liblinear', random_state=42)

# Create and train deep stacking model
print("   Training Deep Stacking Model...")
deep_stacking_model = DeepStackingClassifier(
    level_0_models=level_0_models,
    level_1_models=level_1_models,
    final_estimator=final_estimator,
    cv=7
)

# Train on scaled data
scaler_final = RobustScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_val_scaled = scaler_final.transform(X_val)
test_final_scaled = scaler_final.transform(test_selected)

deep_stacking_model.fit(X_train_scaled, y_train)

# Evaluate the deep stacking model
y_pred_deep = deep_stacking_model.predict(X_val_scaled)
deep_accuracy = accuracy_score(y_val, y_pred_deep)

print(f"Deep Stacking Model Validation Accuracy: {deep_accuracy:.6f} ({deep_accuracy*100:.4f}%)")

# Additional ensemble: Voting classifier with optimized weights
print("\n5. Weighted Voting Ensemble...")
voting_models = [
    ('gb1', level_0_models[0]),
    ('hist_gb', level_0_models[1]),
    ('rf', level_0_models[2]),
    ('et', level_0_models[3]),
    ('mlp', level_0_models[4]),
    ('svm', level_0_models[5])
]

voting_classifier = VotingClassifier(
    estimators=voting_models,
    voting='soft'
)

voting_classifier.fit(X_train_scaled, y_train)
y_pred_voting = voting_classifier.predict(X_val_scaled)
voting_accuracy = accuracy_score(y_val, y_pred_voting)

print(f"Voting Ensemble Validation Accuracy: {voting_accuracy:.6f} ({voting_accuracy*100:.4f}%)")

# Choose the best model
if deep_accuracy > voting_accuracy:
    final_model = deep_stacking_model
    final_accuracy = deep_accuracy
    model_name = "Deep Stacking"
else:
    final_model = voting_classifier
    final_accuracy = voting_accuracy
    model_name = "Voting Ensemble"

print(f"\nBest model: {model_name} with {final_accuracy:.6f} accuracy")

# Final training on full balanced dataset
print("\n6. Final Training on Full Dataset...")
X_full_scaled = scaler_final.fit_transform(X_selected)
final_model.fit(X_full_scaled, y_balanced)

# Make predictions on test set
print("Making predictions on test set...")
test_predictions = final_model.predict(test_final_scaled)

# Create submission file
submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': test_predictions
})

submission_df.to_csv('submission_ultimate.csv', index=False)

# Save the final accuracy
with open('accuracy_ultimate.txt', 'w') as f:
    f.write(str(final_accuracy))

# Results summary
print("\n" + "="*70)
print("ULTIMATE RESULTS SUMMARY")
print("="*70)
print(f"Final Model: {model_name}")
print(f"Validation Accuracy: {final_accuracy:.6f} ({final_accuracy*100:.4f}%)")
print(f"Target Accuracy: 0.977327 (97.7327%)")
print(f"Gap to target: {(0.977327 - final_accuracy)*100:.4f}%")

if final_accuracy >= 0.977327:
    print("🎉 TARGET ACHIEVED: 97.73%+ ACCURACY REACHED! 🎉")
    print("🏆 OPTIMISTIX LEVEL PERFORMANCE UNLOCKED! 🏆")
else:
    print(f"Close to Optimistix target! Need {(0.977327 - final_accuracy)*100:.4f}% more")

print(f"\nSubmission file: submission_ultimate.csv")
print(f"Model used: {model_name}")
print(f"Features selected: {len(consensus_features)}")
print(f"Training samples: {len(X_selected)}")

print("\n" + "="*70)
print("ULTIMATE MODEL ANALYSIS COMPLETE")
print("="*70)