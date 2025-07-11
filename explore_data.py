import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
print("Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print("Dataset shapes:")
print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")
print(f"Sample submission: {sample_submission.shape}")

print("\n" + "="*50)
print("TRAIN DATASET INFO")
print("="*50)
print("\nColumn names and types:")
print(train_df.dtypes)

print("\nFirst few rows:")
print(train_df.head())

print("\nDataset info:")
print(train_df.info())

print("\nMissing values:")
print(train_df.isnull().sum())

print("\nTarget variable distribution:")
if 'Personality' in train_df.columns:
    print(train_df['Personality'].value_counts())
    print(f"Percentage distribution:")
    print(train_df['Personality'].value_counts(normalize=True) * 100)

print("\n" + "="*50)
print("TEST DATASET INFO")
print("="*50)
print("\nTest dataset columns:")
print(test_df.columns.tolist())
print(f"Test dataset shape: {test_df.shape}")

print("\nFirst few rows of test data:")
print(test_df.head())

print("\nMissing values in test data:")
print(test_df.isnull().sum())

print("\n" + "="*50)
print("SAMPLE SUBMISSION INFO")
print("="*50)
print(sample_submission.head())

# Basic statistics for numerical columns
print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS")
print("="*50)
print("\nNumerical columns statistics:")
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 0:
    print(train_df[numerical_cols].describe())

# Check for categorical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {categorical_cols.tolist()}")

if len(categorical_cols) > 1:  # More than just the target
    for col in categorical_cols:
        if col != 'Personality':
            print(f"\nUnique values in {col}:")
            print(train_df[col].value_counts())