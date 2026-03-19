"""
Credit Card Fraud Detection - Main Model Training Script
Author: Aaryan Rathod
Date: 2025

This script implements a complete fraud detection pipeline using multiple models
and hyperparameter tuning. It includes:
- Data loading and preprocessing
- SMOTE for handling class imbalance
- RandomForest and XGBoost model training
- Hyperparameter tuning using RandomizedSearchCV with recall optimization
- Model evaluation with confusion matrices and classification reports
"""

import pandas as pd
import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    make_scorer, recall_score, classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, f1_score
)
from xgboost import XGBClassifier, plot_importance, callback
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, StratifiedKFold    
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def load_data(filepath):
    """Load credit card fraud dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    """Display basic data exploration and statistics."""
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isna().sum())


def prepare_features(df):
    """Separate features and target variable."""
    X = df.drop(columns='Class', axis=1)
    y = df['Class']
    return X, y


def visualize_class_distribution(y):
    """Plot the distribution of fraud vs legitimate transactions."""
    count_class = y.value_counts()   
    plt.figure(figsize=(8, 5))
    plt.bar(count_class.index, count_class.values)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(count_class.index, ['Legitimate (0)', 'Fraud (1)'])
    plt.show()


def apply_smote_resampling(X_train, y_train):
    """Apply SMOTE to balance the training dataset."""
    smote = SMOTE(sampling_strategy='minority')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train RandomForest classifier and evaluate."""
    print("\n=== RANDOM FOREST CLASSIFIER ===")
    rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return rf, y_pred


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier and evaluate."""
    print("\n=== XGBOOST CLASSIFIER ===")
    xgb = XGBClassifier(
        n_jobs=-1,
        class_weight="balanced"
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return xgb, y_pred


def calculate_pos_weight(y_train):
    """Calculate scale_pos_weight for imbalanced classification in XGBoost."""
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print(f"Scale pos weight = {scale_pos_weight:.2f}")
    return scale_pos_weight


def hyperparameter_tuning_xgboost(X_train, y_train, X_test, y_test, scale_pos_weight):
    """
    Perform hyperparameter tuning for XGBoost using RandomizedSearchCV.
    Optimizes for recall to minimize false negatives (missed fraud cases).
    """
    print("\n=== XGBOOST HYPERPARAMETER TUNING ===")
    
    # Initialize XGBoost with AUCPR metric
    xgb = XGBClassifier(
        tree_method="hist",  # Standard histogram method (GPU alternative: gpu_hist)
        use_label_encoder=False, 
        eval_metric='aucpr', 
        random_state=42
    )
    
    # Define parameter distribution for random search
    param_dist = {
        "n_estimators": np.arange(500, 3000, 500),
        "learning_rate": np.linspace(0.02, 0.1, 5),
        "max_depth": np.arange(3, 7, 1),
        "min_child_weight": np.arange(1, 10, 2),
        "gamma": np.linspace(0, 2, 5),
        "subsample": np.linspace(0.6, 0.9, 4),
        "colsample_bytree": np.linspace(0.7, 1.0, 4),
        "scale_pos_weight": [scale_pos_weight]
    }
    
    # Create recall scorer to prioritize catching fraud
    recall_scorer = make_scorer(recall_score, pos_label=1)
    
    # Perform randomized search
    rand_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=40,
        scoring=recall_scorer,
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    # Fit the model
    rand_search.fit(X_train, y_train)
    
    # Evaluate best model
    print(f"\nBest parameters: {rand_search.best_params_}")
    print(f"Best CV recall score: {rand_search.best_score_:.4f}")
    
    best_xgb = rand_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_xgb


def main():
    """Main execution function."""
    # Set file path (adjust as needed)
    filepath = r'creditcard.csv'
    
    # Load and explore data
    df = load_data(filepath)
    explore_data(df)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Visualize class distribution
    visualize_class_distribution(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Apply SMOTE to training data
    X_train_res, y_train_res = apply_smote_resampling(X_train, y_train)
    print(f"\nTraining set after SMOTE - Class distribution:\n{y_train_res.value_counts()}")
    
    # Train baseline models
    rf_model, rf_preds = train_random_forest(X_train_res, y_train_res, X_test, y_test)
    xgb_model, xgb_preds = train_xgboost(X_train_res, y_train_res, X_test, y_test)
    
    # Calculate scale_pos_weight for XGBoost
    X_train2, X_val, y_train2, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    scale_pos_weight = calculate_pos_weight(y_train2)
    
    # Hyperparameter tuning
    best_xgb = hyperparameter_tuning_xgboost(X_train, y_train, X_test, y_test, scale_pos_weight)


if __name__ == "__main__":
    main()
