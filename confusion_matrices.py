"""
Confusion Matrices Comparison for Fraud Detection
Author: Aaryan Rathod

This script comprehensively evaluates different resampling techniques for handling
imbalanced credit card fraud data:
- Baseline (no resampling)
- RandomOverSampler (duplicates minority examples)
- SMOTE (synthetic oversampling)
- ADASYN (adaptive synthetic sampling)

For each resampling method, we train and evaluate:
- Logistic Regression
- Random Forest
- XGBoost

This comparison helps identify the best balancing strategy for fraud detection.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, r2_score, 
    confusion_matrix, precision_recall_curve, auc, 
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours


def load_data(filepath):
    """Load the credit card fraud dataset."""
    df = pd.read_csv(filepath)
    return df


def prepare_data(df):
    """Prepare features and target."""
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred, y_proba, model_name, resampling_method):
    """Evaluate and print model performance metrics."""
    print(f"\n{'='*60}")
    print(f"{resampling_method.upper()} - {model_name.upper()}")
    print(f"{'='*60}")
    
    # Classification metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Precision-Recall curve metrics
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_proba)
    
    print(f"\nPR-AUC: {pr_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    return cm, pr_auc


def plot_confusion_matrix(cm, title):
    """Visualize confusion matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Non-Fraud', 'Fraud'], 
        yticklabels=['Non-Fraud', 'Fraud']
    )
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title(title)
    plt.show()


def plot_precision_recall_curve(y_true, y_proba, pr_auc, title):
    """Visualize precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# =====================================================================
# BASELINE - WITHOUT RESAMPLING
# =====================================================================
def run_baseline_models(X_train, X_test, y_train, y_test):
    """Train models without any resampling."""
    print("\n" + "="*60)
    print("BASELINE (NO RESAMPLING)")
    print("="*60)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    
    cm_lr, pr_auc_lr = evaluate_model(y_test, y_pred_lr, y_proba_lr, "LogisticRegression", "Baseline")
    plot_confusion_matrix(cm_lr, "Baseline - Logistic Regression")
    plot_precision_recall_curve(y_test, y_proba_lr, pr_auc_lr, "Baseline - LR PR Curve")
    
    # Random Forest
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    cm_rf, pr_auc_rf = evaluate_model(y_test, y_pred_rf, y_proba_rf, "RandomForest", "Baseline")
    plot_confusion_matrix(cm_rf, "Baseline - Random Forest")
    plot_precision_recall_curve(y_test, y_proba_rf, pr_auc_rf, "Baseline - RF PR Curve")
    
    # XGBoost
    xgb = XGBClassifier(n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    
    cm_xgb, pr_auc_xgb = evaluate_model(y_test, y_pred_xgb, y_proba_xgb, "XGBoost", "Baseline")
    plot_confusion_matrix(cm_xgb, "Baseline - XGBoost")
    plot_precision_recall_curve(y_test, y_proba_xgb, pr_auc_xgb, "Baseline - XGB PR Curve")


# =====================================================================
# RANDOM OVERSAMPLER
# =====================================================================
def run_random_oversampler(X_train, X_test, y_train, y_test):
    """Train models with RandomOverSampler resampling."""
    print("\n" + "="*60)
    print("RANDOM OVERSAMPLER")
    print("="*60)
    
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    class_counts = y_resampled.value_counts()
    print(f"Class distribution after resampling:\n{class_counts}\n")
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_resampled, y_resampled)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    
    cm_lr, pr_auc_lr = evaluate_model(y_test, y_pred_lr, y_proba_lr, "LogisticRegression", "RandomOverSampler")
    plot_confusion_matrix(cm_lr, "RandomOverSampler - Logistic Regression")
    
    # Random Forest
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(X_resampled, y_resampled)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    cm_rf, pr_auc_rf = evaluate_model(y_test, y_pred_rf, y_proba_rf, "RandomForest", "RandomOverSampler")
    plot_confusion_matrix(cm_rf, "RandomOverSampler - Random Forest")
    
    # XGBoost
    xgb = XGBClassifier(n_jobs=-1)
    xgb.fit(X_resampled, y_resampled)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    
    cm_xgb, pr_auc_xgb = evaluate_model(y_test, y_pred_xgb, y_proba_xgb, "XGBoost", "RandomOverSampler")
    plot_confusion_matrix(cm_xgb, "RandomOverSampler - XGBoost")


# =====================================================================
# SMOTE (SYNTHETIC MINORITY OVER-SAMPLING TECHNIQUE)
# =====================================================================
def run_smote(X_train, X_test, y_train, y_test):
    """Train models with SMOTE resampling."""
    print("\n" + "="*60)
    print("SMOTE (SYNTHETIC MINORITY OVER-SAMPLING TECHNIQUE)")
    print("="*60)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    class_counts = y_resampled.value_counts()
    print(f"Class distribution after SMOTE:\n{class_counts}\n")
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_resampled, y_resampled)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    
    cm_lr, pr_auc_lr = evaluate_model(y_test, y_pred_lr, y_proba_lr, "LogisticRegression", "SMOTE")
    plot_confusion_matrix(cm_lr, "SMOTE - Logistic Regression")
    
    # Random Forest
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(X_resampled, y_resampled)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    cm_rf, pr_auc_rf = evaluate_model(y_test, y_pred_rf, y_proba_rf, "RandomForest", "SMOTE")
    plot_confusion_matrix(cm_rf, "SMOTE - Random Forest")
    
    # XGBoost
    xgb = XGBClassifier(n_jobs=-1)
    xgb.fit(X_resampled, y_resampled)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    
    cm_xgb, pr_auc_xgb = evaluate_model(y_test, y_pred_xgb, y_proba_xgb, "XGBoost", "SMOTE")
    plot_confusion_matrix(cm_xgb, "SMOTE - XGBoost")


# =====================================================================
# ADASYN (ADAPTIVE SYNTHETIC SAMPLING)
# =====================================================================
def run_adasyn(X_train, X_test, y_train, y_test):
    """Train models with ADASYN resampling."""
    print("\n" + "="*60)
    print("ADASYN (ADAPTIVE SYNTHETIC SAMPLING)")
    print("="*60)
    
    # Preprocess data for ADASYN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train_scaled, y_train)
    
    class_counts = y_resampled.value_counts()
    print(f"Class distribution after ADASYN:\n{class_counts}\n")
    
    class_weights = {0: 1, 1: 5}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, class_weight=class_weights)
    lr.fit(X_resampled, y_resampled)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    cm_lr, pr_auc_lr = evaluate_model(y_test, y_pred_lr, y_proba_lr, "LogisticRegression", "ADASYN")
    plot_confusion_matrix(cm_lr, "ADASYN - Logistic Regression")
    
    # Random Forest
    rf = RandomForestClassifier(n_jobs=-1, class_weight=class_weights)
    rf.fit(X_resampled, y_resampled)
    y_pred_rf = rf.predict(X_test_scaled)
    y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
    
    cm_rf, pr_auc_rf = evaluate_model(y_test, y_pred_rf, y_proba_rf, "RandomForest", "ADASYN")
    plot_confusion_matrix(cm_rf, "ADASYN - Random Forest")
    
    # XGBoost
    xgb = XGBClassifier(class_weight=class_weights, n_jobs=-1)
    xgb.fit(X_resampled, y_resampled)
    y_pred_xgb = xgb.predict(X_test_scaled)
    y_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
    
    cm_xgb, pr_auc_xgb = evaluate_model(y_test, y_pred_xgb, y_proba_xgb, "XGBoost", "ADASYN")
    plot_confusion_matrix(cm_xgb, "ADASYN - XGBoost")


def main():
    """Execute confusion matrix comparisons."""
    # Load and prepare data
    filepath = 'creditcard.csv'
    df = load_data(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}\n")
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Run all resampling methods with all models
    run_baseline_models(X_train, X_test, y_train, y_test)
    run_random_oversampler(X_train, X_test, y_train, y_test)
    run_smote(X_train, X_test, y_train, y_test)
    run_adasyn(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
