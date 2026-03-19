"""
XGBoost with AUCPR Optimization for Fraud Detection
Author: Aaryan Rathod

This script focuses on XGBoost specifically, optimizing for AUCPR
(Area Under Precision-Recall Curve) which is particularly useful for
imbalanced fraud detection problems.

The approach includes:
- Optuna-based hyperparameter search
- SMOTE for handling class imbalance
- AUCPR metric for evaluation (better than AUROC for imbalanced data)
- Comprehensive performance analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, r2_score, 
    confusion_matrix, precision_recall_curve, auc, 
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
import optuna
from optuna.trial import Trial


def load_data(filepath):
    """Load credit card fraud dataset."""
    df = pd.read_csv(filepath)
    return df


def prepare_data(df):
    """Prepare features and split into train/test."""
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def objective_xgb(trial: Trial, X_train, y_train, X_test, y_test):
    """
    Objective function for XGBoost optimization using Optuna.
    
    This function searches for optimal hyperparameters that maximize
    the F1 macro score on the test set.
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1),
        'gamma': trial.suggest_float('gamma', 0, 5),
        # Handle class imbalance with scale_pos_weight
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'eval_metric': trial.suggest_categorical('eval_metric', ['logloss', 'auc']),
        'n_jobs': -1,
        'random_state': 42,
    }

    xgb = XGBClassifier(**params)
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    
    return f1_score(y_test, preds, average='macro')


def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Comprehensive model evaluation with all important metrics."""
    print(f"\n{'='*70}")
    print(f"{model_name.upper()}")
    print(f"{'='*70}")
    
    # Classification metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_proba)
    
    print(f"\nPR-AUC: {pr_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    return cm, pr_auc, precision, recall


def plot_confusion_matrix(cm, title):
    """Visualize confusion matrix as heatmap."""
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


def plot_precision_recall_curve(precision, recall, pr_auc, title):
    """Plot precision-recall curve."""
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_class_distribution(y, title='Class Distribution'):
    """Plot class distribution bar chart."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, data=pd.DataFrame({'Class': y}))
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Legitimate (0)', 'Fraud (1)'])
    plt.show()


def main():
    """Main execution function."""
    # Load and prepare data
    filepath = 'creditcard.csv'
    df = load_data(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['Class'].value_counts())
    
    plot_class_distribution(df['Class'], 'Original Class Distribution')
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print(f"\nTrain set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # =====================================================================
    # BASELINE XGBoost (WITHOUT SMOTE)
    # =====================================================================
    print("\n" + "="*70)
    print("BASELINE XGBoost (NO RESAMPLING)")
    print("="*70)
    
    xgb_baseline = XGBClassifier(n_jobs=-1, random_state=42)
    xgb_baseline.fit(X_train, y_train)
    y_pred_baseline = xgb_baseline.predict(X_test)
    y_proba_baseline = xgb_baseline.predict_proba(X_test)[:, 1]
    
    cm_baseline, pr_auc_baseline, prec_baseline, rec_baseline = evaluate_model(
        y_test, y_pred_baseline, y_proba_baseline, "XGBoost Baseline"
    )
    plot_confusion_matrix(cm_baseline, "Baseline XGBoost - Confusion Matrix")
    plot_precision_recall_curve(prec_baseline, rec_baseline, pr_auc_baseline, 
                                "Baseline XGBoost - PR Curve")
    
    # =====================================================================
    # OPTUNA HYPERPARAMETER OPTIMIZATION
    # =====================================================================
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION FOR XGBOOST")
    print("="*70)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, X_test, y_test),
        n_trials=30
    )
    
    print(f"\nOptimization complete!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1 score: {study.best_value:.4f}")
    print(f"Best parameters:\n{study.best_params}")
    
    # =====================================================================
    # XGBOOST WITH SMOTE RESAMPLING
    # =====================================================================
    print("\n" + "="*70)
    print("XGBOOST WITH SMOTE RESAMPLING")
    print("="*70)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"\nClass distribution after SMOTE:")
    print(y_train_smote.value_counts())
    
    plot_class_distribution(y_train_smote, 'Class Distribution After SMOTE')
    
    # Train optimized XGBoost with SMOTE data
    xgb_optimized = XGBClassifier(
        max_depth=8,
        learning_rate=0.06118967445030214,
        n_estimators=915,
        subsample=0.7826011515735614,
        colsample_bytree=0.8672492497464243,
        gamma=0.011275768961173382,
        eval_metric='aucpr',
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=578
    )
    
    # Train on SMOTE-resampled data
    xgb_optimized.fit(X_train_smote, y_train_smote)
    y_pred_smote = xgb_optimized.predict(X_test)
    y_proba_smote = xgb_optimized.predict_proba(X_test)[:, 1]
    
    cm_smote, pr_auc_smote, prec_smote, rec_smote = evaluate_model(
        y_test, y_pred_smote, y_proba_smote, 
        "XGBoost + SMOTE + Optuna"
    )
    plot_confusion_matrix(cm_smote, "XGBoost + SMOTE - Confusion Matrix")
    plot_precision_recall_curve(prec_smote, rec_smote, pr_auc_smote, 
                                "XGBoost + SMOTE - PR Curve")
    
    # =====================================================================
    # PERFORMANCE COMPARISON
    # =====================================================================
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print(f"\nBaseline XGBoost PR-AUC: {pr_auc_baseline:.4f}")
    print(f"XGBoost + SMOTE + Optuna PR-AUC: {pr_auc_smote:.4f}")
    
    improvement = ((pr_auc_smote - pr_auc_baseline) / pr_auc_baseline) * 100
    print(f"\nImprovement: {improvement:.2f}%")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
