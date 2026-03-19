"""
Hyperparameter Optimization using Optuna
Author: Aaryan Rathod

This script uses Optuna, a Bayesian optimization framework, to find optimal
hyperparameters for multiple fraud detection models:
- Logistic Regression
- Random Forest
- XGBoost

The script performs a complete optimization workflow:
1. Hyperparam search for individual models
2. Evaluation on validation set
3. Ensemble creation with optimal parameters
4. Final evaluation with Precision-Recall metric

This approach is more efficient than grid search and better than random search
for high-dimensional hyperparameter spaces.
"""

import pandas as pd
import numpy as np
import optuna
from optuna.trial import Trial

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, r2_score, 
    confusion_matrix, precision_recall_curve, auc, 
    average_precision_score
)
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load credit card fraud dataset."""
    df = pd.read_csv(filepath)
    return df


def split_data(X, y):
    """
    Split data into train (60%), validation (20%), and test (20%).
    Uses stratification to preserve class distribution.
    """
    # First split: 80% temp, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: Split temp into train (75%) and validation (25%)
    # This gives: train=60%, valid=20%, test=20%
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def objective_rf(trial: Trial, X_train, y_train, X_valid, y_valid):
    """
    Objective function for Random Forest hyperparameter optimization.
    
    Optuna will suggest different hyperparameter values and we train/evaluate
    the model, returning the F1 macro score for maximization.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }

    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_valid)
    
    # Return macro F1 score for optimization
    return f1_score(y_valid, preds, average='macro')


def objective_xgb(trial: Trial, X_train, y_train, X_valid, y_valid):
    """
    Objective function for XGBoost hyperparameter optimization.
    
    Tests various XGBoost hyperparameters including:
    - Tree depth and learning rate
    - Sampling ratios (subsample, colsample)
    - Regularization (gamma)
    - Evaluation metric preference
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'eval_metric': trial.suggest_categorical('eval_metric', ['logloss', 'auc']),
        'n_jobs': -1,
        'random_state': 42,
    }

    xgb = XGBClassifier(**params)
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_valid)
    return f1_score(y_valid, preds, average='macro')


def objective_lr(trial: Trial, X_train, y_train, X_valid, y_valid):
    """
    Objective function for Logistic Regression hyperparameter optimization.
    
    Tests various regularization strengths (C) and solver algorithms.
    """
    params = {
        "C": trial.suggest_float("C", 0.01, 10),
        "solver": trial.suggest_categorical("solver", ['liblinear', 'lbfgs']),
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": 'balanced'
    }

    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_valid)
    return f1_score(y_valid, preds, average="macro")


def optimize_randomforest(X_train, y_train, X_valid, y_valid, n_trials=30):
    """
    Run Optuna optimization for Random Forest.
    
    Returns the best parameters found and displays the results.
    """
    print("\n" + "="*70)
    print("OPTUNA OPTIMIZATION: RANDOM FOREST")
    print("="*70)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_rf(trial, X_train, y_train, X_valid, y_valid),
        n_trials=n_trials,
        n_jobs=-1
    )
    
    print(f"\nBest trial parameters:\n{study.best_params}")
    print(f"Best F1 macro score: {study.best_value:.4f}")
    
    return study.best_params


def optimize_xgboost(X_train, y_train, X_valid, y_valid, n_trials=30):
    """
    Run Optuna optimization for XGBoost.
    
    Returns the best parameters found and displays the detailed trial info.
    """
    print("\n" + "="*70)
    print("OPTUNA OPTIMIZATION: XGBOOST")
    print("="*70)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, X_valid, y_valid),
        n_trials=n_trials
    )
    
    print(f"\nBest trial:\n{study.best_trial}")
    print(f"Best F1 macro score: {study.best_value:.4f}")
    
    return study.best_params


def optimize_lr(X_train, y_train, X_valid, y_valid, n_trials=30):
    """
    Run Optuna optimization for Logistic Regression.
    
    Returns the best parameters found.
    """
    print("\n" + "="*70)
    print("OPTUNA OPTIMIZATION: LOGISTIC REGRESSION")
    print("="*70)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_lr(trial, X_train, y_train, X_valid, y_valid),
        n_trials=n_trials
    )
    
    print(f"\nBest trial parameters:\n{study.best_params}")
    print(f"Best F1 macro score: {study.best_value:.4f}")
    
    return study.best_params


def evaluate_ensemble(model, X_valid, y_valid, model_name):
    """Evaluate a model and print classification report and confusion matrix."""
    y_pred = model.predict(X_valid)
    
    print(f"\n{model_name} - Validation Set Results")
    print("="*70)
    print(classification_report(y_valid, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))


def plot_pr_curve(y_valid, y_proba, pr_auc, model_name):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_valid, y_proba)
    
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """Main execution function."""
    # Load data
    filepath = 'creditcard.csv'
    df = load_data(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}\n")
    
    # Prepare data
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_valid.shape}")
    print(f"Test set: {X_test.shape}\n")
    
    # Optimize individual models
    best_rf_params = optimize_randomforest(X_train, y_train, X_valid, y_valid, n_trials=30)
    best_xgb_params = optimize_xgboost(X_train, y_train, X_valid, y_valid, n_trials=30)
    best_lr_params = optimize_lr(X_train, y_train, X_valid, y_valid, n_trials=30)
    
    # Train best models with optimized parameters
    print("\n" + "="*70)
    print("TRAINING BEST MODELS WITH OPTIMAL PARAMETERS")
    print("="*70)
    
    # Logistic Regression
    lr_test = LogisticRegression(**best_lr_params)
    lr_test.fit(X_train, y_train)
    
    # Random Forest
    rf_test = RandomForestClassifier(**best_rf_params)
    rf_test.fit(X_train, y_train)
    
    # XGBoost
    xgb_test = XGBClassifier(**best_xgb_params)
    xgb_test.fit(X_train, y_train)
    
    # Create ensemble with optimal parameters
    print("\n" + "="*70)
    print("CREATING VOTING ENSEMBLE")
    print("="*70)
    
    # Use weights to reflect model performance
    # XGBoost gets higher weight as it typically performs best
    optuna_voting_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_test),
            ('rf', rf_test),
            ('lr', lr_test)
        ],
        voting='soft',
        weights=[2, 1, 1]  # XGBoost: 2x, RF: 1x, LR: 1x
    )
    
    optuna_voting_model.fit(X_train, y_train)
    
    # Evaluate ensemble on validation set
    y_proba = optuna_voting_model.predict_proba(X_valid)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_valid, y_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\nEnsemble PR-AUC: {pr_auc:.4f}")
    
    evaluate_ensemble(optuna_voting_model, X_valid, y_valid, "Optuna Voting Ensemble")
    plot_pr_curve(y_valid, y_proba, pr_auc, "Optuna Voting Ensemble")
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
