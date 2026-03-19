#!/usr/bin/env python3
"""
Fraud Detection Baseline+ Model
Author: Aaryan Rathod

This module implements an advanced fraud detection baseline that goes beyond
naive approaches by:
- Using cost-sensitive learning with class weights and calibrated probabilities
- Employing an ensemble of LogisticRegression and RandomForest with soft voting
- Calibrating predicted probabilities for better confidence estimates
- Optimizing decision threshold to meet target recall while minimizing false positives
- Providing comprehensive evaluation metrics (confusion matrix, precision, recall, F1, AUC)
"""

import argparse
import warnings
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


@dataclass
class Metrics:
    """
    Container for model evaluation metrics.
    Includes confusion matrix components and various performance scores.
    """
    tn: int
    fp: int
    fn: int
    tp: int
    precision: float
    recall: float
    f1: float
    auprc: float
    auroc: float
    threshold: float


def load_data(csv_path: str):
    """
    Load credit card fraud dataset from CSV.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        X: Feature matrix
        y: Target binary classification labels
    """
    csv_path = os.path.normpath(csv_path)
    df = pd.read_csv(csv_path)
    assert 'Class' in df.columns, "CSV must contain target column 'Class'"
    X = df.drop(columns=['Class'])
    y = df['Class'].astype(int)
    return X, y


def make_pipeline():
    """
    Build an ensemble classifier pipeline with:
    - Logistic Regression with StandardScaler
    - RandomForest with class balancing
    - Probability calibration for better confidence estimates
    - Soft voting for final predictions
    
    Returns:
        VotingClassifier: Ensemble model ready for training
    """
    # Logistic Regression pipeline with preprocessing
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs'))
    ])

    # RandomForest with balanced class weights
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )

    # Calibrate both models to improve probability estimates
    lr_cal = CalibratedClassifierCV(lr, cv=3, method='isotonic')
    rf_cal = CalibratedClassifierCV(rf, cv=3, method='isotonic')

    # Ensemble using soft voting with weights favoring RandomForest
    ensemble = VotingClassifier(
        estimators=[('lr', lr_cal), ('rf', rf_cal)],
        voting='soft', 
        weights=[1.0, 1.2],  # Slightly higher weight for RandomForest
        n_jobs=-1
    )
    return ensemble


def pick_threshold(y_true, y_proba, target_recall=0.85, beta=1.0):
    """
    Select optimal decision threshold based on target recall.
    
    This function finds a threshold that achieves at least the target recall
    while maximizing precision. If no threshold can achieve the target recall,
    it falls back to maximizing F-beta score.
    
    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities for positive class
        target_recall: Minimum desired recall (default 0.85)
        beta: Weight for F-beta score (default 1.0 for F1)
        
    Returns:
        float: Optimal decision threshold
    """
    precision, recall, th = precision_recall_curve(y_true, y_proba)
    th = np.append(th, 1.0)

    # Find thresholds meeting target recall
    ok = recall >= target_recall
    if np.any(ok):
        # Among thresholds with sufficient recall, pick the one with max precision
        idx = np.argmax(precision[ok])
        thr = th[ok][idx]
    else:
        # Fallback: maximize F-beta score
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-12)
        thr = th[np.nanargmax(fbeta)]
    return float(thr)


def evaluate(y_true, y_proba, threshold) -> Metrics:
    """
    Evaluate model performance at a specific threshold.
    
    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        threshold: Decision threshold
        
    Returns:
        Metrics: Object containing evaluation metrics
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auprc = average_precision_score(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)
    return Metrics(tn, fp, fn, tp, precision, recall, f1, auprc, auroc, threshold)


def train_and_eval(X, y, target_recall=0.85, test_size=0.2, random_state=42):
    """
    Train model and evaluate on test set.
    
    Args:
        X: Feature matrix
        y: Target labels
        target_recall: Desired recall on validation set
        test_size: Fraction of data for test set
        random_state: Random seed
        
    Returns:
        model: Trained ensemble model
        metrics: Evaluation metrics on test set
    """
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Train ensemble
    model = make_pipeline()
    model.fit(X_train, y_train)

    # Calibrate threshold on training data
    y_proba_val = model.predict_proba(X_train)[:, 1]
    thr = pick_threshold(y_train, y_proba_val, target_recall=target_recall)

    # Evaluate on test set
    y_proba_test = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_proba_test, thr)
    return model, metrics


def print_report(metrics: Metrics):
    """Print formatted evaluation report."""
    print("\n=== Test Set Results ===")
    print(f"Confusion Matrix (TN FP / FN TP): {metrics.tn} {metrics.fp} / {metrics.fn} {metrics.tp}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1-score:  {metrics.f1:.4f}")
    print(f"AUPRC:     {metrics.auprc:.4f}")
    print(f"AUROC:     {metrics.auroc:.4f}")
    print(f"Threshold: {metrics.threshold:.4f}")


if __name__ == "__main__":
    # Configuration
    csv_path = r"creditcard.csv"
    target_recall = 0.85

    # Load data
    X, y = load_data(csv_path)
    
    # Train and evaluate
    _, metrics = train_and_eval(X, y, target_recall=target_recall)
    
    # Display results
    print_report(metrics)
