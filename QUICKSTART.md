# Credit Card Fraud Detection

This repository contains machine learning models for detecting fraudulent credit card transactions.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis:
   ```bash
   python optuna_optimization.py  # Best performing model
   ```

## Project Files

- `main.py` - Main training pipeline
- `fraud_model_pipeline.py` - Baseline+ approach with calibration
- `confusion_matrices.py` - Resampling technique comparison
- `optuna_optimization.py` - Hyperparameter optimization (BEST)
- `xgb_aucpr.py` - XGBoost with AUCPR focus
- `README.md` - Full documentation with results

## Best Results

**Optuna Voting Ensemble**
- F1 Score: 0.9206
- Recall: 0.85+
- Precision: 0.88+
- PR-AUC: 0.92+

See README.md for detailed comparison of all approaches.

## Author

Aaryan Rathod
