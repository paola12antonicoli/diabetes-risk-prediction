# Healthcare AI & Machine Learning — Project Summary

## Overview

End-to-end machine learning pipeline for diabetes risk prediction using the CDC Diabetes Health Indicators dataset. The project covers data preparation, exploratory analysis, model development, threshold tuning, and interpretability — with a consistent focus on metrics and decisions that make sense in a healthcare context.

This is a data science exercise, not a medical diagnostic tool.

---

## Dataset

- **Source**: CDC Diabetes Health Indicators, UCI ML Repository (ID 891)
- **Size**: 253,680 samples; 21 features
- **Target**: Binary diabetes risk (`Diabetes_binary`)
- **Class distribution**: ~85% negative, ~15% positive — moderately imbalanced
- **Feature types**: Mix of binary indicators (HighBP, HighChol, Smoker, …) and ordinal/continuous variables (BMI, Age, GenHlth, Income, …)

---

## Notebook pipeline

Each notebook is self-contained but intended to be run in order.

**01 — Data Loading and Overview**  
Load the dataset, inspect column types, check for missing values, review the class distribution and feature ranges.

**02 — Exploratory Data Analysis**  
Target distribution, per-feature histograms, feature-vs-target box plots, correlation heatmap, and grouped diabetes rates by Age, BMI, GenHlth, and PhysActivity quintiles. Plots exported to `images/`.

**03 — Preprocessing and Baseline Models**  
Stratified 80/20 split. ColumnTransformer pipeline: median imputation + StandardScaler for numerics, mode imputation for categoricals. Logistic Regression and Random Forest trained as baselines. Multi-metric evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix). Models saved with joblib.

**04 — Model Training and Evaluation**  
Logistic Regression, Random Forest, and Gradient Boosting compared under the same pipeline. ROC and Precision-Recall curves plotted. Threshold sweep (0.3 – 0.7) on the best model to understand the recall/precision trade-off for operational use. Gradient Boosting selected as best model.

**05 — Model Interpretability**  
Feature importance from the tree-based model. SHAP TreeExplainer for global feature attribution (summary bar plot) and individual prediction explanation. Prediction confidence distribution analysis.

---

## Python modules (`src/`)

| File | Purpose |
|------|---------|
| `data_loader.py` | Fetch CDC dataset via ucimlrepo; fallback to local CSV or synthetic sample |
| `preprocessing.py` | Feature-type detection, ColumnTransformer builder, train/test split, class balance check |
| `train_model.py` | Thin wrapper around `model.fit()` for use in scripts |
| `evaluate_model.py` | Returns ROC-AUC and full classification report dict |
| `utils.py` | joblib save/load helpers |
| `__init__.py` | Package-level exports |

---

## Results summary

Model comparison on the test set (threshold = 0.5):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~0.82 | ~0.55 | ~0.48 | ~0.51 | ~0.78 |
| Random Forest | ~0.85 | ~0.62 | ~0.56 | ~0.59 | ~0.81 |
| **Gradient Boosting** | **~0.87** | **~0.65** | **~0.61** | **~0.63** | **~0.84** |

Gradient Boosting performs best on recall and ROC-AUC — the metrics that matter most for a screening-type task where missing positive cases is costly.

### Threshold tuning (Gradient Boosting)

| Threshold | Recall | Precision | Notes |
|-----------|--------|-----------|-------|
| 0.3 | ~0.72 | ~0.45 | High sensitivity, more false alarms |
| 0.4 | ~0.63 | ~0.53 | Good balance for screening |
| 0.5 | ~0.61 | ~0.65 | Default |
| 0.7 | ~0.38 | ~0.75 | High precision, misses more cases |

For a population-level risk flagging scenario, a threshold of 0.4 offers a reasonable balance between catching at-risk individuals and avoiding alert fatigue.

---

## Healthcare considerations

- **Class imbalance**: handled via stratified splits; evaluation relies on recall and ROC-AUC rather than accuracy
- **Threshold flexibility**: the decision boundary is configurable depending on operational constraints
- **Interpretability**: SHAP values allow inspecting individual predictions — relevant for contexts where model decisions need to be explained or audited
- **Scope**: this is a predictive modelling study on survey data; it does not constitute medical advice and is not validated for clinical use

---

## Reproducibility

- All random states fixed to `42`
- Pinned dependency versions in `requirements.txt`
- Fallback data loading (sample CSV or synthetic) so notebooks run offline

---

## Directory structure

```
healthcare-ai-ml/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── 01_data_loading_and_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_and_baseline_model.ipynb
│   ├── 04_model_training_and_evaluation.ipynb
│   └── 05_model_interpretability.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── models/
├── images/
└── reports/
    └── project_summary.md
```

---

**Author**: Paola Antonicoli  
**Last updated**: June 2026  
**License**: MIT
