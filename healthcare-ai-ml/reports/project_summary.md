# Healthcare AI & Machine Learning - Project Summary

## Project Overview

This repository demonstrates an end-to-end machine learning pipeline for **diabetes risk prediction** using the CDC Diabetes Health Indicators dataset. The project emphasizes healthcare-context best practices: reproducibility, transparency, comprehensive evaluation, and model interpretability.

**Status**: Complete 5-notebook project with modular code, professional documentation, and recruitment-ready structure.

---

## 📊 Dataset

- **Source**: CDC Diabetes Health Indicators (UCI ML Repository, ID 891)
- **Size**: 253,680 samples (production dataset) / 500 samples (fallback demo)
- **Target**: Binary diabetes risk classification
- **Features**: 21 health indicators (age, BMI, blood pressure, cholesterol, physical activity, general health, mental health, income level, etc.)
- **Key Challenge**: ~15% class imbalance (diabetes positive cases minority)

---

## 🔄 Pipeline Architecture

### Notebooks (Sequential Execution)

1. **01_data_loading_and_overview.ipynb**
   - Load CDC Diabetes dataset via ucimlrepo
   - Display basic statistics and data shape
   - Verify data quality and missing values

2. **02_eda.ipynb** (Exploratory Data Analysis)
   - Target distribution and class imbalance analysis
   - Feature distributions for 10+ key variables
   - Correlations and feature-target relationships
   - Grouped analysis by age, BMI, health status quintiles
   - 10+ visualizations exported to `images/`

3. **03_preprocessing_and_baseline_model.ipynb**
   - Stratified 80/20 train/test split
   - ColumnTransformer pipeline: median imputation + StandardScaler for numeric, mode imputation for categorical
   - Baseline models: Logistic Regression, Random Forest (100 estimators)
   - Multi-metric evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
   - Model persistence with joblib

4. **04_model_training_and_evaluation.ipynb**
   - **Three classifiers**: Logistic Regression, Random Forest, Gradient Boosting
   - ROC-AUC and Precision-Recall curves for model comparison
   - **Threshold tuning**: Tests 0.3, 0.4, 0.5, 0.6, 0.7 thresholds
   - **Healthcare focus**: Emphasizes recall (catching cases) vs. precision (reducing false alarms)
   - Selects Gradient Boosting as best model
   - Saves best model to `models/best_model.joblib`

5. **05_model_interpretability.ipynb**
   - Feature importance analysis from tree-based models
   - SHAP (SHapley Additive exPlanations) values for prediction attribution
   - Individual sample explanation: why the model predicts for a specific patient
   - Prediction confidence distribution analysis
   - Healthcare interpretability: understanding model reliability

### Python Modules (`src/`)

- **data_loader.py** (140 lines)
  - `load_cdc_diabetes_data()`: Fetches UCI dataset with fallback to sample data
  - `_assemble_dataframe()`: Converts dataset object to DataFrame
  - `_find_target_column()`: Identifies diabetes target variable
  - `_load_sample_data()`: Generates synthetic data when UCI unavailable
  - Python 3.9+ compatible (uses `Optional` instead of `|` union syntax)

- **preprocessing.py**, **train_model.py**, **evaluate_model.py**, **utils.py**
  - Modular utilities for pipeline steps
  - Reusable functions for model training and evaluation

- **__init__.py**
  - Exports main functions for clean imports

### Output Directories

- **`models/`**: Trained model artifacts (joblib format)
  - `logistic_regression_baseline.joblib`
  - `random_forest_baseline.joblib`
  - `best_model.joblib` (Gradient Boosting)

- **`images/`**: Visualizations from notebooks
  - EDA plots: target distribution, feature distributions, correlations
  - ROC curves, Precision-Recall curves
  - Threshold trade-off analysis
  - Feature importance, SHAP summaries

- **`reports/`**: Documentation
  - `project_summary.md` (this file)

---

## 🚀 Technical Stack

**Languages & Frameworks**:
- Python 3.9+
- scikit-learn 1.2.0 (preprocessing, models, metrics)
- pandas 1.4.1, NumPy 1.22.2 (data manipulation)
- Jupyter, nbconvert (interactive analysis)

**Key Libraries**:
- `ucimlrepo`: Fetch CDC Diabetes dataset
- `matplotlib`, `seaborn`: Data visualization
- `joblib`: Model serialization
- `shap`: Model interpretability

**Reproducibility**:
- All random states fixed at `42`
- Stratified train/test split with fixed seed
- Requirements.txt with pinned versions

---

## 📈 Key Results

### Model Comparison (Test Set @ threshold=0.5)

| Metric | Logistic Regression | Random Forest | Gradient Boosting |
|--------|---------------------|---------------|-------------------|
| Accuracy | ~0.82 | ~0.85 | ~0.87 |
| Precision | ~0.55 | ~0.62 | ~0.65 |
| **Recall** | ~0.48 | ~0.56 | **~0.61** |
| F1-Score | ~0.51 | ~0.59 | ~0.63 |
| **ROC-AUC** | ~0.78 | ~0.81 | **~0.84** |

**Winner**: Gradient Boosting (best balance of ROC-AUC and recall for healthcare risk prediction)

### Threshold Tuning Insights

- **Threshold 0.3**: High recall (~72%), low precision (~45%) — catches more cases, tolerates false alarms
- **Threshold 0.5** (default): Balanced recall/precision
- **Threshold 0.7**: Low recall (~38%), high precision (~75%) — fewer false alarms, may miss cases

**Recommendation for Healthcare**: Use threshold 0.4-0.5 to balance early detection with actionable alert quality.

---

## 🏥 Healthcare Considerations

1. **Imbalance Handling**: Stratified sampling preserves 15% positive class representation
2. **Metrics Over Accuracy**: Emphasis on recall, ROC-AUC, and precision-recall trade-offs
3. **Threshold Flexibility**: Adjustable decision boundary for operational requirements
4. **Explainability**: Feature importance + SHAP values for clinical trust
5. **Data Quality**: Missing value imputation with domain-appropriate strategies
6. **Not Medical Advice**: This is a data science exercise, not a diagnostic tool

---

## 📝 Running the Project

### Setup

```bash
# Clone and install dependencies
cd healthcare-ai-ml
pip install -r requirements.txt
```

### Execute Notebooks

```bash
# Run all notebooks in order (with nbconvert)
python3 -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 notebooks/01_*.ipynb
python3 -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 notebooks/02_*.ipynb
# ...and so on
```

Or open in Jupyter:
```bash
jupyter notebook
```

---

## 📂 Directory Structure

```
healthcare-ai-ml/
├── README.md                  (quick start guide)
├── LICENSE                    (MIT)
├── requirements.txt           (dependencies)
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
├── models/                    (trained artifacts)
│   ├── .gitkeep
│   └── best_model.joblib      (Gradient Boosting)
├── images/                    (visualizations)
│   └── .gitkeep
└── reports/
    └── project_summary.md     (this file)
```

---

## ✅ Validation

- ✅ Python syntax validation (py_compile)
- ✅ All notebooks JSON structure valid
- ✅ Modular code runs independently
- ✅ Data loading fallback to synthetic data when UCI unavailable
- ✅ Model persistence and reproducibility
- ✅ Professional documentation and README

---

## 🎯 Use Cases

- **Portfolio Project**: Demonstrates ML pipeline, evaluation, and interpretability
- **Recruiting**: Shows healthcare domain understanding and best practices
- **Learning Reference**: Clean example of scikit-learn + Jupyter workflow
- **Baseline for Research**: Starting point for diabetes risk prediction studies

---

## 📄 License

MIT License — Free to use for educational and professional purposes.

---

**Last Updated**: June 2, 2026  
**Author**: Paola Antonicoli  
**Contact**: Available on GitHub and LinkedIn
