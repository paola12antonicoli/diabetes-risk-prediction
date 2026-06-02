# Healthcare AI & Machine Learning

Healthcare machine learning project for diabetes risk prediction using the CDC Diabetes Health Indicators dataset.

This project is a data science exercise focused on data preparation, exploratory analysis, model development, and interpretability. It is presented as a healthcare risk prediction study, not a medical diagnostic tool.

## Project scope

- Load and validate the CDC Diabetes Health Indicators dataset
- Explore key features and population-level patterns
- Build baseline and improved predictive models
- Evaluate model performance with clear metrics
- Explain model behavior with interpretable methods

## Structure

- `notebooks/` — sequential analysis notebooks
- `src/` — reusable data, preprocessing, training, evaluation, and utility code
- `reports/` — project summary and notes
- `images/` — charts and visual outputs
- `models/` — trained model artifacts and checkpoints

## Dataset

The dataset contains CDC health survey information for diabetes risk prediction. It includes lifestyle and clinical indicators, with a target variable related to diabetes status.

## Getting started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks in order:
   - `notebooks/01_data_loading_and_overview.ipynb`
   - `notebooks/02_eda.ipynb`
   - `notebooks/03_preprocessing_and_baseline_model.ipynb`
   - `notebooks/04_model_training_and_evaluation.ipynb`
   - `notebooks/05_model_interpretability.ipynb`

## Notes

This repository is intended for project documentation and reproducible analysis. It is not medical advice and should not be used for clinical decision-making.
