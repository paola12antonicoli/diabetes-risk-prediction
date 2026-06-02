"""Preprocessing utilities for the healthcare AI project."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Features that are always binary (0/1) in the CDC dataset
_KNOWN_BINARY = {
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
}


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical based on dtype and unique-value count."""
    numeric, categorical = [], []
    for col in X.columns:
        if col in _KNOWN_BINARY:
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Return a ColumnTransformer with imputation and scaling for numeric features."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split that preserves the positive-class ratio."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def check_class_balance(y: pd.Series) -> pd.DataFrame:
    """Return a small summary DataFrame with class counts and percentages."""
    counts = y.value_counts().sort_index()
    pct = y.value_counts(normalize=True).sort_index().mul(100).round(1)
    return pd.DataFrame({"count": counts, "pct": pct})


def preprocess_data(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    """Drop constant columns and optionally exclude the target from feature cleaning."""
    df = df.copy()

    if target_col and target_col in df.columns:
        features = df.drop(columns=[target_col])
    else:
        features = df

    # Drop columns with zero variance (identical value in every row)
    constant_cols = [c for c in features.columns if features[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)

    return df
