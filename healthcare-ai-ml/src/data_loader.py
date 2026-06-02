"""Data loading utilities for the healthcare AI project."""

from typing import Optional, Tuple

import pandas as pd


def load_cdc_diabetes_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Fetch the CDC Diabetes Health Indicators dataset and return (X, y, df).

    Tries the UCI ML Repository first; falls back to a local sample CSV if
    the network request fails or the package is not installed.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return _load_sample_data()

    try:
        dataset = fetch_ucirepo("CDC Diabetes Health Indicators")
        df = _assemble_dataframe(dataset)
    except Exception:
        return _load_sample_data()

    target_column = _find_target_column(df)
    if target_column is None:
        raise ValueError(
            "Could not identify the diabetes target column. "
            "Expected a column named 'Diabetes_binary'."
        )

    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    return X, y, df


def _assemble_dataframe(dataset) -> pd.DataFrame:
    """Convert a ucimlrepo dataset object to a single pandas DataFrame."""
    data = getattr(dataset, "data", None)

    # Preferred path: pre-assembled original DataFrame from ucimlrepo >= 0.0.3
    if data is not None and hasattr(data, "original") and isinstance(data.original, pd.DataFrame):
        df = data.original.copy()
        # Drop the surrogate ID column added by ucimlrepo, if present
        if "ID" in df.columns and df["ID"].is_unique:
            df = df.drop(columns=["ID"])
        return df

    # Fallback: concatenate features + targets DataFrames
    if data is not None and hasattr(data, "features") and hasattr(data, "targets"):
        features: pd.DataFrame = data.features
        targets: pd.DataFrame = data.targets
        if isinstance(features, pd.DataFrame) and isinstance(targets, pd.DataFrame):
            return pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)

    # Last resort: try treating dataset itself as a DataFrame
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()

    raise TypeError(
        f"Unrecognised dataset format returned by ucimlrepo: {type(dataset)}. "
        "Update ucimlrepo or use a local CSV file."
    )


def _find_target_column(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the diabetes target column, or None if not found."""
    candidates = ["Diabetes_binary", "diabetes_binary", "Diabetes", "diabetes", "target"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_sample_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load a small sample from the local CSV fallback.

    Used when ucimlrepo is unavailable or the network request fails.
    """
    from pathlib import Path

    sample_path = Path(__file__).parent.parent / "sample_cdc_diabetes.csv"
    if sample_path.exists():
        df = pd.read_csv(sample_path)
    else:
        import numpy as np

        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            "Age": rng.integers(1, 14, n),
            "BMI": rng.normal(27, 5, n).clip(10, 60),
            "HighBP": rng.integers(0, 2, n),
            "HighChol": rng.integers(0, 2, n),
            "PhysActivity": rng.integers(0, 2, n),
            "GenHlth": rng.integers(1, 6, n),
            "MentHlth": rng.integers(0, 31, n),
            "PhysHlth": rng.integers(0, 31, n),
            "Income": rng.integers(1, 9, n),
            "Diabetes_binary": rng.binomial(1, 0.15, n),
        })

    target_column = _find_target_column(df)
    if target_column is None:
        raise ValueError("Cannot identify target column in fallback sample data.")

    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    return X, y, df
