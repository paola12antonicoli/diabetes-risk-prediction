"""Data loading utilities for the healthcare AI project."""

from typing import Optional, Tuple

import pandas as pd


def load_cdc_diabetes_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Fetch the CDC Diabetes Health Indicators dataset and return X, y, and full dataframe.

    This function uses the ucimlrepo package to retrieve the CDC Diabetes dataset.
    If the UCI repo is unavailable, it will use a local sample file if present.
    """
    from pathlib import Path
    
    # Try UCI repo first
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return _load_sample_data()

    try:
        dataset = fetch_ucirepo('CDC Diabetes Health Indicators')
    except Exception:
        # Fallback to sample data if UCI fetch fails
        return _load_sample_data()

    df = _assemble_dataframe(dataset)
    target_column = _find_target_column(df)

    if target_column is None:
        raise ValueError(
            "Unable to identify the diabetes target column in the dataset. "
            "Expected a column such as 'Diabetes_binary'."
        )

    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    return X, y, df


def _assemble_dataframe(dataset) -> pd.DataFrame:
    """Convert the dataset object returned by ucimlrepo to a pandas DataFrame."""
    if hasattr(dataset, "frame") and dataset.frame is not None:
        return dataset.frame.copy()

    if hasattr(dataset, "data") and hasattr(dataset, "target"):
        data = dataset.data
        target = dataset.target

        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(data)

        df[target_column_name(df, target)] = pd.Series(target)
        return df

    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()

    if isinstance(dataset, dict):
        data = dataset.get("data")
        if data is None:
            raise ValueError("Dataset dict does not contain 'data'.")
        df = pd.DataFrame(data)
        target = dataset.get("target")
        if target is not None:
            df[target_column_name(df, target)] = pd.Series(target)
        return df

    raise TypeError(
        "Unexpected dataset format returned by ucimlrepo.fetch_dataset."
    )


def target_column_name(df: pd.DataFrame, target) -> str:
    """Choose a default target column name when assembling the dataset."""
    if isinstance(target, pd.Series) and target.name:
        return target.name

    known_names = ["Diabetes_binary", "diabetes_binary", "diabetes"]
    for name in known_names:
        if name in df.columns:
            return name

    return "Diabetes_binary"


def _find_target_column(df: pd.DataFrame) -> Optional[str]:
    """Identify the target column within the assembled DataFrame."""
    candidates = [
        "Diabetes_binary",
        "diabetes_binary",
        "Diabetes",
        "diabetes",
        "target",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _load_sample_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load sample diabetes data from a CSV file when UCI repo is unavailable.
    
    This is used as a fallback when the UCI ML Repository cannot be reached.
    In production, use load_cdc_diabetes_data() with a working internet connection.
    """
    from pathlib import Path
    
    # Try to load from local sample file
    sample_path = Path(__file__).parent.parent / "sample_cdc_diabetes.csv"
    if sample_path.exists():
        df = pd.read_csv(sample_path)
    else:
        # Create a minimal sample in memory
        import numpy as np
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'Age': np.random.randint(18, 80, n),
            'BMI': np.random.normal(27, 5, n).clip(10, 60),
            'HighBP': np.random.binomial(1, 0.4, n),
            'HighChol': np.random.binomial(1, 0.5, n),
            'PhysActivity': np.random.binomial(1, 0.7, n),
            'GenHlth': np.random.randint(1, 6, n),
            'MentHlth': np.random.randint(0, 31, n),
            'PhysHlth': np.random.randint(0, 31, n),
            'Income': np.random.randint(1, 9, n),
            'Diabetes_binary': np.random.binomial(1, 0.15, n),
        })
    
    target_column = _find_target_column(df)
    if target_column is None:
        raise ValueError("Cannot find target column in sample data")
    
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    return X, y, df
