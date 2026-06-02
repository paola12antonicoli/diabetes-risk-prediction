"""Preprocessing utilities for the healthcare AI project."""

import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature selection and cleaning to the raw dataset."""
    df = df.copy()
    # Add preprocessing steps here
    return df
