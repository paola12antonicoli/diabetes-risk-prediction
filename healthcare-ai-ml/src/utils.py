"""Utility functions for the healthcare AI project."""

import joblib


def save_model(model, path: str):
    """Save a trained model to disk."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load a trained model from disk."""
    return joblib.load(path)
