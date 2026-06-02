"""Model training utilities for the healthcare AI project."""

from sklearn.base import ClassifierMixin


def train_model(X, y, model: ClassifierMixin):
    """Train a machine learning model and return the fitted estimator."""
    model.fit(X, y)
    return model
