"""Model evaluation utilities for the healthcare AI project."""

from sklearn.metrics import classification_report, roc_auc_score


def evaluate_model(model, X, y):
    """Evaluate a fitted model and return a basic metrics summary."""
    predictions = model.predict(X)
    scores = {
        "roc_auc": roc_auc_score(y, model.predict_proba(X)[:, 1]) if hasattr(model, "predict_proba") else None,
        "classification_report": classification_report(y, predictions, output_dict=True),
    }
    return scores
