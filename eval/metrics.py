"""
Evaluation metrics: accuracy and classification report.
"""

from sklearn.metrics import accuracy_score, classification_report


def compute_accuracy_metrics(y_true, y_pred):
    """Return accuracy and full classification report string."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return acc, report
