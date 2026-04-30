import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    mean_absolute_error,
    cohen_kappa_score,
    confusion_matrix,
)


def calculate_classification_metrics(y_true, y_pred, class_names):
    """
    Calculate classification and ordinal-aware metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0
    )

    mae = mean_absolute_error(y_true, y_pred)

    qwk = cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    error_distances = np.abs(np.array(y_true) - np.array(y_pred))

    adjacent_errors = np.sum(error_distances == 1)
    distant_errors = np.sum(error_distances >= 2)
    total_errors = np.sum(error_distances > 0)

    if total_errors > 0:
        adjacent_error_rate = adjacent_errors / total_errors
        distant_error_rate = distant_errors / total_errors
    else:
        adjacent_error_rate = 0.0
        distant_error_rate = 0.0

    overall_metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mae": mae,
        "qwk": qwk,
        "total_errors": int(total_errors),
        "adjacent_errors": int(adjacent_errors),
        "distant_errors": int(distant_errors),
        "adjacent_error_rate_among_errors": adjacent_error_rate,
        "distant_error_rate_among_errors": distant_error_rate,
    }

    per_class_metrics = []

    for i, class_name in enumerate(class_names):
        per_class_metrics.append({
            "label": i,
            "class_name": class_name,
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1[i],
            "support": int(support[i]),
        })

    return overall_metrics, per_class_metrics, cm