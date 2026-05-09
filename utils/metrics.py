import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification and ordinal metrics.

    Metrics:
    - accuracy
    - macro F1
    - weighted F1
    - MAE
    - quadratic weighted kappa
    - adjacent error rate
    - distant error rate
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)

    qwk = cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic"
    )

    error_distance = np.abs(y_true - y_pred)

    total_errors = np.sum(error_distance > 0)
    adjacent_errors = np.sum(error_distance == 1)
    distant_errors = np.sum(error_distance >= 2)

    total_samples = len(y_true)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mae": mae,
        "qwk": qwk,
        "total_errors": int(total_errors),
        "adjacent_errors": int(adjacent_errors),
        "distant_errors": int(distant_errors),
        "total_error_rate": total_errors / total_samples,
        "adjacent_error_rate": adjacent_errors / total_samples,
        "distant_error_rate": distant_errors / total_samples,
    }

    return metrics


def print_metrics(metrics, title="Metrics"):
    print("=" * 60)
    print(title)
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def get_classification_report(y_true, y_pred, class_names):
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


