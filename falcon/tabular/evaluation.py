from typing import Any

import numpy as np
from numpy import typing as npt
from sklearn import metrics


def _scale_accuracy(accuracy: float, n_classes: int) -> float:
    if accuracy < 0.0 or accuracy > 1.0:
        raise ValueError("Accuracy score should be in range [0,1]")
    if accuracy in {0.0, 1.0} or n_classes < 3:
        return accuracy

    random_performance = 1 / n_classes
    if accuracy <= random_performance:
        return accuracy * 0.5 / random_performance
    return accuracy * 0.5 / (1 - random_performance) + (
        0.5 - 0.5 * random_performance / (1 - random_performance)
    )


def classification_metrics(
    y: npt.NDArray[Any],
    predictions: npt.NDArray[Any],
) -> dict[str, int | float]:
    labels = np.asarray(y).astype(np.str_)
    predicted_labels = np.asarray(predictions).astype(np.str_)
    report = metrics.classification_report(
        labels,
        predicted_labels,
        output_dict=True,
        zero_division=0,
    )
    macro_average = report["macro avg"]
    weighted_average = report["weighted avg"]
    n_classes = int(np.unique(labels).size)
    balanced_accuracy = float(metrics.balanced_accuracy_score(labels, predicted_labels))
    result: dict[str, int | float] = {
        "N_SAMPLES": len(labels),
        "N_CLASSES": n_classes,
        "ACC": float(metrics.accuracy_score(labels, predicted_labels)),
        "BACC": balanced_accuracy,
        "PRECISION": float(macro_average["precision"]),
        "RECALL": float(macro_average["recall"]),
        "F1": float(macro_average["f1-score"]),
        "B_PRECISION": float(weighted_average["precision"]),
        "B_RECALL": float(weighted_average["recall"]),
        "B_F1": float(weighted_average["f1-score"]),
        "SCORE": balanced_accuracy,
    }
    result["SC_SCORE"] = _scale_accuracy(balanced_accuracy, n_classes)
    return result


def regression_metrics(
    y: npt.NDArray[Any],
    predictions: npt.NDArray[Any],
) -> dict[str, int | float]:
    targets = np.asarray(y, dtype=np.float64).reshape(-1)
    predicted_targets = np.asarray(predictions, dtype=np.float64).reshape(-1)
    differences = targets - predicted_targets
    r2 = float(metrics.r2_score(targets, predicted_targets))
    rmse = float(np.sqrt(np.mean(np.square(differences))))
    score = max(r2, 0.0)
    return {
        "N_SAMPLES": len(targets),
        "R2": r2,
        "RMSE": rmse,
        "MSE": float(np.mean(np.square(differences))),
        "MAE": float(np.mean(np.abs(differences))),
        "RMSLE": float(np.log(rmse + 1e-7)),
        "SCORE": score,
        "SC_SCORE": (score + 1) / 2,
    }


__all__ = ["classification_metrics", "regression_metrics"]
