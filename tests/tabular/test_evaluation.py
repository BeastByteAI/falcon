import numpy as np
import pytest

from falcon.tabular.evaluation import classification_metrics, regression_metrics


def test_classification_metrics_are_structured_and_silent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels = np.asarray(["a", "a", "b", "b", "c", "c"])
    predictions = np.asarray(["a", "b", "b", "b", "c", "a"])

    result = classification_metrics(labels, predictions)

    assert result["N_SAMPLES"] == 6
    assert result["N_CLASSES"] == 3
    assert 0.0 <= result["BACC"] <= 1.0
    assert 0.0 <= result["SC_SCORE"] <= 1.0
    assert capsys.readouterr().out == ""


def test_regression_metrics_are_structured_and_silent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    targets = np.asarray([1.0, 2.0, 3.0, 4.0])
    predictions = np.asarray([1.0, 2.5, 2.5, 4.0])

    result = regression_metrics(targets, predictions)

    assert result["N_SAMPLES"] == 4
    assert result["MSE"] >= 0.0
    assert result["RMSE"] >= 0.0
    assert result["MAE"] >= 0.0
    assert capsys.readouterr().out == ""
