from __future__ import annotations

import numpy as np
import pytest

from falcon.config import PortfolioSource, RunConfig
from falcon.sklapi import (
    FalconClassifier,
    FalconRegressor,
    FalconTabularClassifier,
    FalconTabularRegressor,
)
from falcon.tabular.candidates import EstimatorSpec


def _linear_config() -> RunConfig:
    return RunConfig(
        candidate_sources=(
            PortfolioSource(specs=(EstimatorSpec("linear", "linear"),)),
        ),
        ensemble_enabled=False,
        eval_strategy=None,
    )


def test_sklapi_regressor_uses_predictor_for_the_regression_task() -> None:
    X = np.arange(120, dtype=np.float64).reshape(60, 2)
    y = 2 * X[:, 0] - X[:, 1]
    estimator = FalconTabularRegressor(preset=_linear_config())

    result = estimator.fit(X, y)

    assert result is estimator
    assert estimator.predictor_.task == "tabular_regression"
    assert estimator.n_features_in_ == 2
    assert estimator.predict(X[:4]).shape == (4,)
    assert FalconRegressor is FalconTabularRegressor


def test_sklapi_classifier_exposes_native_probabilities() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3))
    y = np.where(X[:, 0] > 0, "positive", "negative")
    estimator = FalconTabularClassifier(preset=_linear_config())

    estimator.fit(X, y)
    probabilities = estimator.predict_proba(X[:5])

    assert estimator.predictor_.task == "tabular_classification"
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert FalconClassifier is FalconTabularClassifier


def test_sklapi_replaces_config_with_preset() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'config'"):
        FalconRegressor(config="PlainLearner")  # type: ignore[call-arg]
