from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.runtime import Runtime
from falcon.tabular.candidates import EstimatorSpec


@pytest.mark.parametrize(
    "task",
    ["tabular_classification", "tabular_regression"],
)
def test_predictor_inference_round_trip(task: str, tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3))
    if task == "tabular_classification":
        y = np.where(X[:, 0] + X[:, 1] > 0, "positive", "negative")
        parameters: dict[str, object] = {"max_iter": 200}
    else:
        y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
        parameters = {"alpha": 1.0}
    config = RunConfig(
        candidate_sources=(
            PortfolioSource(specs=(EstimatorSpec("linear", "linear", parameters),)),
        ),
        ensemble_enabled=False,
        eval_strategy=None,
    )
    predictor = Predictor(task, config=config).fit((X, y))
    artifact_path = tmp_path / f"{task}.fnnx"
    predictor.save(artifact_path)
    runtime = Runtime(str(artifact_path))

    if task == "tabular_classification":
        np.testing.assert_array_equal(runtime.predict(X), predictor.predict(X))
        np.testing.assert_allclose(
            runtime.predict_proba(X),
            predictor.predict_proba(X),
            rtol=1e-5,
            atol=1e-6,
        )
    else:
        np.testing.assert_allclose(
            runtime.predict(X),
            predictor.predict(X),
            rtol=1e-5,
            atol=1e-5,
        )
