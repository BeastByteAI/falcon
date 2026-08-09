from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.runtime import Runtime
from falcon.tabular.candidates import EstimatorSpec
from falcon.tabular.models.sklearn_model import SklearnModel
from falcon.tabular.training import CandidateLearner
from falcon.types import ColumnTypes, DatasetSchema
from tests.fnnx_conformance import assert_fnnx_conforms, extract_fnnx_graph


def _imbalanced_frame(
    n_classes: int,
    n_samples: int = 1_400,
) -> tuple[pd.DataFrame, list[str]]:
    weights = [0.9] if n_classes == 2 else [0.65, 0.25]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=weights,
        class_sep=1.1,
        flip_y=0.05,
        random_state=13,
    )
    features = [f"feature_{index}" for index in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=features)
    frame["target"] = np.asarray([f"class_{label}" for label in y])
    return frame, features


def _decision_config(**overrides: Any) -> RunConfig:
    settings: dict[str, Any] = {
        "candidate_sources": (
            PortfolioSource(
                specs=(
                    EstimatorSpec(
                        "hist",
                        "hist_gradient_boosting",
                        {"max_iter": 40, "min_samples_leaf": 20},
                    ),
                )
            ),
        ),
        "ensemble_enabled": False,
        "plateau_enabled": False,
        "oof_folds": 3,
        "eval_strategy": None,
        "random_state": 5,
    }
    settings.update(overrides)
    return RunConfig(**settings)


def _schema(task: str, n_rows: int, n_features: int) -> DatasetSchema:
    return DatasetSchema(
        column_names=tuple(f"feature_{index}" for index in range(n_features)),
        column_types=(ColumnTypes.NUMERIC_REGULAR,) * n_features,
        target_name="target",
        target_kind=(
            "classification" if task == TABULAR_CLASSIFICATION_TASK else "regression"
        ),
        dimensions=(n_rows, n_features),
    )


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("calibrate", [False, True])
def test_decision_rule_round_trips_through_the_runtime(
    tmp_path: Path,
    n_classes: int,
    calibrate: bool,
) -> None:
    frame, features = _imbalanced_frame(n_classes)
    predictor = Predictor(
        TABULAR_CLASSIFICATION_TASK,
        config=_decision_config(calibrate=calibrate),
    ).fit(frame, features=features, target="target")

    assert predictor._learner is not None
    weights = predictor._learner.decision_weights_
    assert weights is not None
    assert len(weights) == n_classes
    assert weights != (1.0,) * n_classes

    inputs = frame[features]
    predictions = predictor.predict(inputs)
    probabilities = predictor.predict_proba(inputs)

    artifact_path = tmp_path / "decision.fnnx"
    bundle = predictor.save(artifact_path)
    graph = extract_fnnx_graph(bundle)
    assert_fnnx_conforms(graph)
    decision_ops = {
        node.op_type
        for node in graph.model.graph.node
        if "falcon_decision" in node.name
    }
    assert decision_ops == {"ArgMax", "Mul"}

    runtime = Runtime(str(artifact_path))
    np.testing.assert_array_equal(runtime.predict(inputs), predictions)
    np.testing.assert_allclose(
        runtime.predict_proba(inputs),
        probabilities,
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_rule_moves_labels_without_moving_probabilities() -> None:
    frame, features = _imbalanced_frame(2)
    inputs = frame[features]
    tuned = Predictor(
        TABULAR_CLASSIFICATION_TASK,
        config=_decision_config(),
    ).fit(frame, features=features, target="target")
    plain = Predictor(
        TABULAR_CLASSIFICATION_TASK,
        config=_decision_config(decision_metric=None),
    ).fit(frame, features=features, target="target")

    assert plain._learner is not None
    assert plain._learner.decision_weights_ is None
    np.testing.assert_allclose(
        tuned.predict_proba(inputs),
        plain.predict_proba(inputs),
        rtol=1e-5,
        atol=1e-6,
    )
    assert not np.array_equal(tuned.predict(inputs), plain.predict(inputs))

    assert plain.classes_ is not None
    plain_labels = np.asarray(plain.classes_)[
        np.argmax(plain.predict_proba(inputs), axis=1)
    ]
    np.testing.assert_array_equal(plain.predict(inputs), plain_labels)


def test_reported_cross_validation_score_uses_the_deployed_rule() -> None:
    frame, features = _imbalanced_frame(2)
    predictor = Predictor(
        TABULAR_CLASSIFICATION_TASK,
        config=_decision_config(eval_strategy="cv"),
    ).fit(frame, features=features, target="target")

    assert predictor._learner is not None
    learner = predictor._learner
    assert learner.decision_weights_ is not None
    oof_result = learner.oof_predictions()
    assert oof_result is not None
    _, predictions = oof_result
    weighted_result = learner._weighted_oof_predictions()
    assert weighted_result is not None
    _, probabilities = weighted_result

    np.testing.assert_array_equal(
        predictions,
        np.argmax(
            probabilities * np.asarray(learner.decision_weights_, dtype=np.float32),
            axis=1,
        ),
    )
    assert not np.array_equal(predictions, np.argmax(probabilities, axis=1))


def test_classification_single_candidate_runs_an_oof_round_for_the_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, features = _imbalanced_frame(2, n_samples=400)
    X = frame[features].to_numpy(dtype=np.float64)
    y = (frame["target"].to_numpy() == "class_1").astype(np.int64)
    fit_calls: list[int] = []
    original_fit = SklearnModel.fit

    def counting_fit(self: SklearnModel, *args: Any, **kwargs: Any) -> None:
        fit_calls.append(1)
        original_fit(self, *args, **kwargs)

    monkeypatch.setattr(SklearnModel, "fit", counting_fit)
    config = _decision_config()
    learner = CandidateLearner(TABULAR_CLASSIFICATION_TASK, X.shape, config)

    learner.fit(X, y, _schema(TABULAR_CLASSIFICATION_TASK, *X.shape))

    assert len(fit_calls) == config.oof_folds + 1
    assert learner._evaluation_run is not None


def test_a_rare_class_below_the_floor_exports_without_a_decision_rule(
    tmp_path: Path,
) -> None:
    frame, features = _imbalanced_frame(2, n_samples=400)
    inputs = frame[features]
    tuned = Predictor(
        TABULAR_CLASSIFICATION_TASK,
        config=_decision_config(),
    ).fit(frame, features=features, target="target")
    plain = Predictor(
        TABULAR_CLASSIFICATION_TASK,
        config=_decision_config(decision_metric=None),
    ).fit(frame, features=features, target="target")

    assert tuned._learner is not None
    assert tuned._learner.decision_weights_ is None

    artifact_path = tmp_path / "floor.fnnx"
    bundle = tuned.save(artifact_path)
    graph = extract_fnnx_graph(bundle)
    assert not any("falcon_decision" in node.name for node in graph.model.graph.node)

    np.testing.assert_array_equal(tuned.predict(inputs), plain.predict(inputs))
    np.testing.assert_array_equal(
        Runtime(str(artifact_path)).predict(inputs),
        tuned.predict(inputs),
    )


def test_regression_rejects_an_explicit_decision_metric() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 2))
    y = 2.0 * X[:, 0] - X[:, 1]
    config = _decision_config(decision_metric="f1")
    learner = CandidateLearner(TABULAR_REGRESSION_TASK, X.shape, config)

    with pytest.raises(ValueError, match="only available for classification"):
        learner.fit(X, y, _schema(TABULAR_REGRESSION_TASK, *X.shape))


def test_regression_ignores_the_untouched_decision_metric_default() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 2))
    y = 2.0 * X[:, 0] - X[:, 1]
    config = RunConfig(
        candidate_sources=(
            PortfolioSource(specs=(EstimatorSpec("ridge", "linear", {"alpha": 1.0}),)),
        ),
        ensemble_enabled=False,
        plateau_enabled=False,
        oof_folds=3,
        eval_strategy=None,
    )
    learner = CandidateLearner(TABULAR_REGRESSION_TASK, X.shape, config)

    learner.fit(X, y, _schema(TABULAR_REGRESSION_TASK, *X.shape))

    assert learner.decision_weights_ is None
