from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy import typing as npt

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.runtime import Runtime
from falcon.tabular.calibration import fit_temperature
from falcon.tabular.candidates import EstimatorSpec, GreedyWeightedEnsemble
from falcon.tabular.conformal import fit_conformal_quantile
from falcon.tabular.models.sklearn_model import SklearnModel
from falcon.tabular.training import CandidateLearner
from falcon.types import ColumnTypes, DatasetSchema
from tests.fnnx_conformance import assert_fnnx_conforms, extract_fnnx_graph


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


def _regression_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 2))
    y = 3.0 * X[:, 0] - X[:, 1]
    return X, y


def _classification_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(29)
    X = rng.normal(size=(90, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X, y


def _worst_first_regression_specs() -> tuple[EstimatorSpec, ...]:
    return (
        EstimatorSpec("bad-ridge", "linear", {"alpha": 1e9}),
        EstimatorSpec("good-ridge", "linear", {"alpha": 1e-2}),
    )


def _worst_first_classification_specs() -> tuple[EstimatorSpec, ...]:
    return (
        EstimatorSpec("stump", "random_forest", {"n_estimators": 1, "max_depth": 1}),
        EstimatorSpec("logistic", "linear", {"C": 1000.0, "max_iter": 500}),
    )


def _no_ensemble_config(
    specs: tuple[EstimatorSpec, ...],
    **overrides: Any,
) -> RunConfig:
    return RunConfig(
        candidate_sources=(PortfolioSource(specs=specs),),
        ensemble_enabled=False,
        plateau_enabled=False,
        oof_folds=3,
        eval_strategy=None,
        **overrides,
    )


def test_all_candidates_are_trained_and_the_best_one_wins() -> None:
    X, y = _regression_data()
    specs = _worst_first_regression_specs()
    predictor = Predictor(
        TABULAR_REGRESSION_TASK,
        config=_no_ensemble_config(specs),
    ).fit((X, y))

    leaderboard = predictor.leaderboard()
    assert list(leaderboard["candidate"]) == ["bad-ridge", "good-ridge"]
    assert list(leaderboard["weight"]) == [0.0, 1.0]
    assert all(fit_time > 0 for fit_time in leaderboard["fit_time"])
    scores = dict(zip(leaderboard["candidate"], leaderboard["score"], strict=True))
    assert scores["good-ridge"] > scores["bad-ridge"]

    assert predictor._learner is not None
    assert not isinstance(predictor._learner.model, GreedyWeightedEnsemble)
    np.testing.assert_allclose(predictor.predict(X), y, atol=0.5)


def test_single_candidate_path_keeps_one_fit_and_no_oof_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _regression_data()
    fit_calls: list[int] = []
    original_fit = SklearnModel.fit

    def counting_fit(self: SklearnModel, *args: Any, **kwargs: Any) -> None:
        fit_calls.append(1)
        original_fit(self, *args, **kwargs)

    monkeypatch.setattr(SklearnModel, "fit", counting_fit)
    spec = EstimatorSpec("only-ridge", "linear", {"alpha": 1.0})
    config = _no_ensemble_config((spec,))
    learner = CandidateLearner(TABULAR_REGRESSION_TASK, X.shape, config)

    learner.fit(X, y, _schema(TABULAR_REGRESSION_TASK, *X.shape))

    assert len(fit_calls) == 1
    assert learner._evaluation_run is None
    records = learner.leaderboard_records()
    assert len(records) == 1
    assert records[0]["candidate"] == "only-ridge"
    assert records[0]["weight"] == 1.0


def test_single_candidate_with_conformal_adds_only_the_required_oof_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _regression_data()
    fit_calls: list[int] = []
    original_fit = SklearnModel.fit

    def counting_fit(self: SklearnModel, *args: Any, **kwargs: Any) -> None:
        fit_calls.append(1)
        original_fit(self, *args, **kwargs)

    monkeypatch.setattr(SklearnModel, "fit", counting_fit)
    spec = EstimatorSpec("only-ridge", "linear", {"alpha": 1.0})
    config = _no_ensemble_config((spec,), conformal_alpha=0.2)
    learner = CandidateLearner(TABULAR_REGRESSION_TASK, X.shape, config)

    learner.fit(X, y, _schema(TABULAR_REGRESSION_TASK, *X.shape))

    assert len(fit_calls) == config.oof_folds + 1
    assert learner.conformal_quantile_ is not None


def test_calibration_temperature_is_fitted_on_the_winner_oof_predictions() -> None:
    X, y = _classification_data()
    specs = _worst_first_classification_specs()
    config = _no_ensemble_config(specs, calibrate=True)
    learner = CandidateLearner(TABULAR_CLASSIFICATION_TASK, X.shape, config)

    learner.fit(X, y, _schema(TABULAR_CLASSIFICATION_TASK, *X.shape))

    run = learner._evaluation_run
    assert run is not None
    assert [candidate.spec.name for candidate in run.candidates] == [
        "stump",
        "logistic",
    ]
    winner_index = int(np.argmax(run.ensemble.weights))
    winner = run.candidates[winner_index]
    assert winner.spec.name == "logistic"
    assert run.ensemble.weights == (0.0, 1.0)

    oof_result = learner._weighted_oof_predictions()
    assert oof_result is not None
    evaluation_indices, probabilities = oof_result
    np.testing.assert_array_equal(probabilities, winner.oof_predictions)
    assert learner.temperature_ == fit_temperature(
        winner.oof_predictions,
        y[evaluation_indices],
    )


def test_conformal_quantile_is_fitted_on_the_winner_oof_predictions() -> None:
    X, y = _regression_data()
    specs = _worst_first_regression_specs()
    config = _no_ensemble_config(specs, conformal_alpha=0.2)
    learner = CandidateLearner(TABULAR_REGRESSION_TASK, X.shape, config)

    learner.fit(X, y, _schema(TABULAR_REGRESSION_TASK, *X.shape))

    run = learner._evaluation_run
    assert run is not None
    winner = run.candidates[int(np.argmax(run.ensemble.weights))]
    loser = run.candidates[int(np.argmin(run.ensemble.weights))]
    assert winner.spec.name == "good-ridge"

    evaluation_targets = y[run.evaluation_indices]
    winner_quantile = fit_conformal_quantile(
        winner.oof_predictions,
        evaluation_targets,
        0.2,
    )
    loser_quantile = fit_conformal_quantile(
        loser.oof_predictions,
        evaluation_targets,
        0.2,
    )
    assert learner.conformal_quantile_ == winner_quantile
    assert winner_quantile < loser_quantile


def test_exported_bundle_contains_a_single_refit_model(tmp_path: Path) -> None:
    X, y = _regression_data()
    specs = (
        EstimatorSpec("forest-a", "random_forest", {"n_estimators": 3, "max_depth": 3}),
        EstimatorSpec("forest-b", "random_forest", {"n_estimators": 5, "max_depth": 3}),
    )
    predictor = Predictor(
        TABULAR_REGRESSION_TASK,
        config=_no_ensemble_config(specs),
    ).fit((X, y))

    assert len(predictor.leaderboard()) == 2
    artifact_path = tmp_path / "single-model.fnnx"
    bundle = predictor.save(artifact_path)
    extracted = extract_fnnx_graph(bundle)

    assert_fnnx_conforms(extracted)
    tree_nodes = [
        node
        for node in extracted.model.graph.node
        if node.op_type == "TreeEnsembleRegressor"
    ]
    assert len(tree_nodes) == 1
    np.testing.assert_allclose(
        Runtime(str(artifact_path)).predict(X.astype(np.float32)),
        predictor.predict(X),
        rtol=1e-5,
        atol=1e-5,
    )


def test_calibration_and_conformal_round_trip_with_ensembling_off(
    tmp_path: Path,
) -> None:
    X, y = _classification_data()
    predictor = Predictor(
        TABULAR_CLASSIFICATION_TASK,
        config=_no_ensemble_config(
            _worst_first_classification_specs(),
            calibrate=True,
        ),
    ).fit((X, y))
    artifact_path = tmp_path / "calibrated.fnnx"
    predictor.save(artifact_path)
    probabilities = predictor.predict_proba(X)

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        Runtime(str(artifact_path)).predict_proba(X.astype(np.float32)),
        probabilities,
        rtol=1e-5,
        atol=1e-6,
    )

    X_reg, y_reg = _regression_data()
    regression_predictor = Predictor(
        TABULAR_REGRESSION_TASK,
        config=_no_ensemble_config(
            _worst_first_regression_specs(),
            conformal_alpha=0.2,
        ),
    ).fit((X_reg, y_reg))
    regression_path = tmp_path / "conformal.fnnx"
    regression_predictor.save(regression_path)
    lower, upper = Runtime(str(regression_path)).predict_interval(
        X_reg.astype(np.float32)
    )
    predictions = regression_predictor.predict(X_reg)

    assert np.all(lower <= upper)
    np.testing.assert_allclose((lower + upper) / 2, predictions, rtol=1e-4, atol=1e-4)
