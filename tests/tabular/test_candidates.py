from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from numpy import typing as npt
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import balanced_accuracy_score

from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.serialization import SerializedModelRepr, serialize_to_onnx
from falcon.tabular.candidates import (
    CandidateModel,
    CandidateTrainer,
    EstimatorSpec,
    default_portfolio,
)
from falcon.tabular.models.gbdt import get_gbdt_model_classes
from falcon.tabular.models.sklearn_model import SklearnModel
from falcon.tabular.splitting import holdout_indices


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _RecordingModel:
    def __init__(self, clock: _Clock, duration: float) -> None:
        self.clock = clock
        self.duration = duration
        self.X: npt.NDArray[Any] | None = None
        self.y: npt.NDArray[Any] | None = None
        self.sample_weight: npt.NDArray[np.float64] | None = None
        self.validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None
        self.early_stopping_rounds: int | None = None

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self.X = X.copy()
        self.y = y.copy()
        self.sample_weight = None if sample_weight is None else sample_weight.copy()
        self.validation_data = validation_data
        self.early_stopping_rounds = early_stopping_rounds
        self.clock.advance(self.duration)

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
        return np.zeros(len(X), dtype=np.int64)

    def serialize(self) -> SerializedModelRepr:
        raise NotImplementedError


class _RecordingFactory:
    def __init__(self, clock: _Clock, durations: Mapping[str, float]) -> None:
        self.clock = clock
        self.durations = durations
        self.models: list[_RecordingModel] = []
        self.random_states: list[int] = []

    def __call__(
        self,
        spec: EstimatorSpec,
        task: str,
        random_state: int,
        n_classes: int | None,
    ) -> CandidateModel:
        model = _RecordingModel(self.clock, self.durations[spec.name])
        self.models.append(model)
        self.random_states.append(random_state)
        return model


def _spec(
    name: str,
    *,
    early_stopping_rounds: int | None = None,
) -> EstimatorSpec:
    return EstimatorSpec(
        name=name,
        family="hist_gradient_boosting",
        parameters={},
        early_stopping_rounds=early_stopping_rounds,
    )


def test_default_portfolio_is_gbdt_first_and_interleaved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from falcon.tabular import candidates

    monkeypatch.setattr(
        candidates,
        "get_gbdt_model_classes",
        lambda task, n_classes=None: {
            "lightgbm": object,
            "xgboost": object,
            "catboost": object,
        },
    )

    portfolio = default_portfolio(TABULAR_CLASSIFICATION_TASK, n_classes=2)
    families = [spec.family for spec in portfolio]

    assert families[:6] == [
        "lightgbm",
        "hist_gradient_boosting",
        "xgboost",
        "extra_trees",
        "catboost",
        "linear",
    ]
    assert len({spec.name for spec in portfolio}) == len(portfolio)
    assert any(spec.family == "random_forest" for spec in portfolio)
    assert any(
        spec.family == "lightgbm" and spec.parameters.get("num_leaves") == 128
        for spec in portfolio
    )


def test_default_portfolio_applies_catboost_multiclass_parity_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    if "catboost" not in get_gbdt_model_classes(
        TABULAR_CLASSIFICATION_TASK,
        n_classes=2,
    ):
        pytest.skip("catboost is not installed")

    binary_portfolio = default_portfolio(
        TABULAR_CLASSIFICATION_TASK,
        n_classes=2,
    )
    regression_portfolio = default_portfolio(TABULAR_REGRESSION_TASK)
    with caplog.at_level("INFO", logger="falcon"):
        multiclass_portfolio = default_portfolio(
            TABULAR_CLASSIFICATION_TASK,
            n_classes=3,
        )

    assert any(spec.family == "catboost" for spec in binary_portfolio)
    assert any(spec.family == "catboost" for spec in regression_portfolio)
    assert all(spec.family != "catboost" for spec in multiclass_portfolio)
    assert "CatBoost is excluded from multiclass classification" in caplog.text


def test_default_portfolio_degrades_to_sklearn_with_one_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from falcon.tabular import candidates

    monkeypatch.setattr(
        candidates,
        "get_gbdt_model_classes",
        lambda task, n_classes=None: {},
    )

    portfolio = default_portfolio(TABULAR_REGRESSION_TASK)
    caplog.clear()
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    y = np.zeros(40, dtype=np.float32)

    with caplog.at_level("INFO", logger="falcon"):
        run = CandidateTrainer(
            TABULAR_REGRESSION_TASK,
            time_limit=np.finfo(float).eps,
        ).fit(X, y)

    assert {spec.family for spec in portfolio} == {
        "hist_gradient_boosting",
        "extra_trees",
        "linear",
        "random_forest",
    }
    assert len(run.candidates) == 1
    assert run.candidates[0].spec.family == "hist_gradient_boosting"
    assert run.stopped_for_budget
    assert run.candidates[0].model.predict(X).shape == (len(X),)
    graph = serialize_to_onnx(
        [run.candidates[0].model.serialize()],
        task=TABULAR_REGRESSION_TASK,
    )
    onnx.checker.check_model(graph)
    assert run.elapsed_time <= (
        run.candidates[0].fit_time + float(np.finfo(float).eps) + 1.0
    )
    assert caplog.text.count("sklearn-only candidate portfolio") == 1
    assert "time limit is insufficient for the first candidate" in caplog.text


def test_estimator_spec_validates_guards_and_centralized_seed() -> None:
    spec = EstimatorSpec(
        name="bounded",
        family="linear",
        parameters={"alpha": 1.0},
        min_rows=10,
        max_rows=100,
        max_features=5,
    )

    assert spec.is_applicable(n_rows=10, n_features=5)
    assert not spec.is_applicable(n_rows=9, n_features=5)
    assert not spec.is_applicable(n_rows=10, n_features=6)

    with pytest.raises(ValueError, match="min_rows"):
        EstimatorSpec("invalid", "linear", min_rows=10, max_rows=5)
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        EstimatorSpec("invalid", "linear", early_stopping_rounds=0)


def test_budget_reserve_stops_before_next_candidate_and_guarantees_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _Clock()
    factory = _RecordingFactory(clock, {"first": 9.0, "second": 1.0})
    trainer = CandidateTrainer(
        TABULAR_REGRESSION_TASK,
        time_limit=10.0,
        reserve_fraction=0.2,
        model_factory=factory,
        clock=clock,
    )
    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.arange(12, dtype=np.float32)

    with caplog.at_level("INFO", logger="falcon"):
        run = trainer.fit(X, y, specs=[_spec("first"), _spec("second")])

    assert [candidate.spec.name for candidate in run.candidates] == ["first"]
    assert run.stopped_for_budget
    assert run.elapsed_time == 9.0
    assert "time limit is insufficient for the first candidate" in caplog.text
    assert "Candidate 1/2" in caplog.text
    assert "estimated remaining time" in caplog.text


def test_classification_leaves_sample_weights_off_by_default() -> None:
    clock = _Clock()
    factory = _RecordingFactory(clock, {"weighted": 0.0})
    trainer = CandidateTrainer(
        TABULAR_CLASSIFICATION_TASK,
        model_factory=factory,
        clock=clock,
    )
    X = np.arange(48, dtype=np.float32).reshape(24, 2)
    y = np.asarray([0] * 20 + [1] * 4)

    trainer.fit(X, y, specs=[_spec("weighted")])

    model = factory.models[0]
    assert model.X is not None
    assert len(model.X) == len(X)
    assert model.sample_weight is None


def test_balanced_class_weight_opts_into_sample_weights_without_resampling() -> None:
    clock = _Clock()
    factory = _RecordingFactory(clock, {"weighted": 0.0})
    trainer = CandidateTrainer(
        TABULAR_CLASSIFICATION_TASK,
        class_weight="balanced",
        model_factory=factory,
        clock=clock,
    )
    X = np.arange(48, dtype=np.float32).reshape(24, 2)
    y = np.asarray([0] * 20 + [1] * 4)

    trainer.fit(X, y, specs=[_spec("weighted")])

    # The first model built is the throwaway used by the up-front capability check.
    model = factory.models[-1]
    assert model.X is not None
    assert model.sample_weight is not None
    assert len(model.X) == len(X)
    assert len(model.sample_weight) == len(X)
    assert model.sample_weight[y == 1].mean() > model.sample_weight[y == 0].mean()


class _WeightBlindEstimator:
    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> None:
        del X, y


class _WeightBlindModel:
    def __init__(self) -> None:
        self.estimator = _WeightBlindEstimator()

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self.estimator.fit(X, y)

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
        return np.zeros(len(X), dtype=np.int64)

    def serialize(self) -> SerializedModelRepr:
        raise NotImplementedError


def test_balanced_class_weight_rejects_families_that_cannot_take_weights() -> None:
    def factory(
        spec: EstimatorSpec,
        task: str,
        random_state: int,
        n_classes: int | None,
    ) -> CandidateModel:
        return _WeightBlindModel()

    trainer = CandidateTrainer(
        TABULAR_CLASSIFICATION_TASK,
        class_weight="balanced",
        model_factory=factory,
    )
    X = np.arange(48, dtype=np.float32).reshape(24, 2)
    y = np.asarray([0] * 12 + [1] * 12)
    spec = EstimatorSpec("blind", "weight_blind", {})

    with pytest.raises(ValueError, match="weight_blind"):
        trainer.fit(X, y, specs=[spec])


def test_a_model_factory_without_an_estimator_is_allowed_through() -> None:
    clock = _Clock()
    factory = _RecordingFactory(clock, {"opaque": 0.0})
    trainer = CandidateTrainer(
        TABULAR_CLASSIFICATION_TASK,
        class_weight="balanced",
        model_factory=factory,
        clock=clock,
    )
    X = np.arange(48, dtype=np.float32).reshape(24, 2)
    y = np.asarray([0] * 12 + [1] * 12)

    run = trainer.fit(X, y, specs=[_spec("opaque")])

    assert not hasattr(factory.models[-1], "estimator")
    assert len(run.candidates) == 1


def test_sklearn_models_omit_the_sample_weight_keyword_when_it_is_unset() -> None:
    unset = object()
    received: list[object] = []

    class _SpyEstimator:
        def fit(
            self,
            X: npt.NDArray[Any],
            y: npt.NDArray[Any],
            sample_weight: Any = unset,
        ) -> None:
            received.append(sample_weight)

    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10) % 2
    weights = np.ones(10, dtype=np.float64)

    SklearnModel(_SpyEstimator(), TABULAR_CLASSIFICATION_TASK).fit(X, y)
    SklearnModel(_SpyEstimator(), TABULAR_CLASSIFICATION_TASK).fit(
        X,
        y,
        sample_weight=weights,
    )

    assert received[0] is unset
    assert np.array_equal(np.asarray(received[1]), weights)


def test_early_stopping_validation_split_is_group_aware_and_reproducible() -> None:
    X = np.column_stack(
        (
            np.arange(24, dtype=np.float32),
            np.tile(np.asarray([0.0, 1.0], dtype=np.float32), 12),
        )
    )
    groups = np.repeat(np.arange(12), 2)
    y = np.repeat(np.arange(12) % 2, 2)
    spec = _spec("early", early_stopping_rounds=5)
    trained_indices: list[npt.NDArray[np.int64]] = []

    for _ in range(2):
        clock = _Clock()
        factory = _RecordingFactory(clock, {"early": 0.0})
        trainer = CandidateTrainer(
            TABULAR_CLASSIFICATION_TASK,
            random_state=17,
            model_factory=factory,
            clock=clock,
        )
        trainer.fit(X, y, groups=groups, specs=[spec])
        model = factory.models[0]
        assert factory.random_states == [17]
        assert model.X is not None
        assert model.validation_data is not None
        validation_X, _ = model.validation_data
        train_indices = model.X[:, 0].astype(np.int64)
        validation_indices = validation_X[:, 0].astype(np.int64)
        assert set(groups[train_indices]).isdisjoint(groups[validation_indices])
        assert model.early_stopping_rounds == 5
        trained_indices.append(train_indices)

    assert np.array_equal(trained_indices[0], trained_indices[1])


def test_weighted_sklearn_candidate_is_reproducible_and_serializable() -> None:
    X, y = make_classification(
        n_samples=320,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        weights=[0.95, 0.05],
        class_sep=3.0,
        flip_y=0,
        random_state=3,
    )
    X = X.astype(np.float32)
    train_indices, eval_indices = holdout_indices(
        X,
        y,
        TABULAR_CLASSIFICATION_TASK,
        groups=np.arange(len(X)),
        random_state=23,
    )
    spec = EstimatorSpec(
        name="hist",
        family="hist_gradient_boosting",
        parameters={"max_iter": 40, "min_samples_leaf": 5},
    )
    predictions: list[npt.NDArray[Any]] = []
    models: list[CandidateModel] = []

    for _ in range(2):
        run = CandidateTrainer(
            TABULAR_CLASSIFICATION_TASK,
            random_state=23,
        ).fit(X[train_indices], y[train_indices], specs=[spec])
        model = run.candidates[0].model
        predictions.append(model.predict(X[eval_indices]))
        models.append(model)

    assert np.array_equal(predictions[0], predictions[1])
    assert balanced_accuracy_score(y[eval_indices], predictions[0]) > 0.8

    graph = serialize_to_onnx(
        [models[0].serialize()],
        task=TABULAR_CLASSIFICATION_TASK,
    )
    onnx.checker.check_model(graph)


def test_hist_gradient_boosting_uses_external_early_stopping() -> None:
    X, y = make_regression(
        n_samples=80,
        n_features=5,
        n_informative=4,
        noise=8.0,
        random_state=9,
    )
    X = X.astype(np.float32)
    spec = EstimatorSpec(
        name="hist_early",
        family="hist_gradient_boosting",
        parameters={"max_iter": 25, "min_samples_leaf": 5},
        early_stopping_rounds=3,
    )

    run = CandidateTrainer(
        TABULAR_REGRESSION_TASK,
        random_state=11,
    ).fit(X, y, specs=[spec], groups=np.arange(len(X)))
    estimator = cast(Any, run.candidates[0].model).estimator

    assert estimator.early_stopping is False
    assert estimator.warm_start is False
    assert 1 <= estimator.max_iter <= 25
    assert run.candidates[0].model.predict(X).shape == (len(X),)


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost"])
def test_optional_gbdt_candidate_accepts_external_early_stopping_split(
    family: str,
) -> None:
    if family not in get_gbdt_model_classes(TABULAR_CLASSIFICATION_TASK, n_classes=2):
        pytest.skip(f"{family} is not installed")
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=12,
    )
    X = X.astype(np.float32)
    parameter_name = "iterations" if family == "catboost" else "n_estimators"
    spec = EstimatorSpec(
        name=family,
        family=family,
        parameters={parameter_name: 20},
        early_stopping_rounds=3,
    )

    run = CandidateTrainer(
        TABULAR_CLASSIFICATION_TASK,
        random_state=11,
    ).fit(X, y, specs=[spec], groups=np.arange(len(X)))
    model = run.candidates[0].model
    graph = serialize_to_onnx(
        [model.serialize()],
        task=TABULAR_CLASSIFICATION_TASK,
    )
    session = ort.InferenceSession(
        graph.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    labels, probabilities = session.run(
        None,
        {session.get_inputs()[0].name: X},
    )

    assert np.array_equal(np.asarray(labels).reshape(-1), model.predict(X))
    np.testing.assert_allclose(
        probabilities,
        cast(Any, model).predict_proba(X),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time_limit": 0.0}, "time_limit"),
        ({"reserve_fraction": 1.0}, "reserve_fraction"),
    ],
)
def test_candidate_trainer_rejects_invalid_budget_settings(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CandidateTrainer(TABULAR_REGRESSION_TASK, **kwargs)


def test_candidate_trainer_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        CandidateTrainer("forecasting")
