from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from numpy import typing as npt
from onnx import TensorProto, helper
from sklearn.datasets import make_classification

from falcon.config import ONNX_OPSET_VERSION
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.runtime import Runtime
from falcon.serialization import (
    FNNXSerializer,
    SerializedModelRepr,
    serialize_to_onnx,
)
from falcon.tabular.candidates import (
    CandidateModel,
    EstimatorSpec,
    OOFEnsembleTrainer,
    greedy_weighted_selection,
    score_oof_predictions,
)
from falcon.tabular.processors.label_decoder import LabelDecoder
from falcon.tabular.processors.multi_modal_encoder import MultiModalEncoder
from falcon.types import ColumnTypes, DatasetSchema
from tests.fnnx_conformance import assert_fnnx_conforms, extract_fnnx_graph


class _ConstantModel:
    def __init__(self, task: str, value: float = 0.0) -> None:
        self.task = task
        self.value = value
        self.fit_rows: npt.NDArray[np.int64] | None = None
        self.prediction_rows: list[npt.NDArray[np.int64]] = []

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self.fit_rows = X[:, 0].astype(np.int64)
        if self.task == TABULAR_REGRESSION_TASK:
            self.value = float(np.mean(y))

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        self.prediction_rows.append(X[:, 0].astype(np.int64))
        if self.task == TABULAR_CLASSIFICATION_TASK:
            return np.zeros(len(X), dtype=np.int64)
        return np.full(len(X), self.value, dtype=np.float32)

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        self.prediction_rows.append(X[:, 0].astype(np.int64))
        return np.tile(
            np.asarray([[0.6, 0.4]], dtype=np.float32),
            (len(X), 1),
        )

    def serialize(self) -> SerializedModelRepr:
        raise NotImplementedError


class _ConstantFactory:
    def __init__(self) -> None:
        self.models: dict[str, list[_ConstantModel]] = defaultdict(list)
        self.seeds: dict[str, list[int]] = defaultdict(list)

    def __call__(
        self,
        spec: EstimatorSpec,
        task: str,
        random_state: int,
        n_classes: int | None,
    ) -> CandidateModel:
        model = _ConstantModel(task)
        self.models[spec.name].append(model)
        self.seeds[spec.name].append(random_state)
        return model


class _DummyRegressionFactory:
    def __call__(
        self,
        spec: EstimatorSpec,
        task: str,
        random_state: int,
        n_classes: int | None,
    ) -> CandidateModel:
        constant = spec.parameters.get("constant")
        if not isinstance(constant, (int, float)):
            raise ValueError("Dummy regression candidates require a constant")
        return _SerializableConstantRegressor(float(constant))


class _SerializableConstantRegressor:
    def __init__(self, constant: float) -> None:
        self.constant = constant
        self.n_features: int | None = None

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self.n_features = X.shape[1]

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        return np.full(len(X), self.constant, dtype=np.float32)

    def serialize(self) -> SerializedModelRepr:
        if self.n_features is None:
            raise RuntimeError("Model must be fitted before serialization")
        input_info = helper.make_tensor_value_info(
            "model_input",
            TensorProto.FLOAT,
            [None, self.n_features],
        )
        output_info = helper.make_tensor_value_info(
            "prediction",
            TensorProto.FLOAT,
            [None, 1],
        )
        weights = helper.make_tensor(
            "weights",
            TensorProto.FLOAT,
            [self.n_features, 1],
            [0.0] * self.n_features,
        )
        bias = helper.make_tensor(
            "bias",
            TensorProto.FLOAT,
            [1],
            [self.constant],
        )
        graph = helper.make_graph(
            [
                helper.make_node(
                    "MatMul",
                    ["model_input", "weights"],
                    ["zero_prediction"],
                ),
                helper.make_node(
                    "Add",
                    ["zero_prediction", "bias"],
                    ["prediction"],
                ),
            ],
            "constant_regressor",
            [input_info],
            [output_info],
            initializer=[weights, bias],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", ONNX_OPSET_VERSION)],
        )
        return SerializedModelRepr(
            model,
            n_inputs=1,
            n_outputs=1,
            initial_types=["FLOAT32"],
            initial_shapes=[[None, self.n_features]],
        )


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _TimedSerializableConstantRegressor(_SerializableConstantRegressor):
    def __init__(self, constant: float, clock: _Clock, duration: float) -> None:
        super().__init__(constant)
        self.clock = clock
        self.duration = duration

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            validation_data=validation_data,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.clock.advance(self.duration)


class _TimedRegressionFactory:
    def __init__(
        self,
        clock: _Clock,
        warning_is_visible: Callable[[], bool],
    ) -> None:
        self.clock = clock
        self.warning_is_visible = warning_is_visible
        self.calls: list[str] = []
        self.warning_visible_before_second_fold = False

    def __call__(
        self,
        spec: EstimatorSpec,
        task: str,
        random_state: int,
        n_classes: int | None,
    ) -> CandidateModel:
        del task, random_state, n_classes
        self.calls.append(spec.name)
        if len(self.calls) == 2:
            self.warning_visible_before_second_fold = self.warning_is_visible()
        constant = spec.parameters.get("constant")
        if not isinstance(constant, (int, float)):
            raise ValueError("Timed regression candidates require a constant")
        return _TimedSerializableConstantRegressor(
            float(constant),
            self.clock,
            duration=3.0,
        )


def _spec(name: str, parameters: Mapping[str, object] | None = None) -> EstimatorSpec:
    return EstimatorSpec(name, "linear", parameters or {})


def test_oof_collection_uses_group_aware_cross_validation() -> None:
    X = np.column_stack(
        (
            np.arange(24, dtype=np.float32),
            np.tile(np.asarray([0.0, 1.0], dtype=np.float32), 12),
        )
    )
    groups = np.repeat(np.arange(12), 2)
    y = np.repeat(np.arange(12) % 2, 2)
    factory = _ConstantFactory()

    run = OOFEnsembleTrainer(
        TABULAR_CLASSIFICATION_TASK,
        n_splits=4,
        plateau_enabled=False,
        model_factory=factory,
        random_state=19,
    ).fit(X, y, groups=groups, specs=[_spec("constant")])

    assert np.array_equal(run.evaluation_indices, np.arange(len(X)))
    assert len(run.candidates[0].models) == 4
    assert run.candidates[0].oof_predictions.shape == (len(X), 2)
    assert run.candidates[0].oof_score == score_oof_predictions(
        run.candidates[0].oof_predictions,
        y,
        TABULAR_CLASSIFICATION_TASK,
    )
    assert factory.seeds["constant"] == [19, 20, 21, 22]
    for model in factory.models["constant"]:
        assert model.fit_rows is not None
        assert len(model.prediction_rows) == 1
        training_groups = set(groups[model.fit_rows])
        evaluation_groups = set(groups[model.prediction_rows[0]])
        assert training_groups.isdisjoint(evaluation_groups)


def test_large_dataset_uses_single_holdout_oof_model() -> None:
    row_count = 2_500
    X = np.column_stack(
        (
            np.arange(row_count, dtype=np.float32),
            np.linspace(-1.0, 1.0, row_count, dtype=np.float32),
        )
    )
    y = np.linspace(0.0, 10.0, row_count, dtype=np.float32)
    factory = _ConstantFactory()

    run = OOFEnsembleTrainer(
        TABULAR_REGRESSION_TASK,
        plateau_enabled=False,
        model_factory=factory,
        random_state=7,
    ).fit(X, y, groups=np.arange(row_count), specs=[_spec("mean")])

    assert len(run.candidates[0].models) == 1
    assert 0 < len(run.evaluation_indices) < row_count
    assert len(run.evaluation_indices) == row_count // 4
    assert np.array_equal(
        factory.models["mean"][0].prediction_rows[0],
        run.evaluation_indices,
    )


def test_oof_budget_shortfall_warns_after_first_fold_and_keeps_first_candidate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _Clock()
    factory = _TimedRegressionFactory(
        clock,
        lambda: "time limit is insufficient" in caplog.text,
    )
    X = np.arange(60, dtype=np.float32).reshape(30, 2)
    y = np.zeros(len(X), dtype=np.float32)
    specs = [
        _spec("first", {"constant": 0.0}),
        _spec("second", {"constant": 1.0}),
    ]

    with caplog.at_level("INFO", logger="falcon"):
        run = OOFEnsembleTrainer(
            TABULAR_REGRESSION_TASK,
            n_splits=3,
            plateau_enabled=False,
            time_limit=5.0,
            reserve_fraction=0.0,
            model_factory=factory,
            clock=clock,
        ).fit(X, y, groups=np.arange(len(X)), specs=specs)

    assert factory.calls == ["first", "first", "first"]
    assert factory.warning_visible_before_second_fold
    assert caplog.text.count("time limit is insufficient") == 1
    assert "Candidate 1/2" in caplog.text
    assert "estimated remaining time" in caplog.text
    assert run.stopped_for_budget
    assert run.elapsed_time == 9.0
    assert run.candidates[0].fit_time == 9.0
    assert run.ensemble.predict(X).shape == (len(X),)

    graph = run.ensemble.serialize().get_model()
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(
        graph.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    runtime_predictions = session.run(
        None,
        {session.get_inputs()[0].name: X},
    )[0]
    np.testing.assert_allclose(runtime_predictions, run.ensemble.predict(X))


@pytest.mark.parametrize(
    ("task", "y", "predictions"),
    [
        (
            TABULAR_CLASSIFICATION_TASK,
            np.asarray([0, 0, 1, 1]),
            [
                np.asarray([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]),
                np.asarray([[0.4, 0.6], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4]]),
            ],
        ),
        (
            TABULAR_REGRESSION_TASK,
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            [
                np.asarray([1.0, 2.0, 3.0, 4.0]),
                np.asarray([-1.0, 0.0, 1.0, 2.0]),
            ],
        ),
    ],
)
def test_greedy_selection_never_scores_below_best_candidate(
    task: str,
    y: npt.NDArray[Any],
    predictions: list[npt.NDArray[Any]],
) -> None:
    selection = greedy_weighted_selection(
        predictions,
        y,
        task,
        max_iterations=10,
    )

    best_individual = max(
        score_oof_predictions(values, y, task) for values in predictions
    )
    assert selection.score >= best_individual
    assert sum(selection.weights) == pytest.approx(1.0)
    if task == TABULAR_REGRESSION_TASK:
        assert selection.weights == pytest.approx((0.5, 0.5))
        assert selection.score == pytest.approx(0.0)


def _rare_class_probabilities() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Rows the rare class ranks higher on, but never above the majority class."""
    y = np.asarray([0] * 18 + [1] * 2)
    probabilities = np.tile(np.asarray([[0.95, 0.05]]), (len(y), 1))
    probabilities[y == 1] = (0.7, 0.3)
    return probabilities, y


def test_prior_correction_changes_which_class_a_score_credits() -> None:
    probabilities, y = _rare_class_probabilities()

    plain = score_oof_predictions(
        probabilities,
        y,
        TABULAR_CLASSIFICATION_TASK,
        prior_correct=False,
    )
    corrected = score_oof_predictions(
        probabilities,
        y,
        TABULAR_CLASSIFICATION_TASK,
        prior_correct=True,
    )

    assert plain == pytest.approx(0.5)
    assert corrected == pytest.approx(1.0)


def test_greedy_selection_forwards_the_prior_correction_setting() -> None:
    informative, y = _rare_class_probabilities()
    flat = np.tile(np.asarray([[0.95, 0.05]]), (len(y), 1))

    corrected = greedy_weighted_selection(
        [informative, flat],
        y,
        TABULAR_CLASSIFICATION_TASK,
        max_iterations=3,
    )
    plain = greedy_weighted_selection(
        [informative, flat],
        y,
        TABULAR_CLASSIFICATION_TASK,
        max_iterations=3,
        prior_correct=False,
    )

    assert corrected.score > plain.score


def test_plateau_stops_incremental_candidate_training_deterministically(
    caplog: pytest.LogCaptureFixture,
) -> None:
    X = np.column_stack(
        (
            np.arange(40, dtype=np.float32),
            np.linspace(-1.0, 1.0, 40, dtype=np.float32),
        )
    )
    y = np.arange(40) % 2
    specs = [_spec(f"constant-{index}") for index in range(6)]

    def train() -> tuple[list[str], tuple[float, ...], npt.NDArray[Any]]:
        run = OOFEnsembleTrainer(
            TABULAR_CLASSIFICATION_TASK,
            n_splits=2,
            max_iterations=5,
            plateau_patience=2,
            plateau_tolerance=0.0,
            model_factory=_ConstantFactory(),
            random_state=13,
        ).fit(X, y, groups=np.arange(len(X)), specs=specs)
        assert run.stopped_for_plateau
        assert not run.stopped_for_budget
        return (
            [candidate.spec.name for candidate in run.candidates],
            run.ensemble_score_history,
            run.ensemble.predict(X),
        )

    with caplog.at_level("INFO", logger="falcon"):
        first = train()
        second = train()

    assert first[0] == ["constant-0", "constant-1", "constant-2"]
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert np.array_equal(first[2], second[2])
    assert "OOF score plateau" in caplog.text


def _classification_schema(row_count: int, feature_count: int) -> DatasetSchema:
    return DatasetSchema(
        column_names=tuple(f"feature_{index}" for index in range(feature_count)),
        column_types=(ColumnTypes.NUMERIC_REGULAR,) * feature_count,
        target_name="target",
        target_kind="classification",
        dimensions=(row_count, feature_count),
    )


def test_parallel_classification_graph_matches_fold_bagged_native_predictions() -> None:
    X, y = make_classification(
        n_samples=96,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=5,
    )
    X = X.astype(np.float32)
    specs = [
        _spec("logistic-low-c", {"C": 0.2, "max_iter": 200}),
        _spec("logistic-high-c", {"C": 5.0, "max_iter": 200}),
    ]
    run = OOFEnsembleTrainer(
        TABULAR_CLASSIFICATION_TASK,
        n_splits=3,
        max_iterations=10,
        plateau_enabled=False,
        random_state=23,
    ).fit(X, y, groups=np.arange(len(X)), specs=specs)
    decoder = LabelDecoder()
    decoder.fit(
        X,
        np.where(y == 0, "negative", "positive"),
        _classification_schema(len(X), X.shape[1]),
    )
    graph = serialize_to_onnx(
        [run.ensemble.serialize(), decoder.serialize()],
        task=TABULAR_CLASSIFICATION_TASK,
    )

    onnx.checker.check_model(graph)
    assert all(
        node.domain in {"", "ai.onnx", "ai.onnx.ml"} for node in graph.graph.node
    )
    assert any("fold-1" in node.name for node in graph.graph.node)
    session = ort.InferenceSession(
        graph.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    runtime_outputs = session.run(None, {session.get_inputs()[0].name: X})
    runtime_probabilities = next(
        output for output in runtime_outputs if np.asarray(output).ndim == 2
    )
    runtime_labels = next(
        output
        for output in runtime_outputs
        if np.asarray(output).dtype.kind in {"O", "U"}
    )
    native_labels = decoder.transform(run.ensemble.predict(X))

    np.testing.assert_allclose(
        runtime_probabilities,
        run.ensemble.predict_proba(X),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(np.asarray(runtime_labels), native_labels)


def test_fnnx_round_trip_shares_preprocessing_across_fold_branches(
    tmp_path: Path,
) -> None:
    X, y = make_classification(
        n_samples=72,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        random_state=17,
    )
    raw_X = X.astype(np.object_)
    schema = _classification_schema(len(X), X.shape[1])
    groups = np.arange(len(X))
    encoder = MultiModalEncoder()
    encoder.fit(raw_X, y, schema, groups=groups)
    encoded_X = encoder.transform(raw_X)
    run = OOFEnsembleTrainer(
        TABULAR_CLASSIFICATION_TASK,
        n_splits=3,
        plateau_enabled=False,
        random_state=31,
    ).fit(
        encoded_X,
        y,
        groups=groups,
        specs=[_spec("logistic", {"C": 1.0, "max_iter": 200})],
    )
    decoder = LabelDecoder()
    decoder.fit(
        encoded_X,
        np.where(y == 0, "negative", "positive"),
        schema,
    )
    bundle = FNNXSerializer(
        [encoder.serialize(), run.ensemble.serialize(), decoder.serialize()],
        task=TABULAR_CLASSIFICATION_TASK,
        schema=schema,
    ).serialize()
    extracted = extract_fnnx_graph(bundle)

    assert_fnnx_conforms(extracted)
    fold_classifier_nodes = [
        node
        for node in extracted.model.graph.node
        if node.op_type == "LinearClassifier" and "fold-" in node.name
    ]
    assert len(fold_classifier_nodes) == 3
    assert len({tuple(node.input) for node in fold_classifier_nodes}) == 1
    assert fold_classifier_nodes[0].input[0].startswith("falcon-pl-0/")

    artifact_path = tmp_path / "fold-bagged-classifier.fnnx"
    artifact_path.write_bytes(bundle)
    runtime = Runtime(str(artifact_path))
    native_probabilities = run.ensemble.predict_proba(encoded_X)
    native_labels = decoder.transform(run.ensemble.predict(encoded_X))

    np.testing.assert_allclose(
        runtime.predict_proba(raw_X),
        native_probabilities,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        native_probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(runtime.predict(raw_X), native_labels)


def test_parallel_regression_graph_applies_fold_and_member_weighted_means() -> None:
    X = np.column_stack(
        (
            np.arange(60, dtype=np.float32),
            np.linspace(-1.0, 1.0, 60, dtype=np.float32),
        )
    )
    y = np.zeros(len(X), dtype=np.float32)
    specs = [
        _spec("negative", {"constant": -1.0}),
        _spec("positive", {"constant": 1.0}),
    ]
    run = OOFEnsembleTrainer(
        TABULAR_REGRESSION_TASK,
        n_splits=3,
        max_iterations=5,
        plateau_enabled=False,
        model_factory=_DummyRegressionFactory(),
        random_state=29,
    ).fit(X, y, groups=np.arange(len(X)), specs=specs)
    serialized = run.ensemble.serialize()
    graph = serialized.get_model()

    assert len(run.ensemble.members) == 2
    assert run.ensemble.weights == pytest.approx((0.5, 0.5))
    onnx.checker.check_model(graph)
    assert all(
        node.domain in {"", "ai.onnx", "ai.onnx.ml"} for node in graph.graph.node
    )
    assert any("member-1" in node.name for node in graph.graph.node)
    assert any("average_folds" in node.name for node in graph.graph.node)
    session = ort.InferenceSession(
        graph.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    runtime_predictions = session.run(
        None,
        {session.get_inputs()[0].name: X},
    )[0]

    np.testing.assert_allclose(
        runtime_predictions,
        run.ensemble.predict(X),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"plateau_patience": 0}, "plateau_patience"),
        ({"plateau_tolerance": -1.0}, "plateau_tolerance"),
        ({"n_splits": 1}, "n_splits"),
    ],
)
def test_oof_ensemble_trainer_rejects_invalid_settings(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OOFEnsembleTrainer(TABULAR_REGRESSION_TASK, **kwargs)
