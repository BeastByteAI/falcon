from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from numpy import typing as npt
from onnx import TensorProto, helper
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from falcon.config import ONNX_IR_VERSION, ONNX_OPSET_VERSION
from falcon.serialization import SerializedModelRepr
from falcon.tabular.decision import fit_decision_weights, serialize_decision_rule


def _binary_probabilities(
    n_rows: int = 800,
    prevalence: float = 0.12,
    seed: int = 5,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    targets = (rng.random(n_rows) < prevalence).astype(np.int64)
    logits = (
        rng.normal(size=n_rows) + 1.4 * targets + np.log(prevalence / (1 - prevalence))
    )
    positive = 1.0 / (1.0 + np.exp(-logits))
    probabilities = np.column_stack((1.0 - positive, positive)).astype(np.float32)
    return probabilities, targets


def _multiclass_probabilities(
    n_rows: int = 900,
    seed: int = 3,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    prior = np.asarray([0.7, 0.2, 0.1])
    targets = rng.choice(3, size=n_rows, p=prior).astype(np.int64)
    signal = rng.normal(size=(n_rows, 3))
    signal[np.arange(n_rows), targets] += 1.5
    scores = prior * np.exp(signal)
    probabilities = (scores / scores.sum(axis=1, keepdims=True)).astype(np.float32)
    return probabilities, targets


def test_binary_weights_are_equivalent_to_a_threshold() -> None:
    probabilities, targets = _binary_probabilities()

    weights = fit_decision_weights(probabilities, targets, "balanced_accuracy")
    threshold = weights[0] / (weights[0] + weights[1])
    weighted_labels = np.argmax(
        probabilities * np.asarray(weights, dtype=np.float32),
        axis=1,
    )

    assert weights != (1.0, 1.0)
    np.testing.assert_array_equal(
        weighted_labels,
        (probabilities[:, 1] > threshold).astype(np.int64),
    )
    assert balanced_accuracy_score(targets, weighted_labels) > balanced_accuracy_score(
        targets, np.argmax(probabilities, axis=1)
    )


def _score(
    metric: str,
    targets: npt.NDArray[np.int64],
    predicted: npt.NDArray[np.int64],
) -> float:
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(targets, predicted))
    if metric == "mcc":
        return float(matthews_corrcoef(targets, predicted))
    return float(f1_score(targets, predicted, average="macro", zero_division=0.0))


@pytest.mark.parametrize("metric", ["balanced_accuracy", "f1", "mcc"])
def test_binary_tuning_never_scores_below_plain_argmax(metric: str) -> None:
    probabilities, targets = _binary_probabilities()

    weights = fit_decision_weights(probabilities, targets, metric)
    labels = np.argmax(probabilities * np.asarray(weights, dtype=np.float32), axis=1)

    assert len(weights) == 2
    assert not np.array_equal(labels, np.zeros_like(labels))
    assert _score(metric, targets, labels) >= _score(
        metric, targets, np.argmax(probabilities, axis=1)
    )


def test_binary_f1_is_the_macro_average_over_both_classes() -> None:
    """`"f1"` must not depend on which label the encoder happened to map to 1."""
    probabilities, targets = _binary_probabilities()

    weights = fit_decision_weights(probabilities, targets, "f1")
    flipped = fit_decision_weights(probabilities[:, ::-1], 1 - targets, "f1")

    assert weights == pytest.approx(tuple(reversed(flipped)), abs=1e-6)


def test_multiclass_coordinate_ascent_improves_the_metric() -> None:
    probabilities, targets = _multiclass_probabilities()

    weights = fit_decision_weights(probabilities, targets, "balanced_accuracy")
    tuned = balanced_accuracy_score(
        targets,
        np.argmax(probabilities * np.asarray(weights, dtype=np.float32), axis=1),
    )

    assert len(weights) == 3
    assert tuned > balanced_accuracy_score(targets, np.argmax(probabilities, axis=1))


def test_multiclass_weights_are_normalized_without_changing_labels() -> None:
    probabilities, targets = _multiclass_probabilities()

    weights = np.asarray(
        fit_decision_weights(probabilities, targets, "balanced_accuracy"),
        dtype=np.float32,
    )

    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    for scale in (0.25, 7.4, 100.0):
        np.testing.assert_array_equal(
            np.argmax(probabilities * weights, axis=1),
            np.argmax(probabilities * (weights * scale), axis=1),
        )


def test_small_class_count_returns_the_no_op_rule() -> None:
    probabilities, targets = _binary_probabilities(n_rows=400, prevalence=0.05, seed=11)
    assert int(np.bincount(targets).min()) < 50

    assert fit_decision_weights(probabilities, targets, "balanced_accuracy") == (
        1.0,
        1.0,
    )


def test_rule_that_cannot_beat_argmax_returns_the_no_op_rule() -> None:
    rng = np.random.default_rng(19)
    targets = np.repeat(np.asarray([0, 1], dtype=np.int64), 200)
    confidence = rng.uniform(0.9, 0.99, size=len(targets))
    positive = np.where(targets == 1, confidence, 1.0 - confidence)
    probabilities = np.column_stack((1.0 - positive, positive)).astype(np.float32)

    assert fit_decision_weights(probabilities, targets, "balanced_accuracy") == (
        1.0,
        1.0,
    )


def test_unknown_metric_is_rejected() -> None:
    probabilities, targets = _binary_probabilities(n_rows=200)
    with pytest.raises(ValueError, match="decision metric"):
        fit_decision_weights(probabilities, targets, "accuracy")


@pytest.mark.parametrize(
    ("probabilities", "targets", "message"),
    [
        (np.zeros((4, 1), dtype=np.float32), np.zeros(4, dtype=np.int64), "two class"),
        (
            np.full((4, 2), 0.5, dtype=np.float32),
            np.zeros(3, dtype=np.int64),
            "one value per probability row",
        ),
        (
            np.full((4, 2), 0.5, dtype=np.float32),
            np.zeros(4, dtype=np.float64),
            "integer encoded",
        ),
    ],
)
def test_invalid_inputs_are_rejected(
    probabilities: npt.NDArray[np.float32],
    targets: npt.NDArray[np.int64],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_decision_weights(probabilities, targets, "balanced_accuracy")


def _classifier_graph(n_classes: int) -> SerializedModelRepr:
    """A minimal `[labels, probabilities]` graph shaped like a learner export."""
    input_info = helper.make_tensor_value_info(
        "model_input",
        TensorProto.FLOAT,
        [None, n_classes],
    )
    nodes = [
        helper.make_node("Softmax", ["model_input"], ["probabilities"], axis=1),
        helper.make_node(
            "ArgMax",
            ["probabilities"],
            ["label"],
            axis=1,
            keepdims=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "classifier",
        [input_info],
        [
            helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
            helper.make_tensor_value_info(
                "probabilities",
                TensorProto.FLOAT,
                [None, n_classes],
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", ONNX_OPSET_VERSION)],
        ir_version=ONNX_IR_VERSION,
    )
    return SerializedModelRepr(model, 1, 2, ["FLOAT32"], [[None, n_classes]])


@pytest.mark.parametrize("weights", [(0.2, 0.8), (1.0, 3.0, 0.5)])
def test_serialized_rule_matches_the_weighted_argmax(
    weights: tuple[float, ...],
) -> None:
    n_classes = len(weights)
    serialized = serialize_decision_rule(_classifier_graph(n_classes), weights)
    model = serialized.get_model()
    onnx.checker.check_model(model, full_check=True)

    rng = np.random.default_rng(7)
    logits = rng.normal(size=(64, n_classes)).astype(np.float32)
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    labels, probabilities = session.run(None, {"model_input": logits})
    expected = np.argmax(
        probabilities * np.asarray(weights, dtype=np.float32),
        axis=1,
    )

    assert np.asarray(labels).dtype == np.int64
    np.testing.assert_array_equal(np.asarray(labels), expected)
    assert {"Mul", "ArgMax"} <= {
        node.op_type for node in model.graph.node if "falcon_decision" in node.name
    }
    assert all(node.domain in {"", "ai.onnx"} for node in model.graph.node)


def test_serialized_rule_rejects_a_mismatched_class_count() -> None:
    with pytest.raises(ValueError, match="one weight per probability column"):
        serialize_decision_rule(_classifier_graph(3), (0.5, 0.5))
