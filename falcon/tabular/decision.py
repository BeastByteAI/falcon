from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import onnx
from numpy import typing as npt
from onnx import helper
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from falcon.config import DECISION_METRICS, ONNX_OPSET_VERSION
from falcon.serialization import SerializedModelRepr

_MIN_CLASS_COUNT = 50
_THRESHOLD_GRID_SIZE = 101
_LOG_WEIGHT_GRID = np.linspace(-2.0, 2.0, 33)
_COORDINATE_SWEEPS = 3


def _validated_probabilities(
    probabilities: npt.NDArray[Any],
) -> npt.NDArray[np.float32]:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(
            "Decision probabilities must contain at least two class columns"
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Decision probabilities must be finite and non-negative")
    if (values.sum(axis=1) <= 0).any():
        raise ValueError("Each probability row must contain a positive value")
    return values


def _validated_targets(
    targets: npt.NDArray[Any],
    n_rows: int,
    n_classes: int,
) -> npt.NDArray[np.int64]:
    values = np.asarray(targets)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError("Decision targets must contain one value per probability row")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Decision targets must be integer encoded")
    encoded = values.astype(np.int64, copy=False)
    if (encoded < 0).any() or (encoded >= n_classes).any():
        raise ValueError("Decision targets contain an unknown class index")
    return encoded


def _metric_score(
    metric: str,
    targets: npt.NDArray[np.int64],
    predicted: npt.NDArray[np.int64],
) -> float:
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(targets, predicted))
    if metric == "mcc":
        return float(matthews_corrcoef(targets, predicted))
    # Macro on binary too: which label encodes to 1 is an artefact of alphabetical label
    # encoding, so a positive-class F1 would optimise an arbitrary class.
    return float(f1_score(targets, predicted, average="macro", zero_division=0.0))


def _decision_labels(
    probabilities: npt.NDArray[np.float32],
    weights: npt.NDArray[np.float32],
) -> npt.NDArray[np.int64]:
    return np.argmax(probabilities * weights, axis=1).astype(np.int64)


def _tune_binary(
    probabilities: npt.NDArray[np.float32],
    targets: npt.NDArray[np.int64],
    metric: str,
) -> tuple[npt.NDArray[np.float32], float]:
    # Thresholds are drawn from quantiles of the positive score: at low prevalence a
    # uniform grid over [0, 1] spends nearly every point where no row ever lands.
    quantiles = np.quantile(
        probabilities[:, 1], np.linspace(0.0, 1.0, _THRESHOLD_GRID_SIZE)
    )
    best_weights = np.asarray([1.0, 1.0], dtype=np.float32)
    best_score = -np.inf
    for threshold in np.unique(quantiles):
        weights = np.asarray([threshold, 1.0 - threshold], dtype=np.float32)
        score = _metric_score(metric, targets, _decision_labels(probabilities, weights))
        if score > best_score:
            best_weights, best_score = weights, score
    return best_weights, best_score


def _tune_multiclass(
    probabilities: npt.NDArray[np.float32],
    targets: npt.NDArray[np.int64],
    metric: str,
    n_classes: int,
) -> tuple[npt.NDArray[np.float32], float]:
    log_weights = np.zeros(n_classes, dtype=np.float64)
    best_weights = np.ones(n_classes, dtype=np.float32)
    best_score = _metric_score(
        metric,
        targets,
        _decision_labels(probabilities, best_weights),
    )
    for _ in range(_COORDINATE_SWEEPS):
        improved = False
        for index in range(1, n_classes):
            selected = log_weights[index]
            for value in _LOG_WEIGHT_GRID:
                log_weights[index] = value
                weights = np.exp(log_weights).astype(np.float32)
                score = _metric_score(
                    metric,
                    targets,
                    _decision_labels(probabilities, weights),
                )
                if score > best_score:
                    best_weights, best_score, selected = weights, score, value
                    improved = True
            log_weights[index] = selected
        if not improved:
            break
    # The rule is scale-invariant; rescaling to a mean of one keeps reported weights
    # readable without changing any decision.
    normalized = (best_weights.astype(np.float64) / best_weights.mean()).astype(
        np.float32
    )
    return normalized, best_score


def fit_decision_weights(
    probabilities: npt.NDArray[Any],
    targets: npt.NDArray[Any],
    metric: str,
) -> tuple[float, ...]:
    """Fit per-class weights `w` so that `argmax(p * w)` maximises `metric`.

    Returns an all-ones no-op when the rarest class is too small to tune on, or when
    no weighting strictly beats plain argmax.
    """
    if metric not in DECISION_METRICS:
        raise ValueError(
            f"decision metric must be one of {', '.join(sorted(DECISION_METRICS))}"
        )
    values = _validated_probabilities(probabilities)
    n_classes = values.shape[1]
    encoded_targets = _validated_targets(targets, len(values), n_classes)
    no_op = (1.0,) * n_classes

    counts = np.bincount(encoded_targets, minlength=n_classes)
    if int(counts.min()) < _MIN_CLASS_COUNT:
        return no_op

    baseline_score = _metric_score(
        metric,
        encoded_targets,
        _decision_labels(values, np.ones(n_classes, dtype=np.float32)),
    )
    if n_classes == 2:
        weights, score = _tune_binary(values, encoded_targets, metric)
    else:
        weights, score = _tune_multiclass(values, encoded_targets, metric, n_classes)
    if score <= baseline_score:
        return no_op
    return tuple(float(weight) for weight in weights)


def serialize_decision_rule(
    serialized: SerializedModelRepr,
    weights: tuple[float, ...],
) -> SerializedModelRepr:
    weight_values = np.asarray(weights, dtype=np.float32)
    if weight_values.ndim != 1 or weight_values.size < 2:
        raise ValueError("A decision rule needs one weight per class")
    if not np.isfinite(weight_values).all() or (weight_values < 0).any():
        raise ValueError("Decision weights must be finite and non-negative")

    model = deepcopy(serialized.get_model())
    if len(model.graph.output) < 2:
        raise ValueError("A decision rule graph must expose labels and probabilities")
    label_output = model.graph.output[0]
    probability_output = model.graph.output[-1]
    if (
        not probability_output.type.HasField("tensor_type")
        or probability_output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise ValueError("Classifier probabilities must be a float tensor")
    if (
        not label_output.type.HasField("tensor_type")
        or label_output.type.tensor_type.elem_type != onnx.TensorProto.INT64
    ):
        raise ValueError("Classifier labels must be an int64 tensor")
    dimensions = probability_output.type.tensor_type.shape.dim
    if (
        len(dimensions) == 2
        and dimensions[1].dim_value
        and dimensions[1].dim_value != weight_values.size
    ):
        raise ValueError("The decision rule needs one weight per probability column")

    weights_name = "falcon_decision_weights"
    scores_name = "falcon_decision_scores"
    label_name = "falcon_decision_label"
    model.graph.initializer.append(
        helper.make_tensor(
            weights_name,
            onnx.TensorProto.FLOAT,
            [1, int(weight_values.size)],
            weight_values.tolist(),
        )
    )
    model.graph.node.extend(
        [
            helper.make_node(
                "Mul",
                [probability_output.name, weights_name],
                [scores_name],
                name="falcon_decision/apply_weights",
            ),
            helper.make_node(
                "ArgMax",
                [scores_name],
                [label_name],
                axis=1,
                keepdims=0,
                name="falcon_decision/argmax",
            ),
        ]
    )
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info(label_name, onnx.TensorProto.INT64, [None])
    )
    if not any(opset.domain in {"", "ai.onnx"} for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("", ONNX_OPSET_VERSION))

    return SerializedModelRepr(
        model,
        serialized.get_n_inputs(),
        serialized.get_n_outputs(),
        serialized.get_initial_types().copy(),
        [shape.copy() for shape in serialized.get_initial_shapes()],
        serialized.get_type(),
    )


__all__ = ["fit_decision_weights", "serialize_decision_rule"]
