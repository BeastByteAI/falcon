from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np
import onnx
from numpy import typing as npt
from onnx import helper

from falcon.config import ONNX_OPSET_VERSION
from falcon.serialization import SerializedModelRepr


def _validated_vector(
    values: npt.NDArray[Any],
    name: str,
) -> npt.NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"Conformal {name} must be a non-empty vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"Conformal {name} must contain only finite values")
    return vector


def fit_conformal_quantile(
    predictions: npt.NDArray[Any],
    targets: npt.NDArray[Any],
    alpha: float,
) -> float:
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Conformal alpha must be between zero and one")
    predicted = _validated_vector(predictions, "predictions")
    actual = _validated_vector(targets, "targets")
    if predicted.shape != actual.shape:
        raise ValueError("Conformal predictions and targets must have the same shape")

    residuals = np.abs(actual - predicted)
    rank = min(math.ceil((len(residuals) + 1) * (1 - alpha)), len(residuals))
    quantile = float(np.partition(residuals, rank - 1)[rank - 1])
    if quantile > np.finfo(np.float32).max:
        raise ValueError("The conformal quantile cannot be represented as float32")
    graph_quantile = np.float32(quantile)
    if graph_quantile < quantile:
        graph_quantile = np.nextafter(graph_quantile, np.float32(np.inf))
    return float(graph_quantile)


def serialize_conformal_interval(
    serialized: SerializedModelRepr,
    quantile: float,
) -> SerializedModelRepr:
    if not np.isfinite(quantile) or quantile < 0:
        raise ValueError("The conformal quantile must be finite and non-negative")

    model = deepcopy(serialized.get_model())
    if len(model.graph.output) != 1:
        raise ValueError("A conformal regression graph must expose one prediction")
    prediction_output = model.graph.output[0]
    if (
        not prediction_output.type.HasField("tensor_type")
        or prediction_output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise ValueError("Regression predictions must be a float tensor")

    source_name = prediction_output.name
    quantile_name = "falcon_conformal_quantile"
    lower_name = "falcon_conformal_lower"
    upper_name = "falcon_conformal_upper"
    model.graph.initializer.append(
        helper.make_tensor(
            quantile_name,
            onnx.TensorProto.FLOAT,
            [],
            [quantile],
        )
    )
    model.graph.node.extend(
        [
            helper.make_node(
                "Sub",
                [source_name, quantile_name],
                [lower_name],
                name="falcon_conformal/subtract_quantile",
            ),
            helper.make_node(
                "Add",
                [source_name, quantile_name],
                [upper_name],
                name="falcon_conformal/add_quantile",
            ),
        ]
    )
    lower_output = deepcopy(prediction_output)
    lower_output.name = lower_name
    upper_output = deepcopy(prediction_output)
    upper_output.name = upper_name
    model.graph.output.extend([lower_output, upper_output])
    if not any(opset.domain in {"", "ai.onnx"} for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("", ONNX_OPSET_VERSION))

    return SerializedModelRepr(
        model,
        serialized.get_n_inputs(),
        serialized.get_n_outputs() + 2,
        serialized.get_initial_types().copy(),
        [shape.copy() for shape in serialized.get_initial_shapes()],
        serialized.get_type(),
    )


__all__ = ["fit_conformal_quantile", "serialize_conformal_interval"]
