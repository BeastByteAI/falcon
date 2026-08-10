from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import onnx
from numpy import typing as npt
from onnx import helper
from scipy.optimize import minimize_scalar

from falcon.config import ONNX_OPSET_VERSION
from falcon.serialization import SerializedModelRepr

_PROBABILITY_FLOOR = np.float32(1e-7)
_MIN_LOG_TEMPERATURE = -5.0
_MAX_LOG_TEMPERATURE = 5.0


def temperature_scale_probabilities(
    probabilities: npt.NDArray[Any],
    temperature: float,
) -> npt.NDArray[np.float32]:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(
            "Classification probabilities must contain at least two class columns"
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Classification probabilities must be finite and non-negative")
    if (values.sum(axis=1) <= 0).any():
        raise ValueError("Each probability row must contain a positive value")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite value greater than zero")

    logits = np.log(np.maximum(values, _PROBABILITY_FLOOR)) / np.float32(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return np.asarray(
        exponentials / np.sum(exponentials, axis=1, keepdims=True),
        dtype=np.float32,
    )


def _validated_targets(
    targets: npt.NDArray[Any],
    n_rows: int,
    n_classes: int,
) -> npt.NDArray[np.int64]:
    values = np.asarray(targets)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(
            "Calibration targets must contain one value per probability row"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Calibration targets must be integer encoded")
    encoded = values.astype(np.int64, copy=False)
    if (encoded < 0).any() or (encoded >= n_classes).any():
        raise ValueError("Calibration targets contain an unknown class index")
    return encoded


def _negative_log_likelihood(
    probabilities: npt.NDArray[np.float32],
    targets: npt.NDArray[np.int64],
) -> float:
    selected = probabilities[np.arange(len(targets)), targets]
    return -float(np.mean(np.log(np.maximum(selected, _PROBABILITY_FLOOR))))


def fit_temperature(
    probabilities: npt.NDArray[Any],
    targets: npt.NDArray[Any],
) -> float:
    values = temperature_scale_probabilities(probabilities, 1.0)
    encoded_targets = _validated_targets(targets, len(values), values.shape[1])
    baseline_loss = _negative_log_likelihood(values, encoded_targets)

    def objective(log_temperature: float) -> float:
        scaled = temperature_scale_probabilities(
            values,
            float(np.exp(log_temperature)),
        )
        return _negative_log_likelihood(scaled, encoded_targets)

    result = minimize_scalar(
        objective,
        bounds=(_MIN_LOG_TEMPERATURE, _MAX_LOG_TEMPERATURE),
        method="bounded",
        options={"xatol": 1e-5},
    )
    if not result.success:
        return 1.0
    temperature = float(np.float32(np.exp(result.x)))
    if objective(float(np.log(temperature))) >= baseline_loss - 1e-7:
        return 1.0
    return temperature


def serialize_temperature_scaling(
    serialized: SerializedModelRepr,
    temperature: float,
) -> SerializedModelRepr:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite value greater than zero")

    model = deepcopy(serialized.get_model())
    if len(model.graph.output) < 2:
        raise ValueError(
            "A calibrated classifier graph must expose labels and probabilities"
        )
    probability_output = model.graph.output[-1]
    if (
        not probability_output.type.HasField("tensor_type")
        or probability_output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise ValueError("Classifier probabilities must be a float tensor")

    source_name = probability_output.name
    floor_name = "falcon_temperature_probability_floor"
    temperature_name = "falcon_temperature_value"
    clipped_name = "falcon_temperature_clipped_probabilities"
    logits_name = "falcon_temperature_log_probabilities"
    scaled_logits_name = "falcon_temperature_scaled_logits"
    calibrated_name = "falcon_calibrated_probabilities"
    model.graph.initializer.extend(
        [
            helper.make_tensor(
                floor_name,
                onnx.TensorProto.FLOAT,
                [],
                [float(_PROBABILITY_FLOOR)],
            ),
            helper.make_tensor(
                temperature_name,
                onnx.TensorProto.FLOAT,
                [],
                [temperature],
            ),
        ]
    )
    model.graph.node.extend(
        [
            helper.make_node(
                "Clip",
                [source_name, floor_name],
                [clipped_name],
                name="falcon_temperature/clip",
            ),
            helper.make_node(
                "Log",
                [clipped_name],
                [logits_name],
                name="falcon_temperature/log",
            ),
            helper.make_node(
                "Div",
                [logits_name, temperature_name],
                [scaled_logits_name],
                name="falcon_temperature/divide",
            ),
            helper.make_node(
                "Softmax",
                [scaled_logits_name],
                [calibrated_name],
                axis=1,
                name="falcon_temperature/softmax",
            ),
        ]
    )
    calibrated_output = deepcopy(probability_output)
    calibrated_output.name = calibrated_name
    model.graph.output[-1].CopyFrom(calibrated_output)
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


__all__ = [
    "fit_temperature",
    "serialize_temperature_scaling",
    "temperature_scale_probabilities",
]
