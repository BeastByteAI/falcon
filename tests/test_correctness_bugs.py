from __future__ import annotations

import warnings
from typing import Any, get_origin, get_type_hints

import numpy as np
import pytest
from numpy import typing as npt

from falcon import Predictor, sklapi
from falcon.abstract import Pipeline
from falcon.runtime import Runtime
from falcon.serialization import (
    DEFAULT_PRODUCER_NAME,
    SerializedModelRepr,
    input_tags,
)
from falcon.sklapi import FalconTabularClassifier, FalconTabularRegressor
from falcon.tabular.ingestion import ingest_data
from falcon.types import ColumnTypes, DatasetSchema


class _TypedStep:
    def __init__(self, input_type: type[Any], output_type: type[Any]) -> None:
        self.input_type = input_type
        self.output_type = output_type

    def get_input_type(self) -> type[Any]:
        return self.input_type

    def get_output_type(self) -> type[Any]:
        return self.output_type

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None:
        return None

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return X

    def serialize(self) -> SerializedModelRepr:
        raise NotImplementedError


class _RecordingRuntime:
    def __init__(self, outputs: dict[str, npt.NDArray[Any]]) -> None:
        self.outputs = outputs
        self.inputs: dict[str, npt.NDArray[Any]] | None = None

    def compute(
        self,
        inputs: dict[str, npt.NDArray[Any]],
        attributes: dict[str, Any],
    ) -> dict[str, npt.NDArray[Any]]:
        self.inputs = inputs
        return self.outputs


def test_pipeline_validates_the_second_element_type() -> None:
    pipeline = Pipeline(task="tabular_regression", dataset_size=(1, 1))
    pipeline.add_step(_TypedStep(str, int))

    with pytest.raises(RuntimeError, match="input type"):
        pipeline.add_step(_TypedStep(str, float))


def test_datetime_input_tag_uses_the_producer_name() -> None:
    assert input_tags[ColumnTypes.DATETIME_YMDHMS_ISO8601] == [
        f"{DEFAULT_PRODUCER_NAME}::datetime_ymdhms_iso8601:v1"
    ]


@pytest.mark.parametrize(
    ("estimator_class", "expected_task"),
    [
        (FalconTabularClassifier, "tabular_classification"),
        (FalconTabularRegressor, "tabular_regression"),
    ],
)
def test_sklapi_resolves_configuration_for_its_task(
    monkeypatch: pytest.MonkeyPatch,
    estimator_class: type[FalconTabularClassifier | FalconTabularRegressor],
    expected_task: str,
) -> None:
    calls: list[tuple[str, str]] = []

    class RecordingPredictor:
        def __init__(
            self,
            task: str,
            preset: str,
            eval_strategy: str,
        ) -> None:
            del eval_strategy
            calls.append((task, preset))

    monkeypatch.setattr(sklapi, "Predictor", RecordingPredictor)

    estimator = estimator_class(preset="balanced")
    estimator._new_predictor()

    assert calls == [(expected_task, "balanced")]


@pytest.mark.parametrize(
    ("features", "target", "message"),
    [
        ([], 1, "Features List cannot be empty"),
        (["feature"], 1, "Expected list of integers as features"),
        ([0], "target", "Expected integer as target"),
    ],
)
def test_ingestion_raises_for_invalid_numpy_column_selectors(
    features: list[int] | list[str], target: int | str, message: str
) -> None:
    data = np.arange(12).reshape(4, 3)

    with pytest.raises(ValueError, match=message):
        ingest_data(
            data,
            task="tabular_regression",
            features=features,
            target=target,
        )


def test_runtime_predict_does_not_mutate_caller_inputs() -> None:
    outputs: dict[str, npt.NDArray[np.float32]] = {
        "y_pred": np.asarray([1.0, 2.0], dtype=np.float32)
    }
    backend = _RecordingRuntime(outputs)
    runtime = Runtime.__new__(Runtime)
    runtime.runtime = backend
    runtime._input_names = ["first", "second"]
    first: npt.NDArray[np.float32] = np.asarray([1.0, 2.0], dtype=np.float32)
    second: npt.NDArray[np.float32] = np.asarray([3.0, 4.0], dtype=np.float32)
    inputs = {"first": first, "second": second}

    result = runtime._predict(inputs)

    assert result is outputs
    assert inputs["first"] is first
    assert inputs["second"] is second
    assert first.shape == (2,)
    assert second.shape == (2,)
    assert backend.inputs is not None
    assert backend.inputs["first"].shape == (2, 1)
    assert backend.inputs["second"].shape == (2, 1)
    assert get_origin(get_type_hints(Runtime._predict)["return"]) is dict


def test_predictor_does_not_replace_warnings_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def user_warn(
        message: Warning | str,
        category: type[Warning] | None = None,
        stacklevel: int = 1,
        source: Any = None,
    ) -> None:
        return None

    monkeypatch.setattr(warnings, "warn", user_warn)

    Predictor("tabular_regression", preset="fast", eval_strategy=None)

    assert warnings.warn is user_warn
