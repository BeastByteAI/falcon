from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt

try:
    from fnnx.handlers.local import LocalHandler as _LocalHandler
    from fnnx.runtime import Runtime as _Runtime
except ImportError:
    _Runtime = None
from falcon.constants import (
    DEFAULT_PRODUCER_NAME,
    TABULAR_CLASSIFICATION_TASK,
    TABULAR_REGRESSION_TASK,
)


class Runtime:
    def __init__(self, model_path: str) -> None:
        if _Runtime is None:
            raise ImportError("FNNX is not installed.")
        self.runtime = _Runtime(model_path)
        handler: _LocalHandler = self.runtime.handler

        self._input_names = list(handler.input_specs.keys())
        self._output_names = list(handler.output_specs.keys())

        producer_tags = handler.manifest.get("producer_tags", [])

        self.task = self._detect_task(producer_tags)

        if self.task is None:
            raise RuntimeError("Could not detect task from model tags.")

    def _detect_task(self, producer_tags: list[str]) -> str | None:
        for tag in producer_tags:
            if tag.startswith(
                f"{DEFAULT_PRODUCER_NAME}::{TABULAR_CLASSIFICATION_TASK}"
            ):
                return TABULAR_CLASSIFICATION_TASK
            elif tag.startswith(f"{DEFAULT_PRODUCER_NAME}::{TABULAR_REGRESSION_TASK}"):
                return TABULAR_REGRESSION_TASK
        return None

    def _predict(
        self,
        X: npt.NDArray[Any] | pd.DataFrame | dict[str, npt.NDArray[Any]],
    ) -> dict[str, npt.NDArray[Any]]:
        if isinstance(X, pd.DataFrame):
            inputs = X.to_dict(orient="list")
        elif isinstance(X, np.ndarray):
            inputs = {name: X[:, i] for i, name in enumerate(self._input_names)}
        else:
            inputs = X

        reshaped_inputs = {
            name: np.asarray(values).reshape(-1, 1) for name, values in inputs.items()
        }

        return self.runtime.compute(reshaped_inputs, {})

    def predict(
        self,
        X: npt.NDArray[Any] | pd.DataFrame | dict[str, npt.NDArray[Any]],
    ) -> npt.NDArray[Any]:
        return self._predict(X)["y_pred"]

    def predict_proba(
        self,
        X: npt.NDArray[Any] | pd.DataFrame | dict[str, npt.NDArray[Any]],
    ) -> npt.NDArray[Any]:
        if self.task != TABULAR_CLASSIFICATION_TASK:
            raise RuntimeError(f"Called predict_proba on a model for {self.task} task.")
        return self._predict(X)["probabilities"]

    def predict_interval(
        self,
        X: npt.NDArray[Any] | pd.DataFrame | dict[str, npt.NDArray[Any]],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        if self.task != TABULAR_REGRESSION_TASK:
            raise RuntimeError(
                f"Called predict_interval on a model for {self.task} task."
            )
        if not {"y_lower", "y_upper"}.issubset(self._output_names):
            raise RuntimeError("This model does not expose prediction intervals.")
        outputs = self._predict(X)
        return outputs["y_lower"], outputs["y_upper"]
