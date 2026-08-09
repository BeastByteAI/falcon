from typing import Any

import numpy as np
from numpy import typing as npt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import log_loss, mean_squared_error

from falcon.config import ML_ONNX_OPSET_VERSION, ONNX_OPSET_VERSION
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.serialization import SerializedModelRepr


class SklearnModel:
    def __init__(self, estimator: Any, task: str) -> None:
        if task not in {TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK}:
            raise ValueError(f"Unknown task `{task}`")
        self.estimator = estimator
        self.task = task
        self._shape: list[int | None] | None = None

    def _fit_estimator(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        sample_weight: npt.NDArray[np.float64] | None,
    ) -> None:
        if sample_weight is None:
            self.estimator.fit(X, y)
        else:
            self.estimator.fit(X, y, sample_weight=sample_weight)

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self._shape = [None, *X.shape[1:]]
        if validation_data is None and early_stopping_rounds is None:
            self._fit_estimator(X, y, sample_weight)
            return
        if validation_data is None or early_stopping_rounds is None:
            raise ValueError(
                "validation_data and early_stopping_rounds must be provided together"
            )
        if not isinstance(
            self.estimator,
            (HistGradientBoostingClassifier, HistGradientBoostingRegressor),
        ):
            raise ValueError(
                "This sklearn estimator does not support external early stopping"
            )
        self._fit_hist_gradient_boosting(
            X,
            y,
            sample_weight,
            validation_data,
            early_stopping_rounds,
        )

    def _fit_hist_gradient_boosting(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        sample_weight: npt.NDArray[np.float64] | None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]],
        early_stopping_rounds: int,
    ) -> None:
        validation_X, validation_y = validation_data
        max_iter = int(self.estimator.max_iter)
        self.estimator.set_params(warm_start=True)
        best_iteration = 1
        best_loss = float("inf")
        iterations_without_improvement = 0
        trained_iterations = 0
        for iteration in range(1, max_iter + 1):
            self.estimator.set_params(max_iter=iteration)
            self._fit_estimator(X, y, sample_weight)
            trained_iterations = iteration
            if self.task == TABULAR_CLASSIFICATION_TASK:
                loss = log_loss(
                    validation_y,
                    self.estimator.predict_proba(validation_X),
                    labels=self.estimator.classes_,
                )
            else:
                loss = mean_squared_error(
                    validation_y,
                    self.estimator.predict(validation_X),
                )
            if loss < best_loss - float(self.estimator.tol):
                best_loss = loss
                best_iteration = iteration
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            if iterations_without_improvement >= early_stopping_rounds:
                break

        if best_iteration < trained_iterations:
            self.estimator.set_params(max_iter=best_iteration, warm_start=False)
            self._fit_estimator(X, y, sample_weight)
        else:
            self.estimator.set_params(warm_start=False)

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        dtype = np.int64 if self.task == TABULAR_CLASSIFICATION_TASK else np.float32
        return np.asarray(self.estimator.predict(X), dtype=dtype).reshape(-1)

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if self.task != TABULAR_CLASSIFICATION_TASK:
            raise RuntimeError("Regression models do not expose probabilities")
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float32)

    def serialize(self) -> SerializedModelRepr:
        if self._shape is None:
            raise RuntimeError("The model must be fitted before it can be serialized")
        options: dict[int, dict[str, bool]] = {}
        if self.task == TABULAR_CLASSIFICATION_TASK:
            options[id(self.estimator)] = {"zipmap": False}
        model = convert_sklearn(
            self.estimator,
            initial_types=[("model_input", FloatTensorType(self._shape))],
            target_opset={"": ONNX_OPSET_VERSION, "ai.onnx.ml": ML_ONNX_OPSET_VERSION},
            options=options,
        )
        return SerializedModelRepr(
            model,
            len(model.graph.input),
            len(model.graph.output),
            ["FLOAT32"],
            [self._shape],
        )


__all__ = ["SklearnModel"]
