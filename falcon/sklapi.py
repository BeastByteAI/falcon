from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd
from numpy import typing as npt
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import ClassifierMixin as _ClassifierMixin
from sklearn.base import RegressorMixin as _RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from falcon.config import RunConfig
from falcon.predictor import Predictor
from falcon.tabular.splitting import GroupBy, resolve_groups
from falcon.utils import logger


class _FalconBaseEstimator(_BaseEstimator):
    _task: str

    def __init__(
        self,
        preset: str | RunConfig = "balanced",
        eval_strategy: str | Callable[..., Any] | BaseCrossValidator | None = "dynamic",
    ) -> None:
        self.preset = preset
        self.eval_strategy = eval_strategy

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        return tags

    def _get_tags(self) -> dict[str, Any]:
        tags = super()._get_tags()
        if "string" not in tags["X_types"]:
            tags["X_types"].append("string")
        return tags

    def _resolve_fit_groups(
        self,
        X: pd.DataFrame | npt.NDArray[Any],
        group_by: GroupBy | None,
    ) -> GroupBy | None:
        if group_by is None or not isinstance(X, pd.DataFrame):
            return group_by
        return resolve_groups(
            X.to_numpy(dtype=object),
            tuple(str(column) for column in X.columns),
            group_by,
        )

    def _new_predictor(self) -> Predictor:
        eval_strategy = (
            "auto" if self.eval_strategy == "dynamic" else self.eval_strategy
        )
        if isinstance(self.preset, RunConfig):
            return Predictor(
                task=self._task,
                config=self.preset,
                eval_strategy=eval_strategy,
            )
        return Predictor(
            task=self._task,
            preset=self.preset,
            eval_strategy=eval_strategy,
        )

    def _fit_predictor(
        self,
        X: pd.DataFrame | npt.NDArray[Any],
        y: pd.DataFrame | npt.NDArray[Any],
        group_by: GroupBy | None,
    ) -> None:
        resolved_groups = self._resolve_fit_groups(X, group_by)
        checked_X, checked_y = check_X_y(X, y, dtype=None)
        self.n_features_in_ = checked_X.shape[1]
        self.predictor_ = self._new_predictor()
        self.predictor_.fit((checked_X, checked_y), group_by=resolved_groups)

    def predict(self, X: pd.DataFrame | npt.NDArray[Any]) -> npt.NDArray[Any]:
        check_is_fitted(self, "predictor_")
        checked_X = check_array(X, dtype=None)
        return self.predictor_.predict(checked_X)

    def save_model(self, filename: str | None = None) -> None:
        check_is_fitted(self, "predictor_")
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")
            filename = f"falcon_{timestamp}.fnnx"
        elif not filename.endswith(".fnnx"):
            filename = f"{filename}.fnnx"
        self.predictor_.save(filename)
        logger.info("The model was saved as `%s`", filename)


class FalconTabularClassifier(_ClassifierMixin, _FalconBaseEstimator):
    _task = "tabular_classification"

    def fit(
        self,
        X: pd.DataFrame | npt.NDArray[Any],
        y: pd.DataFrame | npt.NDArray[Any],
        group_by: GroupBy | None = None,
    ) -> FalconTabularClassifier:
        self.classes_ = unique_labels(y)
        self._fit_predictor(X, y, group_by)
        return self

    def predict_proba(
        self,
        X: pd.DataFrame | npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        check_is_fitted(self, "predictor_")
        checked_X = check_array(X, dtype=None)
        return self.predictor_.predict_proba(checked_X)


class FalconTabularRegressor(_RegressorMixin, _FalconBaseEstimator):
    _task = "tabular_regression"

    def fit(
        self,
        X: pd.DataFrame | npt.NDArray[Any],
        y: pd.DataFrame | npt.NDArray[Any],
        group_by: GroupBy | None = None,
    ) -> FalconTabularRegressor:
        self._fit_predictor(X, y, group_by)
        return self


FalconClassifier = FalconTabularClassifier
FalconRegressor = FalconTabularRegressor

__all__ = [
    "FalconClassifier",
    "FalconRegressor",
    "FalconTabularClassifier",
    "FalconTabularRegressor",
]
