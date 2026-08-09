from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from numpy.random import default_rng
from sklearn import metrics
from sklearn.model_selection import BaseCrossValidator

from falcon import types as ft
from falcon.config import EvalStrategy, RunConfig
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.presets import resolve_run_config
from falcon.tabular.evaluation import (
    classification_metrics,
    regression_metrics,
)
from falcon.tabular.ingestion import (
    TabularData,
    ingest_data,
    ingest_data_with_row_selection,
    read_data,
)
from falcon.tabular.pipelines.simple_tabular_pipeline import SimpleTabularPipeline
from falcon.tabular.splitting import (
    GroupBy,
    callable_holdout_indices,
    cross_validation_indices,
    holdout_indices,
    resolve_evaluation_strategy,
    resolve_groups,
)
from falcon.tabular.training import CandidateLearner, SplitIndices
from falcon.types import DatasetSchema


class _Unspecified:
    pass


UNSPECIFIED = _Unspecified()

_REMOVED_KWARGS = {
    "manager_configuration": (
        "`manager_configuration` was removed in 0.9 — use `preset` or "
        "`config=RunConfig(...)`"
    ),
    "pipeline": "`pipeline` was removed in 0.9 — use `preset` or `RunConfig`",
    "pipeline_options": (
        "`pipeline_options` was removed in 0.9 — use `preset` or `RunConfig`"
    ),
    "extra_pipeline_options": (
        "`extra_pipeline_options` was removed in 0.9 — use `preset` or `RunConfig`"
    ),
}


@dataclass(frozen=True)
class _TrainingData:
    X: npt.NDArray[np.object_]
    y: npt.NDArray[np.object_]
    schema: DatasetSchema
    groups: npt.NDArray[np.int64]


class Predictor:
    def __init__(
        self,
        task: str,
        preset: str = "balanced",
        config: RunConfig | None = None,
        time_limit: float | None | _Unspecified = UNSPECIFIED,
        random_state: int | _Unspecified = UNSPECIFIED,
        eval_strategy: EvalStrategy | _Unspecified = UNSPECIFIED,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            name = next(iter(kwargs))
            if name in _REMOVED_KWARGS:
                raise TypeError(_REMOVED_KWARGS[name])
            raise TypeError(f"Predictor() got an unexpected keyword argument `{name}`")
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        if not isinstance(preset, str):
            raise TypeError("preset must be a string")
        if config is not None and not isinstance(config, RunConfig):
            if isinstance(config, dict):
                raise TypeError(
                    "Dictionary configs were removed in 0.9 — use "
                    "`config=RunConfig(...)`"
                )
            raise TypeError("config must be a RunConfig")
        normalized_task = task.lower()
        if normalized_task not in {
            TABULAR_CLASSIFICATION_TASK,
            TABULAR_REGRESSION_TASK,
        }:
            raise ValueError(f"Unknown task `{task}`")
        overrides: dict[str, Any] = {"config": config}
        if not isinstance(time_limit, _Unspecified):
            overrides["time_limit"] = time_limit
        if not isinstance(random_state, _Unspecified):
            overrides["random_state"] = random_state
        if not isinstance(eval_strategy, _Unspecified):
            overrides["eval_strategy"] = eval_strategy

        self.task = normalized_task
        self.preset = preset
        self.config = resolve_run_config(normalized_task, preset, **overrides)
        if self.config.calibrate and normalized_task != TABULAR_CLASSIFICATION_TASK:
            raise ValueError(
                "Probability calibration is only available for classification"
            )
        if (
            self.config.conformal_alpha is not None
            and normalized_task != TABULAR_REGRESSION_TASK
        ):
            raise ValueError(
                "Conformal prediction intervals are only available for regression"
            )
        self._pipeline: SimpleTabularPipeline | None = None
        self._learner: CandidateLearner | None = None
        self._training_data: _TrainingData | None = None
        self._fit_indices: npt.NDArray[np.int64] | None = None
        self._eval_indices: npt.NDArray[np.int64] | None = None
        self._performance_metrics: dict[str, dict[str, Any]] = {}
        self._features: ft.ColumnsList | None = None
        self._target: str | int | None = None
        self.classes_: npt.NDArray[Any] | None = None

    def _evaluation_split(
        self,
        data: _TrainingData,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64] | None,
        tuple[SplitIndices, ...] | None,
    ]:
        strategy = self.config.eval_strategy
        if strategy is None:
            return np.arange(len(data.X), dtype=np.int64), None, None
        if strategy == "auto":
            strategy = resolve_evaluation_strategy(len(data.X))
        if strategy == "holdout":
            train_indices, eval_indices = holdout_indices(
                data.X,
                data.y,
                self.task,
                data.groups,
                random_state=self.config.random_state,
            )
            return train_indices, eval_indices, None
        if callable(strategy) and not isinstance(strategy, BaseCrossValidator):
            train_indices, eval_indices = callable_holdout_indices(
                strategy,
                data.X,
                data.y,
                data.groups,
            )
            return train_indices, eval_indices, None
        if strategy == "cv" or isinstance(strategy, BaseCrossValidator):
            cv = strategy if isinstance(strategy, BaseCrossValidator) else None
            splits = cross_validation_indices(
                data.X,
                data.y,
                self.task,
                data.groups,
                cv=cv,
                n_splits=self.config.oof_folds,
                random_state=self.config.random_state,
            )
            return (
                np.arange(len(data.X), dtype=np.int64),
                None,
                tuple(splits),
            )
        raise RuntimeError("The resolved evaluation strategy is invalid")

    def fit(
        self,
        data: TabularData,
        features: ft.ColumnsList | None = None,
        target: str | int | None = None,
        group_by: GroupBy | None = None,
    ) -> Predictor:
        X, y, schema, row_indices, source_row_count = ingest_data_with_row_selection(
            data,
            task=self.task,
            features=features,
            target=target,
        )
        groups = resolve_groups(
            X,
            schema.column_names,
            group_by,
            source_row_indices=row_indices,
            source_row_count=source_row_count,
        )
        training_data = _TrainingData(X, y, schema, groups)
        fit_indices, eval_indices, evaluation_splits = self._evaluation_split(
            training_data
        )
        fit_schema = replace(
            schema,
            dimensions=(len(fit_indices), schema.n_features),
        )
        pipeline = SimpleTabularPipeline(
            task=self.task,
            dataset_size=fit_schema.dimensions,
            schema=fit_schema,
            learner=CandidateLearner,
            impute_missing=self.config.impute_missing,
            learner_kwargs={
                "config": self.config,
                "evaluation_splits": evaluation_splits,
            },
        )
        pipeline.fit(
            X[fit_indices],
            y[fit_indices],
            fit_schema,
            groups=groups[fit_indices],
        )
        learner = pipeline.steps[1]
        if not isinstance(learner, CandidateLearner):
            raise RuntimeError("The candidate learner was not assembled correctly")

        self._pipeline = pipeline
        self._learner = learner
        self._training_data = training_data
        self._fit_indices = fit_indices
        self._eval_indices = eval_indices
        self._features = features
        self._target = target
        if pipeline.labels_transformer is not None:
            self.classes_ = pipeline.labels_transformer.le.classes_.copy()
        else:
            self.classes_ = None
        self._collect_fit_metrics(evaluation_splits is not None)
        return self

    def _require_fitted(
        self,
    ) -> tuple[SimpleTabularPipeline, CandidateLearner, _TrainingData]:
        if (
            self._pipeline is None
            or self._learner is None
            or self._training_data is None
        ):
            raise RuntimeError("Predictor.fit must be called before this operation")
        return self._pipeline, self._learner, self._training_data

    def _prediction_array(self, data: Any) -> npt.NDArray[np.object_]:
        _, _, training_data = self._require_fitted()
        if isinstance(data, (str, os.PathLike)):
            data = read_data(os.fspath(data))
        if isinstance(data, pd.DataFrame):
            missing = [
                name
                for name in training_data.schema.column_names
                if name not in data.columns
            ]
            if missing:
                raise ValueError(f"Prediction data is missing feature `{missing[0]}`")
            values = data.loc[:, list(training_data.schema.column_names)].to_numpy(
                dtype=np.object_
            )
        else:
            values = np.asarray(data, dtype=np.object_)
        if values.ndim != 2:
            raise ValueError("Prediction data must be two-dimensional")
        if values.shape[1] != training_data.schema.n_features:
            raise ValueError(
                "Prediction data must contain exactly "
                f"{training_data.schema.n_features} features"
            )
        return values

    def predict(self, data: Any) -> npt.NDArray[Any]:
        pipeline, _, _ = self._require_fitted()
        return np.asarray(pipeline.predict(self._prediction_array(data))).reshape(-1)

    def predict_proba(self, data: Any) -> npt.NDArray[np.float32]:
        if self.task != TABULAR_CLASSIFICATION_TASK:
            raise RuntimeError("predict_proba is only available for classification")
        pipeline, learner, _ = self._require_fitted()
        encoded = pipeline.steps[0].transform(self._prediction_array(data))
        return learner.predict_proba(encoded)

    def _report(
        self,
        y: npt.NDArray[Any],
        predictions: npt.NDArray[Any],
    ) -> dict[str, Any]:
        if self.task == TABULAR_CLASSIFICATION_TASK:
            return classification_metrics(y, predictions)
        return regression_metrics(y, predictions)

    def _collect_fit_metrics(self, has_cv_evaluation: bool) -> None:
        pipeline, learner, training_data = self._require_fitted()
        if self._fit_indices is None:
            raise RuntimeError("Training indices are unavailable")
        fit_X = training_data.X[self._fit_indices]
        fit_y = training_data.y[self._fit_indices]
        performance = {"train": self._report(fit_y, pipeline.predict(fit_X))}
        if self._eval_indices is not None:
            eval_X = training_data.X[self._eval_indices]
            eval_y = training_data.y[self._eval_indices]
            performance["eval"] = self._report(eval_y, pipeline.predict(eval_X))
        elif has_cv_evaluation:
            oof_result = learner.oof_predictions()
            if oof_result is None:
                raise RuntimeError("Cross-validation predictions are unavailable")
            evaluation_indices, predictions = oof_result
            if pipeline.labels_transformer is not None:
                predictions = pipeline.labels_transformer.transform(predictions)
            performance["eval_cv"] = self._report(
                fit_y[evaluation_indices], predictions
            )
        self._performance_metrics = performance

    def _evaluation_data(
        self,
        data: TabularData,
    ) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.object_]]:
        _, _, training_data = self._require_fitted()
        if isinstance(data, (str, os.PathLike)):
            data = read_data(os.fspath(data))
        features: ft.ColumnsList | None
        target: str | int | None
        if isinstance(data, tuple):
            features = None
            target = None
        elif isinstance(data, pd.DataFrame):
            features = list(training_data.schema.column_names)
            target = training_data.schema.target_name
        else:
            features = self._features
            target = self._target
        X, y, _ = ingest_data(
            data,
            task=self.task,
            features=features,
            target=target,
        )
        return X, y

    def evaluate(self, test_data: TabularData) -> dict[str, Any]:
        X, y = self._evaluation_data(test_data)
        result = self._report(y, self.predict(X))
        self._performance_metrics["test"] = result
        return result

    def leaderboard(self) -> pd.DataFrame:
        _, learner, _ = self._require_fitted()
        columns = ["candidate", "family", "score", "fit_time", "weight"]
        return pd.DataFrame(learner.leaderboard_records(), columns=columns)

    def feature_importance(
        self,
        n_repeats: int = 10,
    ) -> list[dict[str, str | float]]:
        if n_repeats < 1:
            raise ValueError("n_repeats must be at least 1")
        _, _, training_data = self._require_fitted()
        if self._fit_indices is None:
            raise RuntimeError("Training indices are unavailable")
        X = training_data.X[self._fit_indices]
        y = training_data.y[self._fit_indices]
        scoring: Callable[[npt.ArrayLike, npt.ArrayLike], float]
        if self.task == TABULAR_CLASSIFICATION_TASK:
            scoring = metrics.balanced_accuracy_score
            y = y.astype(np.str_)
        else:
            scoring = metrics.r2_score
        baseline = scoring(y, self.predict(X))
        rng = default_rng(self.config.random_state)
        importances = np.zeros((n_repeats, X.shape[1]), dtype=np.float64)
        for feature_index in range(X.shape[1]):
            for repeat_index in range(n_repeats):
                permuted = X.copy()
                order = rng.permutation(len(X))
                permuted[:, feature_index] = permuted[order, feature_index]
                importances[repeat_index, feature_index] = baseline - scoring(
                    y, self.predict(permuted)
                )
        means = np.mean(importances, axis=0)
        standard_deviations = np.std(importances, axis=0)
        denominator = float(np.abs(means.sum()))
        scaled = means / denominator if denominator else np.zeros_like(means)
        result: list[dict[str, str | float]] = [
            {
                "feature_name": feature_name,
                "importance": float(mean),
                "std": float(standard_deviation),
                "scaled_importance": float(scaled_importance),
            }
            for feature_name, mean, standard_deviation, scaled_importance in zip(
                training_data.schema.column_names,
                means,
                standard_deviations,
                scaled,
                strict=True,
            )
        ]
        return sorted(
            result,
            key=lambda item: float(item["importance"]),
            reverse=True,
        )

    def save(self, path: str | os.PathLike[str] | None = None) -> bytes:
        pipeline, _, training_data = self._require_fitted()
        serializer = pipeline.save(
            feature_names=list(training_data.schema.column_names),
            schema=training_data.schema,
        )
        if self._performance_metrics:
            serializer.metadata_payload["metrics"] = {
                "performance": self._performance_metrics
            }
        serialized = serializer.serialize()
        if path is not None:
            with open(path, "wb") as model_file:
                model_file.write(serialized)
        return serialized

    def _performance_summary(
        self,
        test_data: TabularData | None = None,
    ) -> dict[str, dict[str, Any]]:
        self._require_fitted()
        if test_data is not None:
            self.evaluate(test_data)
        print(
            "\n",
            pd.DataFrame.from_dict(self._performance_metrics, orient="index"),
            "\n",
        )
        return {
            name: values.copy() for name, values in self._performance_metrics.items()
        }


__all__ = ["Predictor"]
