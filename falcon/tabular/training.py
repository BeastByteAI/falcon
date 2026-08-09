from __future__ import annotations

from collections.abc import Sequence
from time import monotonic
from typing import Any

import numpy as np
from numpy import typing as npt

from falcon.config import RunConfig
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.serialization import SerializedModelRepr
from falcon.tabular.calibration import (
    fit_temperature,
    serialize_temperature_scaling,
    temperature_scale_probabilities,
)
from falcon.tabular.candidates import (
    CandidateModel,
    CandidateTrainer,
    EnsembleRun,
    EstimatorSpec,
    GreedyWeightedEnsemble,
    OOFEnsembleTrainer,
    score_oof_predictions,
)
from falcon.tabular.conformal import (
    fit_conformal_quantile,
    serialize_conformal_interval,
)
from falcon.tabular.decision import fit_decision_weights, serialize_decision_rule
from falcon.types import DatasetSchema, Float32Array, Int64Array

SplitIndices = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]


class CandidateLearner:
    def __init__(
        self,
        task: str,
        dataset_size: tuple[int, ...],
        config: RunConfig,
        evaluation_splits: Sequence[SplitIndices] | None = None,
    ) -> None:
        self.task = task
        self.dataset_size = dataset_size
        self.config = config
        self.evaluation_splits = (
            None if evaluation_splits is None else tuple(evaluation_splits)
        )
        self.model: CandidateModel | GreedyWeightedEnsemble | None = None
        self._ensemble_run: EnsembleRun | None = None
        self._evaluation_run: EnsembleRun | None = None
        self._leaderboard: list[dict[str, str | float]] = []
        self.temperature_: float | None = None
        self.conformal_quantile_: float | None = None
        self.decision_weights_: tuple[float, ...] | None = None

    def _decision_metric(self) -> str | None:
        if self.task != TABULAR_CLASSIFICATION_TASK:
            return None
        return self.config.decision_metric

    def _candidate_specs(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        groups: npt.ArrayLike | None,
        schema: DatasetSchema,
    ) -> tuple[tuple[EstimatorSpec, ...], float]:
        n_classes = (
            int(np.unique(y).size) if self.task == TABULAR_CLASSIFICATION_TASK else None
        )
        started_at = monotonic()
        specs: list[EstimatorSpec] = []
        for source in self.config.candidate_sources:
            remaining_time = self._remaining_time_limit(monotonic() - started_at)
            specs.extend(
                source.get_candidates(
                    self.task,
                    X=X,
                    y=y,
                    groups=groups,
                    n_classes=n_classes,
                    n_splits=self.config.oof_folds,
                    time_limit=remaining_time,
                    random_state=self.config.random_state,
                    schema=schema,
                    dataset_aware_ordering=self.config.dataset_aware_ordering,
                    config=self.config,
                )
            )
        if not specs:
            raise ValueError("Candidate sources produced no estimator specifications")
        return tuple(specs), monotonic() - started_at

    def _remaining_time_limit(self, elapsed: float) -> float | None:
        if self.config.time_limit is None:
            return None
        return max(float(np.finfo(float).eps), self.config.time_limit - elapsed)

    def _ensemble_trainer(
        self,
        *,
        time_limit: float | None,
        max_iterations: int | None = None,
    ) -> OOFEnsembleTrainer:
        return OOFEnsembleTrainer(
            self.task,
            max_iterations=(
                self.config.ensemble_max_iterations
                if max_iterations is None
                else max_iterations
            ),
            plateau_enabled=self.config.plateau_enabled,
            plateau_patience=self.config.plateau_patience,
            plateau_tolerance=self.config.plateau_tolerance,
            n_splits=self.config.oof_folds,
            time_limit=time_limit,
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,
            prior_correct=self.config.prior_correct,
        )

    def _fit_ensemble(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        specs: tuple[EstimatorSpec, ...],
        groups: npt.ArrayLike | None,
        time_limit: float | None,
    ) -> None:
        run = self._ensemble_trainer(time_limit=time_limit).fit(
            X,
            y,
            specs=specs,
            groups=groups,
            splits=self.evaluation_splits,
        )
        self._ensemble_run = run
        self.model = run.ensemble
        self._leaderboard = self._run_leaderboard(run)

    def _run_leaderboard(self, run: EnsembleRun) -> list[dict[str, str | float]]:
        return [
            {
                "candidate": candidate.spec.name,
                "family": candidate.spec.family,
                "score": candidate.oof_score,
                "fit_time": candidate.fit_time,
                "weight": weight,
            }
            for candidate, weight in zip(
                run.candidates, run.ensemble.weights, strict=True
            )
        ]

    def _fit_best_candidate(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        specs: tuple[EstimatorSpec, ...],
        groups: npt.ArrayLike | None,
        time_limit: float | None,
    ) -> None:
        winner_spec = specs[0]
        refit_time_limit = time_limit
        needs_oof = (
            len(specs) > 1
            or self.evaluation_splits is not None
            or self.config.calibrate
            or self.config.conformal_alpha is not None
            or self._decision_metric() is not None
        )
        if needs_oof:
            # max_iterations=1 keeps greedy selection at its seed, so the ensemble
            # weights are 1.0 on the best-scoring candidate and 0.0 elsewhere.
            selection_run = self._ensemble_trainer(
                time_limit=time_limit,
                max_iterations=1,
            ).fit(X, y, specs=specs, groups=groups, splits=self.evaluation_splits)
            self._evaluation_run = selection_run
            winner_spec = selection_run.candidates[
                int(np.argmax(selection_run.ensemble.weights))
            ].spec
            if time_limit is not None:
                refit_time_limit = max(
                    float(np.finfo(float).eps),
                    time_limit - selection_run.elapsed_time,
                )

        refit_run = CandidateTrainer(
            self.task,
            time_limit=refit_time_limit,
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,
        ).fit(X, y, specs=(winner_spec,), groups=groups)
        winner = refit_run.candidates[0]
        self.model = winner.model
        if self._evaluation_run is None:
            if self.task == TABULAR_CLASSIFICATION_TASK:
                predict_proba = getattr(winner.model, "predict_proba", None)
                if not callable(predict_proba):
                    raise TypeError(
                        "Classification candidate models must expose predict_proba"
                    )
                predictions = predict_proba(X)
            else:
                predictions = winner.model.predict(X)
            self._leaderboard = [
                {
                    "candidate": winner.spec.name,
                    "family": winner.spec.family,
                    "score": score_oof_predictions(
                        predictions,
                        y,
                        self.task,
                        prior_correct=self.config.prior_correct,
                    ),
                    "fit_time": winner.fit_time,
                    "weight": 1.0,
                }
            ]
        else:
            self._leaderboard = self._run_leaderboard(self._evaluation_run)

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None:
        if self.config.calibrate and self.task != TABULAR_CLASSIFICATION_TASK:
            raise ValueError(
                "Probability calibration is only available for classification"
            )
        if (
            self.config.conformal_alpha is not None
            and self.task != TABULAR_REGRESSION_TASK
        ):
            raise ValueError(
                "Conformal prediction intervals are only available for regression"
            )
        if (
            self.config.decision_metric is not None
            and self.task != TABULAR_CLASSIFICATION_TASK
            and "decision_metric" in self.config._provided_fields
        ):
            raise ValueError(
                "A tuned decision rule is only available for classification"
            )
        specs, source_elapsed = self._candidate_specs(X, y, groups, schema)
        training_time_limit = self._remaining_time_limit(source_elapsed)
        if self.config.ensemble_enabled:
            self._fit_ensemble(X, y, specs, groups, training_time_limit)
        else:
            self._fit_best_candidate(X, y, specs, groups, training_time_limit)
        if self.config.calibrate:
            oof_result = self._weighted_oof_predictions()
            if oof_result is None:
                raise RuntimeError("OOF probabilities are unavailable for calibration")
            evaluation_indices, probabilities = oof_result
            self.temperature_ = fit_temperature(
                probabilities,
                np.asarray(y)[evaluation_indices],
            )
        decision_metric = self._decision_metric()
        if decision_metric is not None:
            oof_result = self._weighted_oof_predictions()
            if oof_result is None:
                raise RuntimeError(
                    "OOF probabilities are unavailable for the decision rule"
                )
            evaluation_indices, probabilities = oof_result
            self._assert_classes_are_encoded()
            # The rule is tuned on calibrated probabilities because the graph applies
            # it downstream of the temperature Softmax, and in multiclass a weighted
            # argmax is not invariant to temperature.
            if self.temperature_ is not None:
                probabilities = temperature_scale_probabilities(
                    probabilities,
                    self.temperature_,
                )
            weights = fit_decision_weights(
                probabilities,
                np.asarray(y)[evaluation_indices],
                decision_metric,
            )
            # An all-ones vector is the guard's no-op result. Keeping it as "no rule"
            # avoids inert graph nodes and leaves prediction on the model's own argmax.
            if any(weight != 1.0 for weight in weights):
                self.decision_weights_ = weights
        if self.config.conformal_alpha is not None:
            oof_result = self._weighted_oof_predictions()
            if oof_result is None:
                raise RuntimeError(
                    "OOF predictions are unavailable for conformal intervals"
                )
            evaluation_indices, predictions = oof_result
            self.conformal_quantile_ = fit_conformal_quantile(
                predictions,
                np.asarray(y)[evaluation_indices],
                self.config.conformal_alpha,
            )

    def _fitted_model(self) -> CandidateModel | GreedyWeightedEnsemble:
        if self.model is None:
            raise RuntimeError("The candidate learner has not been fitted")
        return self.model

    def _assert_classes_are_encoded(self) -> None:
        classes = getattr(self._fitted_model(), "classes", None)
        if classes is None:
            return
        expected = np.arange(len(classes), dtype=np.int64)
        if not np.array_equal(np.asarray(classes), expected):
            raise RuntimeError(
                "The tuned decision rule requires contiguous encoded class labels"
            )

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.decision_weights_ is None:
            return self._fitted_model().predict(X)
        # predict_proba applies the temperature; predict does not, and a weighted
        # argmax is not temperature-invariant, so the rule must read the scaled scores.
        weights = np.asarray(self.decision_weights_, dtype=np.float32)
        return np.argmax(self.predict_proba(X) * weights, axis=1).astype(np.int64)

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if self.task != TABULAR_CLASSIFICATION_TASK:
            raise RuntimeError("Regression predictors do not expose probabilities")
        predict_proba = getattr(self._fitted_model(), "predict_proba", None)
        if not callable(predict_proba):
            raise RuntimeError("The fitted classification model has no probabilities")
        probabilities = np.asarray(predict_proba(X), dtype=np.float32)
        if self.temperature_ is None:
            return probabilities
        return temperature_scale_probabilities(probabilities, self.temperature_)

    def _weighted_oof_predictions(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]] | None:
        run = self._ensemble_run or self._evaluation_run
        if run is None:
            return None
        weighted = np.asarray(
            np.sum(
                np.stack(
                    [
                        candidate.oof_predictions * np.float32(weight)
                        for candidate, weight in zip(
                            run.candidates, run.ensemble.weights, strict=True
                        )
                    ],
                    axis=0,
                ),
                axis=0,
                dtype=np.float32,
            )
        )
        return run.evaluation_indices.copy(), weighted

    def oof_predictions(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[Any]] | None:
        oof_result = self._weighted_oof_predictions()
        if oof_result is None:
            return None
        evaluation_indices, weighted = oof_result
        if self.task == TABULAR_CLASSIFICATION_TASK:
            run = self._ensemble_run or self._evaluation_run
            if run is None:
                raise RuntimeError("Classification OOF run is unavailable")
            if run.ensemble.classes is None:
                raise RuntimeError("Classification classes are unavailable")
            scores = weighted
            if self.temperature_ is not None:
                scores = temperature_scale_probabilities(scores, self.temperature_)
            if self.decision_weights_ is not None:
                scores = scores * np.asarray(self.decision_weights_, dtype=np.float32)
            predictions: npt.NDArray[Any] = run.ensemble.classes[
                np.argmax(scores, axis=1)
            ]
        else:
            predictions = weighted.reshape(-1)
        return evaluation_indices, predictions

    def leaderboard_records(self) -> list[dict[str, str | float]]:
        return [record.copy() for record in self._leaderboard]

    def serialize(self) -> SerializedModelRepr:
        serialized = self._fitted_model().serialize()
        if self.temperature_ is not None:
            serialized = serialize_temperature_scaling(serialized, self.temperature_)
        if self.decision_weights_ is not None:
            serialized = serialize_decision_rule(serialized, self.decision_weights_)
        if self.conformal_quantile_ is not None:
            serialized = serialize_conformal_interval(
                serialized,
                self.conformal_quantile_,
            )
        return serialized

    def get_input_type(self) -> object:
        return Float32Array

    def get_output_type(self) -> object:
        return Int64Array if self.task == TABULAR_CLASSIFICATION_TASK else Float32Array


__all__ = ["CandidateLearner", "SplitIndices"]
