from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import monotonic
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy import typing as npt
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import has_fit_parameter

from falcon.config import ClassWeight
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.serialization import SerializedModelRepr, serialize_parallel_ensemble
from falcon.tabular.models.gbdt import GBDTModel, get_gbdt_model_classes
from falcon.tabular.models.sklearn_model import SklearnModel
from falcon.tabular.splitting import holdout_indices, out_of_fold_indices
from falcon.types import Float32Array, Int64Array
from falcon.utils import logger

_DEFAULT_RESERVE_FRACTION = 0.2
_EARLY_STOPPING_TEST_SIZE = 0.2
_SKLEARN_FAMILIES = {
    "extra_trees",
    "hist_gradient_boosting",
    "linear",
    "random_forest",
}


class CandidateModel(Protocol):
    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None: ...

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]: ...

    def serialize(self) -> SerializedModelRepr: ...


class CandidateClassifierModel(CandidateModel, Protocol):
    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]: ...


CandidateModelFactory = Callable[
    ["EstimatorSpec", str, int, int | None], CandidateModel
]


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    family: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    min_rows: int = 1
    max_rows: int | None = None
    max_features: int | None = None
    early_stopping_rounds: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Estimator name must not be empty")
        if not self.family:
            raise ValueError("Estimator family must not be empty")
        if self.min_rows < 1:
            raise ValueError("min_rows must be at least 1")
        if self.max_rows is not None and self.max_rows < self.min_rows:
            raise ValueError("max_rows must be greater than or equal to min_rows")
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features must be at least 1")
        if self.early_stopping_rounds is not None and self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be at least 1")
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )

    def is_applicable(self, *, n_rows: int, n_features: int) -> bool:
        if n_rows < self.min_rows:
            return False
        if self.max_rows is not None and n_rows > self.max_rows:
            return False
        return self.max_features is None or n_features <= self.max_features


@dataclass(frozen=True)
class TrainedCandidate:
    spec: EstimatorSpec
    model: CandidateModel
    fit_time: float


@dataclass(frozen=True)
class CandidateRun:
    candidates: tuple[TrainedCandidate, ...]
    elapsed_time: float
    stopped_for_budget: bool


def _sklearn_portfolio(task: str) -> list[EstimatorSpec]:
    tree_criterion = "gini" if task == TABULAR_CLASSIFICATION_TASK else "squared_error"
    linear_parameters: dict[str, object]
    if task == TABULAR_CLASSIFICATION_TASK:
        linear_parameters = {"C": 1.0, "max_iter": 1_000}
    else:
        linear_parameters = {"alpha": 1.0}
    return [
        EstimatorSpec(
            "hist_gradient_boosting_default",
            "hist_gradient_boosting",
            {
                "learning_rate": 0.08,
                "l2_regularization": 0.1,
                "max_iter": 200,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
            },
            early_stopping_rounds=20,
        ),
        EstimatorSpec(
            "extra_trees_zeroshot",
            "extra_trees",
            {
                "criterion": tree_criterion,
                "max_features": 0.75,
                "min_samples_leaf": 1,
                "n_estimators": 300,
            },
        ),
        EstimatorSpec("linear_default", "linear", linear_parameters),
        EstimatorSpec(
            "random_forest_zeroshot",
            "random_forest",
            {
                "criterion": tree_criterion,
                "max_features": 0.75,
                "min_samples_leaf": 1,
                "n_estimators": 300,
            },
        ),
        EstimatorSpec(
            "hist_gradient_boosting_large",
            "hist_gradient_boosting",
            {
                "learning_rate": 0.04,
                "l2_regularization": 0.1,
                "max_iter": 300,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 10,
            },
            early_stopping_rounds=20,
        ),
    ]


def _gbdt_portfolio(available_families: set[str]) -> list[EstimatorSpec]:
    primary: dict[str, EstimatorSpec] = {
        "lightgbm": EstimatorSpec(
            "lightgbm_default",
            "lightgbm",
            {"n_estimators": 300},
            early_stopping_rounds=20,
        ),
        "xgboost": EstimatorSpec(
            "xgboost_default",
            "xgboost",
            {"n_estimators": 300},
            early_stopping_rounds=20,
        ),
        "catboost": EstimatorSpec(
            "catboost_default",
            "catboost",
            {"iterations": 300},
            early_stopping_rounds=20,
        ),
    }
    zeroshot: dict[str, EstimatorSpec] = {
        "lightgbm": EstimatorSpec(
            "lightgbm_zeroshot_large",
            "lightgbm",
            {
                "colsample_bytree": 0.9,
                "learning_rate": 0.03,
                "min_child_samples": 3,
                "n_estimators": 500,
                "num_leaves": 128,
            },
            early_stopping_rounds=20,
        ),
        "xgboost": EstimatorSpec(
            "xgboost_zeroshot_r33",
            "xgboost",
            {
                "colsample_bytree": 0.6917311125174739,
                "learning_rate": 0.018063876087523967,
                "max_depth": 10,
                "min_child_weight": 0.6028633586934382,
                "n_estimators": 500,
            },
            early_stopping_rounds=20,
        ),
        "catboost": EstimatorSpec(
            "catboost_zeroshot_r177",
            "catboost",
            {
                "depth": 6,
                "grow_policy": "SymmetricTree",
                "iterations": 500,
                "l2_leaf_reg": 2.1542798306067823,
                "learning_rate": 0.06864209415792857,
            },
            early_stopping_rounds=20,
        ),
    }
    family_order = ("lightgbm", "xgboost", "catboost")
    return [
        portfolio[family]
        for portfolio in (primary, zeroshot)
        for family in family_order
        if family in available_families
    ]


def _interleave_portfolios(
    gbdt_specs: Sequence[EstimatorSpec],
    sklearn_specs: Sequence[EstimatorSpec],
) -> tuple[EstimatorSpec, ...]:
    ordered: list[EstimatorSpec] = []
    for index in range(max(len(gbdt_specs), len(sklearn_specs))):
        if index < len(gbdt_specs):
            ordered.append(gbdt_specs[index])
        if index < len(sklearn_specs):
            ordered.append(sklearn_specs[index])
    return tuple(ordered)


def _portfolio_from_families(
    task: str,
    available_families: set[str],
) -> tuple[EstimatorSpec, ...]:
    sklearn_specs = _sklearn_portfolio(task)
    if not available_families:
        logger.info(
            "Optional GBDT libraries are unavailable; using sklearn-only candidate "
            "portfolio."
        )
        return tuple(sklearn_specs)
    return _interleave_portfolios(
        _gbdt_portfolio(available_families),
        sklearn_specs,
    )


def _validate_task(task: str) -> None:
    if task not in {TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK}:
        raise ValueError(f"Unknown task `{task}`")


def default_portfolio(
    task: str,
    *,
    n_classes: int | None = None,
) -> tuple[EstimatorSpec, ...]:
    _validate_task(task)
    available = set(get_gbdt_model_classes(task, n_classes=n_classes))
    return _portfolio_from_families(task, available)


def _build_sklearn_model(
    spec: EstimatorSpec,
    task: str,
    random_state: int,
) -> SklearnModel:
    parameters = dict(spec.parameters)
    parameters.pop("random_seed", None)
    parameters.pop("random_state", None)
    estimator: Any
    if spec.family == "hist_gradient_boosting":
        model_class = (
            HistGradientBoostingClassifier
            if task == TABULAR_CLASSIFICATION_TASK
            else HistGradientBoostingRegressor
        )
        parameters["early_stopping"] = False
        parameters["random_state"] = random_state
        estimator = model_class(**parameters)
    elif spec.family == "extra_trees":
        model_class = (
            ExtraTreesClassifier
            if task == TABULAR_CLASSIFICATION_TASK
            else ExtraTreesRegressor
        )
        parameters.setdefault("n_jobs", 1)
        parameters["random_state"] = random_state
        estimator = model_class(**parameters)
    elif spec.family == "random_forest":
        model_class = (
            RandomForestClassifier
            if task == TABULAR_CLASSIFICATION_TASK
            else RandomForestRegressor
        )
        parameters.setdefault("n_jobs", 1)
        parameters["random_state"] = random_state
        estimator = model_class(**parameters)
    elif spec.family == "linear" and task == TABULAR_CLASSIFICATION_TASK:
        parameters["random_state"] = random_state
        estimator = LogisticRegression(**parameters)
    elif spec.family == "linear":
        estimator = Ridge(**parameters)
    else:
        raise ValueError(f"Unknown sklearn estimator family `{spec.family}`")
    return SklearnModel(estimator, task)


def _build_candidate_model(
    spec: EstimatorSpec,
    task: str,
    random_state: int,
    available_gbdt: Mapping[str, type[GBDTModel]],
) -> CandidateModel:
    if spec.family in _SKLEARN_FAMILIES:
        return _build_sklearn_model(spec, task, random_state)
    if spec.family not in available_gbdt:
        raise ImportError(
            f"Estimator family `{spec.family}` is unavailable; install falcon-ml[gbdt]"
        )
    parameters = dict(spec.parameters)
    parameters.pop("random_seed", None)
    parameters.pop("random_state", None)
    model_class: Any = available_gbdt[spec.family]
    return model_class(random_state=random_state, **parameters)


def _supports_sample_weight(model: CandidateModel) -> bool:
    """Report whether the underlying estimator accepts `sample_weight` in `fit`.

    Wrappers may declare support explicitly; introspecting the wrapper itself would
    always report support because the `CandidateModel` protocol requires the keyword.
    A model exposing neither a declaration nor an estimator is allowed through.
    """
    declared = getattr(model, "supports_sample_weight", None)
    if isinstance(declared, bool):
        return declared
    estimator = getattr(model, "estimator", None)
    if estimator is None:
        return True
    return bool(has_fit_parameter(estimator, "sample_weight"))


def _validated_training_data(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    groups: npt.ArrayLike | None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any] | None]:
    feature_values = np.asarray(X)
    target_values = np.asarray(y)
    if feature_values.ndim != 2:
        raise ValueError("Features must be two-dimensional")
    if target_values.ndim == 2 and target_values.shape[1] == 1:
        target_values = target_values[:, 0]
    if target_values.ndim != 1:
        raise ValueError("Targets must be one-dimensional")
    if len(feature_values) != len(target_values):
        raise ValueError("Features and targets must contain the same number of rows")
    if len(feature_values) == 0:
        raise ValueError("Training data must not be empty")
    if groups is None:
        return feature_values, target_values, None
    group_values = np.asarray(groups)
    if group_values.ndim == 2 and group_values.shape[1] == 1:
        group_values = group_values[:, 0]
    if group_values.ndim != 1 or len(group_values) != len(feature_values):
        raise ValueError("Groups must contain one value per feature row")
    return feature_values, target_values, group_values


class CandidateTrainer:
    def __init__(
        self,
        task: str,
        *,
        time_limit: float | None = None,
        reserve_fraction: float = _DEFAULT_RESERVE_FRACTION,
        random_state: int = 42,
        class_weight: ClassWeight = "none",
        model_factory: CandidateModelFactory | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        _validate_task(task)
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be greater than zero")
        if not 0 <= reserve_fraction < 1:
            raise ValueError("reserve_fraction must be in the range [0, 1)")
        if class_weight not in {"none", "balanced"}:
            raise ValueError("class_weight must be either 'none' or 'balanced'")
        self.task = task
        self.time_limit = time_limit
        self.reserve_fraction = reserve_fraction
        self.random_state = random_state
        self.class_weight = class_weight
        self.model_factory = model_factory
        self.clock = monotonic if clock is None else clock

    def _candidate_budget(self) -> float | None:
        if self.time_limit is None:
            return None
        return self.time_limit * (1 - self.reserve_fraction)

    def _budget_prevents_next_fit(
        self,
        elapsed: float,
        observed_fit_times: Sequence[float],
    ) -> bool:
        budget = self._candidate_budget()
        if budget is None:
            return False
        remaining = budget - elapsed
        if remaining <= 0:
            return True
        if not observed_fit_times:
            return False
        return float(np.mean(observed_fit_times)) > remaining

    def _fit_inputs(
        self,
        spec: EstimatorSpec,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        groups: npt.NDArray[Any] | None,
        *,
        random_state: int | None = None,
    ) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[np.float64] | None,
        tuple[npt.NDArray[Any], npt.NDArray[Any]] | None,
        int | None,
    ]:
        train_X = X
        train_y = y
        validation_data = None
        early_stopping_rounds = None
        if spec.early_stopping_rounds is not None:
            split_random_state = (
                self.random_state if random_state is None else random_state
            )
            try:
                train_indices, validation_indices = holdout_indices(
                    X,
                    y,
                    self.task,
                    groups,
                    test_size=_EARLY_STOPPING_TEST_SIZE,
                    random_state=split_random_state,
                )
            except ValueError as error:
                logger.info(
                    "Candidate %s will train without early stopping: %s",
                    spec.name,
                    error,
                )
            else:
                train_X = X[train_indices]
                train_y = y[train_indices]
                validation_data = (X[validation_indices], y[validation_indices])
                early_stopping_rounds = spec.early_stopping_rounds
        sample_weight = None
        if self.task == TABULAR_CLASSIFICATION_TASK and self.class_weight == "balanced":
            sample_weight = np.asarray(
                compute_sample_weight(class_weight="balanced", y=train_y),
                dtype=np.float64,
            )
        return (
            train_X,
            train_y,
            sample_weight,
            validation_data,
            early_stopping_rounds,
        )

    def _available_gbdt(
        self,
        specs: Sequence[EstimatorSpec] | None,
        n_classes: int | None,
    ) -> dict[str, type[GBDTModel]]:
        if self.model_factory is not None:
            return {}
        if specs is not None and all(
            spec.family in _SKLEARN_FAMILIES for spec in specs
        ):
            return {}
        return get_gbdt_model_classes(self.task, n_classes=n_classes)

    def _model(
        self,
        spec: EstimatorSpec,
        n_classes: int | None,
        available_gbdt: Mapping[str, type[GBDTModel]],
        *,
        random_state: int | None = None,
    ) -> CandidateModel:
        model_random_state = self.random_state if random_state is None else random_state
        if self.model_factory is not None:
            return self.model_factory(
                spec,
                self.task,
                model_random_state,
                n_classes,
            )
        return _build_candidate_model(
            spec,
            self.task,
            model_random_state,
            available_gbdt,
        )

    def _assert_sample_weight_support(
        self,
        specs: Sequence[EstimatorSpec],
        n_classes: int | None,
        available_gbdt: Mapping[str, type[GBDTModel]],
    ) -> None:
        if self.class_weight != "balanced" or self.task != TABULAR_CLASSIFICATION_TASK:
            return
        unsupported = sorted(
            {
                spec.family
                for spec in specs
                if not _supports_sample_weight(
                    self._model(spec, n_classes, available_gbdt)
                )
            }
        )
        if unsupported:
            raise ValueError(
                "class_weight='balanced' requires estimators accepting sample_weight; "
                f"these families do not: {', '.join(unsupported)}"
            )

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        specs: Sequence[EstimatorSpec] | None = None,
        groups: npt.ArrayLike | None = None,
    ) -> CandidateRun:
        feature_values, target_values, group_values = _validated_training_data(
            X, y, groups
        )
        n_classes = None
        if self.task == TABULAR_CLASSIFICATION_TASK:
            n_classes = int(np.unique(target_values).size)
            if n_classes < 2:
                raise ValueError("Classification requires at least two target classes")
        available_gbdt = self._available_gbdt(specs, n_classes)
        selected_specs = (
            _portfolio_from_families(self.task, set(available_gbdt))
            if specs is None
            else tuple(specs)
        )
        applicable_specs = tuple(
            spec
            for spec in selected_specs
            if spec.is_applicable(
                n_rows=len(feature_values),
                n_features=feature_values.shape[1],
            )
        )
        if not applicable_specs:
            raise ValueError("No candidate specifications apply to this dataset")
        self._assert_sample_weight_support(applicable_specs, n_classes, available_gbdt)

        started_at = self.clock()
        observed_fit_times: list[float] = []
        trained: list[TrainedCandidate] = []
        stopped_for_budget = False
        last_error: Exception | None = None
        total_candidates = len(applicable_specs)
        for index, spec in enumerate(applicable_specs, start=1):
            elapsed = self.clock() - started_at
            if trained and self._budget_prevents_next_fit(elapsed, observed_fit_times):
                stopped_for_budget = True
                logger.info(
                    "Candidate training stopped to preserve the time reserved for "
                    "ensembling and export."
                )
                break
            model = self._model(spec, n_classes, available_gbdt)
            (
                train_X,
                train_y,
                sample_weight,
                validation_data,
                early_stopping_rounds,
            ) = self._fit_inputs(
                spec,
                feature_values,
                target_values,
                group_values,
            )
            fit_started_at = self.clock()
            try:
                model.fit(
                    train_X,
                    train_y,
                    sample_weight=sample_weight,
                    validation_data=validation_data,
                    early_stopping_rounds=early_stopping_rounds,
                )
            except Exception as error:
                last_error = error
                logger.warning(
                    "Candidate %d/%d (%s) failed: %s",
                    index,
                    total_candidates,
                    spec.name,
                    error,
                )
            else:
                fit_time = self.clock() - fit_started_at
                trained.append(TrainedCandidate(spec, model, fit_time))
            fit_time = self.clock() - fit_started_at
            observed_fit_times.append(fit_time)
            estimated_remaining = float(np.mean(observed_fit_times)) * (
                total_candidates - index
            )
            logger.info(
                "Candidate %d/%d (%s) completed in %.2fs; estimated remaining time "
                "%.2fs.",
                index,
                total_candidates,
                spec.name,
                fit_time,
                estimated_remaining,
            )
            candidate_budget = self._candidate_budget()
            if (
                index == 1
                and candidate_budget is not None
                and fit_time > candidate_budget
            ):
                logger.warning(
                    "The time limit is insufficient for the first candidate: it took "
                    "%.2fs with %.2fs available for candidate training. Continuing with "
                    "the fitted candidate.",
                    fit_time,
                    candidate_budget,
                )

        if not trained:
            raise RuntimeError("No candidate model could be trained") from last_error
        return CandidateRun(
            tuple(trained),
            self.clock() - started_at,
            stopped_for_budget,
        )


def score_oof_predictions(
    predictions: npt.NDArray[Any],
    y: npt.NDArray[Any],
    task: str,
    *,
    prior_correct: bool = True,
) -> float:
    """Return a higher-is-better OOF score; regression uses negative RMSE.

    With `prior_correct`, classification decisions are taken at `argmax p_c / pi_c`
    rather than plain argmax. That is the asymptotic Bayes rule for balanced accuracy,
    and it approximates the tuned decision rule the exported model carries.
    """
    _validate_task(task)
    target_values = np.asarray(y).reshape(-1)
    prediction_values = np.asarray(predictions)
    if not np.isfinite(prediction_values).all():
        raise ValueError("OOF predictions must contain only finite values")
    if task == TABULAR_CLASSIFICATION_TASK:
        classes, counts = np.unique(target_values, return_counts=True)
        if classes.size < 2:
            raise ValueError("Classification requires at least two target classes")
        if prediction_values.ndim != 2:
            raise ValueError("Classification OOF predictions must be two-dimensional")
        if prediction_values.shape != (len(target_values), len(classes)):
            raise ValueError(
                "Classification OOF predictions must contain one column per class"
            )
        scores = prediction_values
        if prior_correct:
            scores = prediction_values / (counts / len(target_values))
        predicted_labels = classes[np.argmax(scores, axis=1)]
        return float(balanced_accuracy_score(target_values, predicted_labels))

    prediction_values = prediction_values.reshape(-1)
    if prediction_values.shape != target_values.shape:
        raise ValueError("Regression OOF predictions must match the target shape")
    residuals = prediction_values.astype(np.float64) - target_values.astype(np.float64)
    return -float(np.sqrt(np.mean(np.square(residuals))))


@dataclass(frozen=True)
class GreedySelection:
    weights: tuple[float, ...]
    score: float
    iterations: int


def greedy_weighted_selection(
    predictions: Sequence[npt.NDArray[Any]],
    y: npt.NDArray[Any],
    task: str,
    *,
    max_iterations: int = 100,
    prior_correct: bool = True,
) -> GreedySelection:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if not predictions:
        raise ValueError("At least one candidate prediction is required")
    candidate_predictions = tuple(
        np.asarray(candidate, dtype=np.float64) for candidate in predictions
    )
    expected_shape = candidate_predictions[0].shape
    if any(candidate.shape != expected_shape for candidate in candidate_predictions):
        raise ValueError("All candidate OOF predictions must have the same shape")

    individual_scores = np.asarray(
        [
            score_oof_predictions(candidate, y, task, prior_correct=prior_correct)
            for candidate in candidate_predictions
        ]
    )
    first_index = int(np.argmax(individual_scores))
    counts = np.zeros(len(candidate_predictions), dtype=np.int64)
    counts[first_index] = 1
    prediction_sum = candidate_predictions[first_index].copy()
    current_score = float(individual_scores[first_index])
    iterations = 1

    for iteration in range(2, max_iterations + 1):
        trial_scores = np.asarray(
            [
                score_oof_predictions(
                    (prediction_sum + candidate) / iteration,
                    y,
                    task,
                    prior_correct=prior_correct,
                )
                for candidate in candidate_predictions
            ]
        )
        best_index = int(np.argmax(trial_scores))
        best_score = float(trial_scores[best_index])
        improvement_floor = np.finfo(np.float64).eps * max(1.0, abs(current_score))
        if best_score <= current_score + improvement_floor:
            break
        counts[best_index] += 1
        prediction_sum += candidate_predictions[best_index]
        current_score = best_score
        iterations = iteration

    weights = tuple(float(count / iterations) for count in counts)
    return GreedySelection(weights, current_score, iterations)


@dataclass(frozen=True)
class OOFCandidate:
    spec: EstimatorSpec
    models: tuple[CandidateModel, ...]
    oof_predictions: npt.NDArray[np.float32]
    oof_score: float
    fit_time: float


@dataclass(frozen=True)
class EnsembleMember:
    spec: EstimatorSpec
    models: tuple[CandidateModel, ...]
    weight: float


class GreedyWeightedEnsemble:
    def __init__(
        self,
        task: str,
        candidates: Sequence[OOFCandidate],
        selection: GreedySelection,
        *,
        classes: Sequence[int] | None = None,
    ) -> None:
        _validate_task(task)
        if len(candidates) != len(selection.weights):
            raise ValueError("The selection must contain one weight per candidate")
        if not candidates:
            raise ValueError("An ensemble requires at least one candidate")
        if any(weight < 0 for weight in selection.weights):
            raise ValueError("Ensemble weights must be non-negative")
        total_weight = sum(selection.weights)
        if total_weight <= 0:
            raise ValueError("At least one ensemble weight must be positive")
        if task == TABULAR_CLASSIFICATION_TASK:
            if classes is None or len(classes) < 2:
                raise ValueError(
                    "Classification ensembles require at least two classes"
                )
            class_values = np.asarray(classes)
            if not np.issubdtype(class_values.dtype, np.integer):
                raise ValueError("Classification targets must be integer encoded")
            self.classes: npt.NDArray[np.int64] | None = class_values.astype(np.int64)
        else:
            self.classes = None

        self.task = task
        self.score = selection.score
        self.iterations = selection.iterations
        self.weights = tuple(weight / total_weight for weight in selection.weights)
        self.members = tuple(
            EnsembleMember(candidate.spec, candidate.models, weight)
            for candidate, weight in zip(candidates, self.weights, strict=True)
            if weight > 0
        )

    def _member_prediction(
        self,
        member: EnsembleMember,
        X: npt.NDArray[Any],
    ) -> npt.NDArray[np.float32]:
        if self.task == TABULAR_CLASSIFICATION_TASK:
            fold_predictions = [
                np.asarray(
                    model.predict_proba(X),
                    dtype=np.float32,
                )
                for model in (
                    model for model in member.models if hasattr(model, "predict_proba")
                )
            ]
            if len(fold_predictions) != len(member.models):
                raise RuntimeError(
                    "A classification fold model does not expose probabilities"
                )
        else:
            fold_predictions = [
                np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
                for model in member.models
            ]
        return np.mean(
            np.stack(fold_predictions, axis=0),
            axis=0,
            dtype=np.float32,
        )

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if self.task != TABULAR_CLASSIFICATION_TASK:
            raise RuntimeError("Regression ensembles do not expose probabilities")
        weighted = [
            self._member_prediction(member, X) * np.float32(member.weight)
            for member in self.members
        ]
        return np.sum(np.stack(weighted, axis=0), axis=0, dtype=np.float32)

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.task == TABULAR_CLASSIFICATION_TASK:
            if self.classes is None:
                raise RuntimeError("Classification ensemble classes are unavailable")
            return self.classes[np.argmax(self.predict_proba(X), axis=1)]
        weighted = [
            self._member_prediction(member, X) * np.float32(member.weight)
            for member in self.members
        ]
        return np.sum(np.stack(weighted, axis=0), axis=0, dtype=np.float32)

    def serialize(self) -> SerializedModelRepr:
        return serialize_parallel_ensemble(
            [[model.serialize() for model in member.models] for member in self.members],
            [member.weight for member in self.members],
            self.task,
            classes=None if self.classes is None else self.classes.tolist(),
        )

    def get_input_type(self) -> object:
        return Float32Array

    def get_output_type(self) -> object:
        return Int64Array if self.task == TABULAR_CLASSIFICATION_TASK else Float32Array


@dataclass(frozen=True)
class EnsembleRun:
    candidates: tuple[OOFCandidate, ...]
    ensemble: GreedyWeightedEnsemble
    evaluation_indices: npt.NDArray[np.int64]
    ensemble_score_history: tuple[float, ...]
    elapsed_time: float
    stopped_for_budget: bool
    stopped_for_plateau: bool


class OOFEnsembleTrainer:
    def __init__(
        self,
        task: str,
        *,
        max_iterations: int = 100,
        plateau_enabled: bool = True,
        plateau_patience: int = 3,
        plateau_tolerance: float = 1e-4,
        n_splits: int = 5,
        time_limit: float | None = None,
        reserve_fraction: float = _DEFAULT_RESERVE_FRACTION,
        random_state: int = 42,
        class_weight: ClassWeight = "none",
        prior_correct: bool = True,
        model_factory: CandidateModelFactory | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._candidate_trainer = CandidateTrainer(
            task,
            time_limit=time_limit,
            reserve_fraction=reserve_fraction,
            random_state=random_state,
            class_weight=class_weight,
            model_factory=model_factory,
            clock=clock,
        )
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if plateau_patience < 1:
            raise ValueError("plateau_patience must be at least 1")
        if plateau_tolerance < 0:
            raise ValueError("plateau_tolerance must be non-negative")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.max_iterations = max_iterations
        self.task = task
        self.random_state = random_state
        self.prior_correct = prior_correct
        self.clock = self._candidate_trainer.clock
        self.plateau_enabled = plateau_enabled
        self.plateau_patience = plateau_patience
        self.plateau_tolerance = plateau_tolerance
        self.n_splits = n_splits

    def _fit_oof_candidate(
        self,
        spec: EstimatorSpec,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        groups: npt.NDArray[Any] | None,
        splits: Sequence[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
        evaluation_indices: npt.NDArray[np.int64],
        n_classes: int | None,
        available_gbdt: Mapping[str, type[GBDTModel]],
        *,
        warn_for_budget_shortfall: bool = False,
    ) -> OOFCandidate:
        output_shape: tuple[int, ...]
        if self.task == TABULAR_CLASSIFICATION_TASK:
            if n_classes is None:
                raise RuntimeError("Classification class count is unavailable")
            output_shape = (len(X), n_classes)
        else:
            output_shape = (len(X),)
        all_predictions = np.full(output_shape, np.nan, dtype=np.float32)
        models: list[CandidateModel] = []
        started_at = self.clock()
        budget_shortfall_warned = False
        for fold_index, (train_indices, eval_indices) in enumerate(splits):
            fold_seed = self.random_state + fold_index
            model = self._candidate_trainer._model(
                spec,
                n_classes,
                available_gbdt,
                random_state=fold_seed,
            )
            fold_groups = None if groups is None else groups[train_indices]
            (
                train_X,
                train_y,
                sample_weight,
                validation_data,
                early_stopping_rounds,
            ) = self._candidate_trainer._fit_inputs(
                spec,
                X[train_indices],
                y[train_indices],
                fold_groups,
                random_state=fold_seed,
            )
            model.fit(
                train_X,
                train_y,
                sample_weight=sample_weight,
                validation_data=validation_data,
                early_stopping_rounds=early_stopping_rounds,
            )
            expected_shape: tuple[int, ...]
            if self.task == TABULAR_CLASSIFICATION_TASK:
                if n_classes is None:
                    raise RuntimeError("Classification class count is unavailable")
                predict_proba = getattr(model, "predict_proba", None)
                if not callable(predict_proba):
                    raise TypeError(
                        "Classification candidate models must expose predict_proba"
                    )
                fold_predictions = np.asarray(
                    predict_proba(X[eval_indices]),
                    dtype=np.float32,
                )
                expected_shape = (len(eval_indices), n_classes)
            else:
                fold_predictions = np.asarray(
                    model.predict(X[eval_indices]),
                    dtype=np.float32,
                ).reshape(-1)
                expected_shape = (len(eval_indices),)
            if fold_predictions.shape != expected_shape:
                raise ValueError(
                    f"Candidate {spec.name} produced OOF predictions with shape "
                    f"{fold_predictions.shape}; expected {expected_shape}"
                )
            all_predictions[eval_indices] = fold_predictions
            models.append(model)

            candidate_budget = self._candidate_trainer._candidate_budget()
            completed_folds = fold_index + 1
            elapsed = self.clock() - started_at
            estimated_fit_time = elapsed / completed_folds * len(splits)
            if (
                warn_for_budget_shortfall
                and not budget_shortfall_warned
                and candidate_budget is not None
                and estimated_fit_time > candidate_budget
            ):
                logger.warning(
                    "The time limit is insufficient for the first candidate: "
                    "%d/%d OOF folds took %.2fs, estimating %.2fs with %.2fs "
                    "available for candidate training. Continuing with the first "
                    "candidate.",
                    completed_folds,
                    len(splits),
                    elapsed,
                    estimated_fit_time,
                    candidate_budget,
                )
                budget_shortfall_warned = True

        oof_predictions = all_predictions[evaluation_indices]
        score = score_oof_predictions(
            oof_predictions,
            y[evaluation_indices],
            self.task,
            prior_correct=self.prior_correct,
        )
        return OOFCandidate(
            spec,
            tuple(models),
            oof_predictions,
            score,
            self.clock() - started_at,
        )

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        specs: Sequence[EstimatorSpec] | None = None,
        groups: npt.ArrayLike | None = None,
        splits: Sequence[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
        | None = None,
    ) -> EnsembleRun:
        feature_values, target_values, group_values = _validated_training_data(
            X, y, groups
        )
        classes: npt.NDArray[np.int64] | None = None
        n_classes = None
        if self.task == TABULAR_CLASSIFICATION_TASK:
            if not np.issubdtype(target_values.dtype, np.integer):
                raise ValueError("Classification targets must be integer encoded")
            classes = np.unique(target_values).astype(np.int64)
            n_classes = len(classes)
            if n_classes < 2:
                raise ValueError("Classification requires at least two target classes")
        available_gbdt = self._candidate_trainer._available_gbdt(specs, n_classes)
        selected_specs = (
            _portfolio_from_families(self.task, set(available_gbdt))
            if specs is None
            else tuple(specs)
        )
        applicable_specs = tuple(
            spec
            for spec in selected_specs
            if spec.is_applicable(
                n_rows=len(feature_values),
                n_features=feature_values.shape[1],
            )
        )
        if not applicable_specs:
            raise ValueError("No candidate specifications apply to this dataset")
        self._candidate_trainer._assert_sample_weight_support(
            applicable_specs,
            n_classes,
            available_gbdt,
        )

        if splits is None:
            resolved_splits = out_of_fold_indices(
                feature_values,
                target_values,
                self.task,
                group_values,
                n_splits=self.n_splits,
                random_state=self.random_state,
            )
        else:
            resolved_splits = list(splits)
            if not resolved_splits:
                raise ValueError("At least one OOF split is required")
        evaluation_indices = np.sort(
            np.concatenate([eval_indices for _, eval_indices in resolved_splits])
        ).astype(np.int64, copy=False)
        if np.unique(evaluation_indices).size != evaluation_indices.size:
            raise RuntimeError("OOF evaluation folds contain duplicate rows")
        if len(resolved_splits) > 1 and evaluation_indices.size != len(feature_values):
            raise RuntimeError("Cross-validation OOF folds do not cover every row")

        started_at = self.clock()
        observed_fit_times: list[float] = []
        candidates: list[OOFCandidate] = []
        score_history: list[float] = []
        current_selection: GreedySelection | None = None
        best_ensemble_score: float | None = None
        without_improvement = 0
        stopped_for_budget = False
        stopped_for_plateau = False
        last_error: Exception | None = None
        total_candidates = len(applicable_specs)

        for index, spec in enumerate(applicable_specs, start=1):
            elapsed = self.clock() - started_at
            if candidates and self._candidate_trainer._budget_prevents_next_fit(
                elapsed, observed_fit_times
            ):
                stopped_for_budget = True
                logger.info(
                    "Candidate training stopped to preserve the time reserved for "
                    "ensembling and export."
                )
                break

            fit_started_at = self.clock()
            try:
                candidate = self._fit_oof_candidate(
                    spec,
                    feature_values,
                    target_values,
                    group_values,
                    resolved_splits,
                    evaluation_indices,
                    n_classes,
                    available_gbdt,
                    warn_for_budget_shortfall=index == 1,
                )
            except Exception as error:
                last_error = error
                logger.warning(
                    "Candidate %d/%d (%s) failed: %s",
                    index,
                    total_candidates,
                    spec.name,
                    error,
                )
            else:
                candidates.append(candidate)
                current_selection = greedy_weighted_selection(
                    [trained.oof_predictions for trained in candidates],
                    target_values[evaluation_indices],
                    self.task,
                    max_iterations=self.max_iterations,
                    prior_correct=self.prior_correct,
                )
                score_history.append(current_selection.score)
                if (
                    best_ensemble_score is None
                    or current_selection.score
                    > best_ensemble_score + self.plateau_tolerance
                ):
                    best_ensemble_score = current_selection.score
                    without_improvement = 0
                else:
                    without_improvement += 1

            fit_time = self.clock() - fit_started_at
            observed_fit_times.append(fit_time)
            estimated_remaining = float(np.mean(observed_fit_times)) * (
                total_candidates - index
            )
            logger.info(
                "Candidate %d/%d (%s) completed in %.2fs; estimated remaining time "
                "%.2fs.",
                index,
                total_candidates,
                spec.name,
                fit_time,
                estimated_remaining,
            )
            if (
                self.plateau_enabled
                and candidates
                and without_improvement >= self.plateau_patience
            ):
                stopped_for_plateau = True
                logger.info(
                    "Candidate training stopped after an OOF score plateau: no "
                    "improvement greater than %.6g for %d candidates.",
                    self.plateau_tolerance,
                    self.plateau_patience,
                )
                break

        if not candidates or current_selection is None:
            raise RuntimeError("No candidate model could be trained") from last_error
        ensemble = GreedyWeightedEnsemble(
            self.task,
            candidates,
            current_selection,
            classes=None if classes is None else classes.tolist(),
        )
        return EnsembleRun(
            tuple(candidates),
            ensemble,
            evaluation_indices,
            tuple(score_history),
            self.clock() - started_at,
            stopped_for_budget,
            stopped_for_plateau,
        )


__all__ = [
    "CandidateClassifierModel",
    "CandidateModel",
    "CandidateRun",
    "CandidateTrainer",
    "EnsembleMember",
    "EnsembleRun",
    "EstimatorSpec",
    "GreedySelection",
    "GreedyWeightedEnsemble",
    "OOFCandidate",
    "OOFEnsembleTrainer",
    "TrainedCandidate",
    "default_portfolio",
    "greedy_weighted_selection",
    "score_oof_predictions",
]
