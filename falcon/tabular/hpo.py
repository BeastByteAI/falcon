from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
from numpy import typing as npt

from falcon.config import ClassWeight
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.tabular.candidates import (
    CandidateTrainer,
    EstimatorSpec,
    score_oof_predictions,
)
from falcon.tabular.splitting import out_of_fold_indices
from falcon.utils import logger

_EARLY_STOPPING_FAMILIES = {
    "catboost",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
}
_SUPPORTED_FAMILIES = {
    "catboost",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "linear",
    "random_forest",
    "xgboost",
}


def _load_optuna() -> Any:
    try:
        return import_module("optuna")
    except ImportError as error:
        raise ImportError(
            "HPO candidate sources require the hpo extra; "
            "install it with `pip install falcon-ml[hpo]`"
        ) from error


def _base_parameters(family: str, task: str) -> dict[str, object]:
    classification = task == TABULAR_CLASSIFICATION_TASK
    if family == "hist_gradient_boosting":
        return {
            "learning_rate": 0.08,
            "l2_regularization": 0.1,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        }
    if family in {"extra_trees", "random_forest"}:
        return {
            "criterion": "gini" if classification else "squared_error",
            "max_features": 0.75,
            "min_samples_leaf": 1,
            "n_estimators": 300,
        }
    if family == "linear":
        return {"C": 1.0, "max_iter": 1_000} if classification else {"alpha": 1.0}
    if family == "lightgbm":
        return {
            "colsample_bytree": 1.0,
            "learning_rate": 0.1,
            "min_child_samples": 20,
            "n_estimators": 300,
            "num_leaves": 31,
        }
    if family == "xgboost":
        return {
            "colsample_bytree": 1.0,
            "learning_rate": 0.3,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "n_estimators": 300,
        }
    if family == "catboost":
        return {
            "depth": 6,
            "iterations": 300,
            "l2_leaf_reg": 3.0,
            "learning_rate": 0.1,
        }
    raise ValueError(
        f"Unknown HPO family `{family}`. Available families: "
        f"{', '.join(sorted(_SUPPORTED_FAMILIES))}."
    )


def _suggest_parameters(trial: Any, family: str, task: str) -> dict[str, object]:
    parameters = _base_parameters(family, task)
    if family == "hist_gradient_boosting":
        parameters.update(
            {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.03, 0.2, log=True
                ),
                "l2_regularization": trial.suggest_categorical(
                    "l2_regularization", [0.0, 0.1, 1.0]
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 300, step=100),
                "max_leaf_nodes": trial.suggest_categorical(
                    "max_leaf_nodes", [15, 31, 63]
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 10, 40, step=10
                ),
            }
        )
    elif family in {"extra_trees", "random_forest"}:
        parameters.update(
            {
                "max_features": trial.suggest_categorical(
                    "max_features", [0.5, 0.75, 1.0]
                ),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            }
        )
    elif family == "linear":
        name = "C" if task == TABULAR_CLASSIFICATION_TASK else "alpha"
        parameters[name] = trial.suggest_float(name, 1e-3, 100.0, log=True)
    elif family == "lightgbm":
        parameters.update(
            {
                "colsample_bytree": trial.suggest_categorical(
                    "colsample_bytree", [0.75, 1.0]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "min_child_samples": trial.suggest_categorical(
                    "min_child_samples", [10, 20, 40]
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63]),
            }
        )
    elif family == "xgboost":
        parameters.update(
            {
                "colsample_bytree": trial.suggest_categorical(
                    "colsample_bytree", [0.75, 1.0]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 0.5, 5.0, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            }
        )
    elif family == "catboost":
        parameters.update(
            {
                "depth": trial.suggest_int("depth", 4, 10),
                "iterations": trial.suggest_int("iterations", 100, 500, step=100),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
            }
        )
    return parameters


def _trial_zero_parameters(family: str, task: str) -> dict[str, object]:
    parameters = _base_parameters(family, task)
    if family in {"extra_trees", "random_forest"}:
        parameters.pop("criterion")
    elif family == "linear" and task == TABULAR_CLASSIFICATION_TASK:
        parameters.pop("max_iter")
    return parameters


def _estimator_spec(
    family: str,
    parameters: dict[str, object],
    trial_number: int,
) -> EstimatorSpec:
    early_stopping_rounds = 20 if family in _EARLY_STOPPING_FAMILIES else None
    return EstimatorSpec(
        name=f"hpo_{family}_trial_{trial_number}",
        family=family,
        parameters=parameters,
        early_stopping_rounds=early_stopping_rounds,
    )


def _trial_objective(
    optuna: Any,
    task: str,
    family: str,
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    groups: npt.NDArray[Any] | None,
    splits: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
    random_state: int,
    class_weight: ClassWeight,
    prior_correct: bool,
) -> Any:
    n_classes = int(np.unique(y).size) if task == TABULAR_CLASSIFICATION_TASK else None
    trainer = CandidateTrainer(
        task,
        random_state=random_state,
        class_weight=class_weight,
    )
    baseline = _estimator_spec(family, _base_parameters(family, task), 0)
    available_gbdt = trainer._available_gbdt((baseline,), n_classes)
    trainer._assert_sample_weight_support((baseline,), n_classes, available_gbdt)

    def objective(trial: Any) -> float:
        spec = _estimator_spec(
            family,
            _suggest_parameters(trial, family, task),
            trial.number,
        )
        output_shape: tuple[int, ...] = (
            (len(X), n_classes) if n_classes is not None else (len(X),)
        )
        predictions = np.full(output_shape, np.nan, dtype=np.float32)
        completed_indices: list[npt.NDArray[np.int64]] = []

        for fold_index, (train_indices, eval_indices) in enumerate(splits):
            fold_seed = random_state + fold_index
            model = trainer._model(
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
            ) = trainer._fit_inputs(
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
            if task == TABULAR_CLASSIFICATION_TASK:
                if n_classes is None:
                    raise RuntimeError("Classification class count is unavailable")
                predict_proba = getattr(model, "predict_proba", None)
                if not callable(predict_proba):
                    raise TypeError(
                        "Classification HPO candidates must expose predict_proba"
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
                    f"HPO candidate {spec.name} produced predictions with shape "
                    f"{fold_predictions.shape}; expected {expected_shape}"
                )
            predictions[eval_indices] = fold_predictions
            completed_indices.append(eval_indices)
            evaluated = np.sort(np.concatenate(completed_indices))
            intermediate_score = score_oof_predictions(
                predictions[evaluated],
                y[evaluated],
                task,
                prior_correct=prior_correct,
            )
            trial.report(intermediate_score, step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        evaluation_indices = np.sort(np.concatenate(completed_indices))
        return score_oof_predictions(
            predictions[evaluation_indices],
            y[evaluation_indices],
            task,
            prior_correct=prior_correct,
        )

    return objective


def _best_distinct_trials(study: Any, top_n: int) -> list[Any]:
    complete_trials = [
        trial
        for trial in study.trials
        if trial.value is not None and trial.state.name == "COMPLETE"
    ]
    complete_trials.sort(key=lambda trial: (-float(trial.value), trial.number))
    selected: list[Any] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for trial in complete_trials:
        signature = tuple(
            sorted((name, repr(value)) for name, value in trial.params.items())
        )
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(trial)
        if len(selected) == top_n:
            break
    return selected


def generate_hpo_candidates(
    task: str,
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    *,
    groups: npt.ArrayLike | None,
    family: str,
    n_trials: int,
    top_n: int,
    n_splits: int,
    time_limit: float | None,
    time_budget_fraction: float,
    random_state: int,
    class_weight: ClassWeight = "none",
    prior_correct: bool = True,
) -> tuple[EstimatorSpec, ...]:
    if task not in {TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK}:
        raise ValueError(f"Unknown task `{task}`")
    if family not in _SUPPORTED_FAMILIES:
        _base_parameters(family, task)
    optuna = _load_optuna()
    feature_values = np.asarray(X)
    target_values = np.asarray(y)
    if target_values.ndim == 2 and target_values.shape[1] == 1:
        target_values = target_values[:, 0]
    if (
        feature_values.ndim != 2
        or target_values.ndim != 1
        or len(feature_values) != len(target_values)
    ):
        raise ValueError("HPO features and targets must contain matching rows")
    if task == TABULAR_CLASSIFICATION_TASK and not np.issubdtype(
        target_values.dtype, np.integer
    ):
        raise ValueError("Classification targets must be integer encoded")
    group_values = None if groups is None else np.asarray(groups)
    if group_values is not None and group_values.ndim == 2:
        if group_values.shape[1] == 1:
            group_values = group_values[:, 0]
    if group_values is not None and (
        group_values.ndim != 1 or len(group_values) != len(feature_values)
    ):
        raise ValueError("Groups must contain one value per HPO feature row")

    splits = out_of_fold_indices(
        feature_values,
        target_values,
        task,
        group_values,
        n_splits=n_splits,
        random_state=random_state,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        ),
    )
    study.enqueue_trial(_trial_zero_parameters(family, task))
    timeout = None if time_limit is None else time_limit * time_budget_fraction
    study.optimize(
        _trial_objective(
            optuna,
            task,
            family,
            feature_values,
            target_values,
            group_values,
            splits,
            random_state,
            class_weight,
            prior_correct,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    selected_trials = _best_distinct_trials(study, top_n)
    if not selected_trials:
        raise RuntimeError(f"HPO produced no successful `{family}` trials")
    logger.info(
        "HPO completed %d trial(s) for %s and emitted %d candidate(s).",
        len(study.trials),
        family,
        len(selected_trials),
    )
    return tuple(
        _estimator_spec(
            family,
            {**_base_parameters(family, task), **trial.params},
            trial.number,
        )
        for trial in selected_trials
    )


__all__ = ["generate_hpo_candidates"]
