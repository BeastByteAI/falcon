from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.types import ColumnTypes, DatasetSchema

if TYPE_CHECKING:
    from falcon.tabular.candidates import EstimatorSpec


_MIN_ROWS = 100
_MAX_ROWS = 400_000
_MIN_FEATURES = 3
_MAX_FEATURES = 10_936


@dataclass(frozen=True)
class DatasetMetaFeatures:
    n_rows: int
    n_features: int
    class_balance: float | None
    categorical_fraction: float
    text_fraction: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_rows, bool)
            or not isinstance(self.n_rows, Integral)
            or self.n_rows < 1
        ):
            raise ValueError("n_rows must be positive")
        if (
            isinstance(self.n_features, bool)
            or not isinstance(self.n_features, Integral)
            or self.n_features < 1
        ):
            raise ValueError("n_features must be positive")
        if self.class_balance is not None and not _is_fraction(
            self.class_balance, include_zero=False
        ):
            raise ValueError("class_balance must be between zero and one")
        if not _is_fraction(self.categorical_fraction):
            raise ValueError("categorical_fraction must be between zero and one")
        if not _is_fraction(self.text_fraction):
            raise ValueError("text_fraction must be between zero and one")
        if self.categorical_fraction + self.text_fraction > 1:
            raise ValueError(
                "categorical_fraction and text_fraction must not sum above one"
            )


@dataclass(frozen=True)
class _PerformanceProfile:
    task: str
    meta_features: DatasetMetaFeatures
    mean_ranks: tuple[tuple[str, float], ...]


def _is_fraction(value: object, *, include_zero: bool = True) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    numeric_value = float(value)
    above_lower_bound = numeric_value >= 0 if include_zero else numeric_value > 0
    return math.isfinite(numeric_value) and above_lower_bound and numeric_value <= 1


def _ranks(**values: float) -> tuple[tuple[str, float], ...]:
    return tuple(values.items())


# Mean per-dataset ranks reduced from the public TabRepo 2023-11-14 config results.
_PERFORMANCE_PROFILES = (
    _PerformanceProfile(
        TABULAR_CLASSIFICATION_TASK,
        DatasetMetaFeatures(1_031, 929, 0.1270, 0.0025, 0.0),
        _ranks(
            catboost_zeroshot_r177=3.167,
            catboost_default=3.375,
            lightgbm_default=4.292,
            xgboost_zeroshot_r33=4.333,
            linear_default=5.292,
            xgboost_default=5.375,
            lightgbm_zeroshot_large=6.000,
            random_forest_zeroshot=6.521,
            extra_trees_zeroshot=6.646,
        ),
    ),
    _PerformanceProfile(
        TABULAR_CLASSIFICATION_TASK,
        DatasetMetaFeatures(42_580, 11, 0.0980, 0.2302, 0.0),
        _ranks(
            catboost_zeroshot_r177=3.162,
            lightgbm_default=3.737,
            xgboost_default=3.921,
            catboost_default=3.973,
            xgboost_zeroshot_r33=4.211,
            lightgbm_zeroshot_large=4.447,
            random_forest_zeroshot=6.459,
            extra_trees_zeroshot=7.054,
            linear_default=7.789,
        ),
    ),
    _PerformanceProfile(
        TABULAR_CLASSIFICATION_TASK,
        DatasetMetaFeatures(7_540, 21, 0.1543, 0.9231, 0.0),
        _ranks(
            catboost_zeroshot_r177=2.667,
            catboost_default=2.956,
            lightgbm_default=4.267,
            xgboost_zeroshot_r33=4.400,
            xgboost_default=4.489,
            lightgbm_zeroshot_large=4.911,
            extra_trees_zeroshot=6.822,
            random_forest_zeroshot=6.844,
            linear_default=7.644,
        ),
    ),
    _PerformanceProfile(
        TABULAR_CLASSIFICATION_TASK,
        DatasetMetaFeatures(6_660, 20, 0.1421, 0.0627, 0.0),
        _ranks(
            catboost_default=2.580,
            catboost_zeroshot_r177=2.580,
            lightgbm_default=3.909,
            xgboost_default=4.398,
            xgboost_zeroshot_r33=4.807,
            lightgbm_zeroshot_large=5.557,
            extra_trees_zeroshot=6.580,
            linear_default=7.091,
            random_forest_zeroshot=7.500,
        ),
    ),
    _PerformanceProfile(
        TABULAR_REGRESSION_TASK,
        DatasetMetaFeatures(1_147, 126, None, 0.0034, 0.0),
        _ranks(
            catboost_default=3.250,
            lightgbm_zeroshot_large=3.750,
            catboost_zeroshot_r177=4.000,
            xgboost_default=4.500,
            lightgbm_default=4.750,
            xgboost_zeroshot_r33=5.500,
            random_forest_zeroshot=6.250,
            extra_trees_zeroshot=6.250,
            linear_default=6.750,
        ),
    ),
    _PerformanceProfile(
        TABULAR_REGRESSION_TASK,
        DatasetMetaFeatures(34_525, 18, None, 0.1727, 0.0),
        _ranks(
            catboost_zeroshot_r177=2.125,
            lightgbm_default=2.875,
            lightgbm_zeroshot_large=2.875,
            catboost_default=3.000,
            xgboost_default=6.000,
            xgboost_zeroshot_r33=6.143,
            extra_trees_zeroshot=6.250,
            random_forest_zeroshot=6.625,
            linear_default=8.750,
        ),
    ),
    _PerformanceProfile(
        TABULAR_REGRESSION_TASK,
        DatasetMetaFeatures(9_622, 13, None, 0.4000, 0.0),
        _ranks(
            catboost_zeroshot_r177=2.545,
            catboost_default=3.636,
            lightgbm_default=3.909,
            lightgbm_zeroshot_large=4.000,
            xgboost_zeroshot_r33=4.700,
            xgboost_default=5.364,
            extra_trees_zeroshot=5.909,
            random_forest_zeroshot=6.727,
            linear_default=7.818,
        ),
    ),
    _PerformanceProfile(
        TABULAR_REGRESSION_TASK,
        DatasetMetaFeatures(3_759, 9, None, 0.0, 0.0),
        _ranks(
            lightgbm_default=3.000,
            xgboost_default=3.333,
            xgboost_zeroshot_r33=3.667,
            catboost_zeroshot_r177=4.000,
            extra_trees_zeroshot=4.667,
            lightgbm_zeroshot_large=5.333,
            random_forest_zeroshot=6.000,
            catboost_default=6.000,
            linear_default=9.000,
        ),
    ),
)


def extract_dataset_meta_features(
    y: npt.ArrayLike,
    schema: DatasetSchema,
) -> DatasetMetaFeatures:
    target_values = np.asarray(y)
    if target_values.ndim != 1:
        raise ValueError("Target data must be one-dimensional")
    if len(target_values) != schema.n_rows:
        raise ValueError("Target data does not match the dataset schema")

    class_balance = None
    if schema.target_kind == "classification":
        _, class_counts = np.unique(target_values, return_counts=True)
        class_balance = float(class_counts.min() / len(target_values))

    categorical_types = {ColumnTypes.CAT_LOW_CARD, ColumnTypes.CAT_HIGH_CARD}
    categorical_columns = sum(
        column_type in categorical_types for column_type in schema.column_types
    )
    text_columns = schema.column_types.count(ColumnTypes.TEXT_UTF8)
    return DatasetMetaFeatures(
        n_rows=schema.n_rows,
        n_features=schema.n_features,
        class_balance=class_balance,
        categorical_fraction=categorical_columns / schema.n_features,
        text_fraction=text_columns / schema.n_features,
    )


def _in_corpus_range(meta_features: DatasetMetaFeatures) -> bool:
    return (
        _MIN_ROWS <= meta_features.n_rows <= _MAX_ROWS
        and _MIN_FEATURES <= meta_features.n_features <= _MAX_FEATURES
    )


def _scaled_log(value: int, minimum: int, maximum: int) -> float:
    return (math.log(value) - math.log(minimum)) / (
        math.log(maximum) - math.log(minimum)
    )


def _profile_distance(
    left: DatasetMetaFeatures,
    right: DatasetMetaFeatures,
    task: str,
) -> float:
    differences = [
        _scaled_log(left.n_rows, _MIN_ROWS, _MAX_ROWS)
        - _scaled_log(right.n_rows, _MIN_ROWS, _MAX_ROWS),
        _scaled_log(left.n_features, _MIN_FEATURES, _MAX_FEATURES)
        - _scaled_log(right.n_features, _MIN_FEATURES, _MAX_FEATURES),
        left.categorical_fraction - right.categorical_fraction,
        left.text_fraction - right.text_fraction,
    ]
    if task == TABULAR_CLASSIFICATION_TASK:
        if left.class_balance is None or right.class_balance is None:
            raise ValueError("Classification ordering requires class_balance")
        differences.append((left.class_balance - right.class_balance) * 2)
    return sum(difference**2 for difference in differences)


def reorder_portfolio(
    specs: Sequence[EstimatorSpec],
    meta_features: DatasetMetaFeatures,
    task: str,
    *,
    random_state: int,
) -> tuple[EstimatorSpec, ...]:
    if task not in {TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK}:
        raise ValueError(f"Unknown task `{task}`")
    if task == TABULAR_CLASSIFICATION_TASK and meta_features.class_balance is None:
        raise ValueError("Classification ordering requires class_balance")
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer")

    static_order = tuple(specs)
    if not static_order or not _in_corpus_range(meta_features):
        return static_order
    profiles = tuple(
        profile for profile in _PERFORMANCE_PROFILES if profile.task == task
    )
    nearest = min(
        profiles,
        key=lambda profile: _profile_distance(
            meta_features,
            profile.meta_features,
            task,
        ),
    )
    ranks = dict(nearest.mean_ranks)
    known_positions = [
        index for index, spec in enumerate(static_order) if spec.name in ranks
    ]
    if len(known_positions) < 2:
        return static_order

    generator = np.random.default_rng(random_state)
    tie_breakers = generator.random(len(static_order))
    ordered_known = sorted(
        ((index, static_order[index]) for index in known_positions),
        key=lambda item: (
            ranks[item[1].name],
            tie_breakers[item[0]],
        ),
    )
    reordered = list(static_order)
    for index, (_, spec) in zip(known_positions, ordered_known, strict=True):
        reordered[index] = spec
    return tuple(reordered)


__all__ = [
    "DatasetMetaFeatures",
    "extract_dataset_meta_features",
    "reorder_portfolio",
]
