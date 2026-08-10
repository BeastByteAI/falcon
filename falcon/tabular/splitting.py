from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)

from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK

GroupBy: TypeAlias = str | Sequence[str] | npt.ArrayLike
SplitIndices: TypeAlias = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]
EvalStrategy: TypeAlias = Literal["cv", "holdout"]

_DEFAULT_CV_SPLITS = 5
_DEFAULT_TEST_SIZE = 0.25
_HOLDOUT_ROW_THRESHOLD = 2_500


def _factorize_rows(values: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    try:
        hashes = pd.util.hash_pandas_object(
            pd.DataFrame(values), index=False
        ).to_numpy()
    except TypeError as error:
        raise ValueError("Group values must be scalar and hashable") from error
    groups, _ = pd.factorize(hashes, sort=False)
    return np.asarray(groups, dtype=np.int64)


def _named_group_columns(
    group_by: GroupBy,
    column_names: tuple[str, ...],
    n_rows: int,
) -> tuple[str, ...] | None:
    if isinstance(group_by, str):
        return (group_by,)
    if not isinstance(group_by, (list, tuple)):
        return None
    if not group_by:
        raise ValueError("group_by must not be empty")
    if not all(isinstance(value, str) for value in group_by):
        return None

    names = cast(Sequence[str], group_by)
    if all(name in column_names for name in names) or len(names) != n_rows:
        return tuple(names)
    return None


def _column_indices(
    requested_names: tuple[str, ...], column_names: tuple[str, ...]
) -> list[int]:
    indices: list[int] = []
    for name in requested_names:
        matching = [
            index
            for index, column_name in enumerate(column_names)
            if column_name == name
        ]
        if not matching:
            raise ValueError(f"Unknown group_by feature column `{name}`")
        if len(matching) > 1:
            raise ValueError(f"group_by feature column `{name}` is ambiguous")
        indices.append(matching[0])
    return indices


def _explicit_group_values(
    group_by: GroupBy,
    n_rows: int,
    source_row_indices: npt.NDArray[np.int64] | None,
    source_row_count: int | None,
) -> npt.NDArray[Any]:
    values = np.asarray(group_by, dtype=np.object_)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("Explicit group_by values must be one-dimensional")
    if values.shape[0] == n_rows:
        return values
    if (
        source_row_indices is not None
        and source_row_count is not None
        and values.shape[0] == source_row_count
    ):
        return values[source_row_indices]
    raise ValueError(
        "Explicit group_by values must contain one value per input data row"
    )


def resolve_groups(
    X: npt.NDArray[Any],
    column_names: tuple[str, ...],
    group_by: GroupBy | None = None,
    *,
    source_row_indices: npt.NDArray[np.int64] | None = None,
    source_row_count: int | None = None,
) -> npt.NDArray[np.int64]:
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("Features must be two-dimensional when resolving groups")
    if values.shape[1] != len(column_names):
        raise ValueError("Feature names do not match the feature array")

    if group_by is None:
        grouping_values = values
    else:
        requested_names = _named_group_columns(group_by, column_names, values.shape[0])
        if requested_names is None:
            grouping_values = _explicit_group_values(
                group_by,
                values.shape[0],
                source_row_indices,
                source_row_count,
            )
        else:
            grouping_values = values[:, _column_indices(requested_names, column_names)]
    return _factorize_rows(np.asarray(grouping_values))


def resolve_evaluation_strategy(n_rows: int) -> EvalStrategy:
    return "cv" if n_rows < _HOLDOUT_ROW_THRESHOLD else "holdout"


def _normalized_groups(
    X: npt.NDArray[Any], groups: npt.ArrayLike | None
) -> npt.NDArray[np.int64]:
    if groups is None:
        column_names = tuple(f"feature_{index}" for index in range(X.shape[1]))
        return resolve_groups(X, column_names)
    group_values = np.asarray(groups, dtype=np.object_)
    if group_values.ndim == 2 and group_values.shape[1] == 1:
        group_values = group_values[:, 0]
    if group_values.ndim != 1 or group_values.shape[0] != X.shape[0]:
        raise ValueError("Groups must contain one value per feature row")
    return _factorize_rows(group_values)


def _validate_split_inputs(
    X: npt.NDArray[Any], y: npt.NDArray[Any], groups: npt.ArrayLike | None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.int64]]:
    feature_values = np.asarray(X)
    target_values = np.asarray(y)
    if feature_values.ndim != 2:
        raise ValueError("Features must be two-dimensional")
    if target_values.ndim == 2 and target_values.shape[1] == 1:
        target_values = target_values[:, 0]
    if target_values.ndim != 1:
        raise ValueError("Target values must be one-dimensional")
    if feature_values.shape[0] != target_values.shape[0]:
        raise ValueError("Features and target must contain the same number of rows")
    normalized_groups = _normalized_groups(feature_values, groups)
    if np.unique(normalized_groups).size < 2:
        raise ValueError("At least two distinct groups are required to split the data")
    return feature_values, target_values, normalized_groups


def _validate_task(task: str) -> None:
    if task not in {TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK}:
        raise ValueError(f"Unknown tabular task `{task}`")


def _classification_split_count(
    y: npt.NDArray[Any],
    groups: npt.NDArray[np.int64],
    requested_splits: int,
) -> int:
    class_codes, classes = pd.factorize(pd.Series(y), sort=False)
    if (class_codes < 0).any():
        raise ValueError("Classification targets must not contain missing values")
    if len(classes) < 2:
        raise ValueError("Classification splitting requires at least two classes")
    groups_per_class = [
        np.unique(groups[class_codes == class_code]).size
        for class_code in range(len(classes))
    ]
    n_splits = min(requested_splits, np.unique(groups).size, *groups_per_class)
    if n_splits < 2:
        raise ValueError(
            "Stratified group splitting requires each class to occur in at least two groups"
        )
    return int(n_splits)


def _classification_codes(y: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    class_codes, _ = pd.factorize(pd.Series(y), sort=False)
    return np.asarray(class_codes, dtype=np.int64)


def _coerce_indices(values: Any, split_name: str) -> npt.NDArray[np.int64]:
    indices = np.asarray(values)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(
            f"{split_name} indices must be a one-dimensional integer array"
        )
    return indices.astype(np.int64, copy=False)


def _validated_split(
    split: tuple[Any, Any],
    groups: npt.NDArray[np.int64],
    split_name: str,
) -> SplitIndices:
    train_indices = _coerce_indices(split[0], "Training")
    eval_indices = _coerce_indices(split[1], "Evaluation")
    if train_indices.size == 0 or eval_indices.size == 0:
        raise ValueError(f"{split_name} must contain non-empty train and eval subsets")
    n_rows = groups.shape[0]
    if (
        (train_indices < 0).any()
        or (eval_indices < 0).any()
        or (train_indices >= n_rows).any()
        or (eval_indices >= n_rows).any()
    ):
        raise ValueError(f"{split_name} contains an out-of-range row index")
    if (
        np.unique(train_indices).size != train_indices.size
        or np.unique(eval_indices).size != eval_indices.size
    ):
        raise ValueError(f"{split_name} contains duplicate row indices")
    if np.intersect1d(train_indices, eval_indices).size:
        raise ValueError(f"{split_name} places a row on both sides")
    overlapping_groups = np.intersect1d(groups[train_indices], groups[eval_indices])
    if overlapping_groups.size:
        raise ValueError(
            f"{split_name} places group {overlapping_groups[0]} on both sides"
        )
    return train_indices, eval_indices


def _holdout_balance_score(
    y: npt.NDArray[Any], eval_indices: npt.NDArray[np.int64], test_size: float
) -> float:
    class_codes, classes = pd.factorize(pd.Series(y), sort=False)
    overall = np.bincount(class_codes, minlength=len(classes)) / len(class_codes)
    evaluation = np.bincount(class_codes[eval_indices], minlength=len(classes)) / len(
        eval_indices
    )
    size_difference = abs(len(eval_indices) / len(y) - test_size)
    return float(size_difference + np.abs(overall - evaluation).sum())


def holdout_indices(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    task: str,
    groups: npt.ArrayLike | None = None,
    *,
    test_size: float = _DEFAULT_TEST_SIZE,
    random_state: int = 42,
) -> SplitIndices:
    _validate_task(task)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    feature_values, target_values, normalized_groups = _validate_split_inputs(
        X, y, groups
    )

    if task == TABULAR_CLASSIFICATION_TASK:
        requested_splits = max(2, round(1 / test_size))
        n_splits = _classification_split_count(
            target_values, normalized_groups, requested_splits
        )
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        split_targets = _classification_codes(target_values)
        candidates = [
            _validated_split(split, normalized_groups, "Holdout split")
            for split in splitter.split(
                feature_values, split_targets, normalized_groups
            )
        ]
        return min(
            candidates,
            key=lambda split: _holdout_balance_score(
                target_values, split[1], test_size
            ),
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    split = next(splitter.split(feature_values, target_values, normalized_groups))
    return _validated_split(split, normalized_groups, "Holdout split")


def cross_validation_indices(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    task: str,
    groups: npt.ArrayLike | None = None,
    *,
    cv: BaseCrossValidator | None = None,
    n_splits: int = _DEFAULT_CV_SPLITS,
    random_state: int = 42,
) -> list[SplitIndices]:
    _validate_task(task)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    feature_values, target_values, normalized_groups = _validate_split_inputs(
        X, y, groups
    )

    if cv is None and task == TABULAR_CLASSIFICATION_TASK:
        split_count = _classification_split_count(
            target_values, normalized_groups, n_splits
        )
        splitter: BaseCrossValidator = StratifiedGroupKFold(n_splits=split_count)
        split_targets = _classification_codes(target_values)
    elif cv is None:
        split_count = min(n_splits, np.unique(normalized_groups).size)
        splitter = GroupKFold(n_splits=split_count)
        split_targets = target_values
    else:
        splitter = cv
        split_targets = target_values

    splits = [
        _validated_split(split, normalized_groups, f"Cross-validation split {index}")
        for index, split in enumerate(
            splitter.split(feature_values, split_targets, normalized_groups), start=1
        )
    ]
    if not splits:
        raise ValueError("Cross-validator produced no splits")
    return splits


def out_of_fold_indices(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    task: str,
    groups: npt.ArrayLike | None = None,
    *,
    n_splits: int = _DEFAULT_CV_SPLITS,
    random_state: int = 42,
) -> list[SplitIndices]:
    if resolve_evaluation_strategy(len(X)) == "holdout":
        return [
            holdout_indices(
                X,
                y,
                task,
                groups,
                random_state=random_state,
            )
        ]
    return cross_validation_indices(
        X,
        y,
        task,
        groups,
        n_splits=n_splits,
        random_state=random_state,
    )


def _row_hashes(X: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[np.uint64]:
    combined = np.column_stack((X, y.reshape(-1, 1)))
    hashes = pd.util.hash_pandas_object(pd.DataFrame(combined), index=False).to_numpy(
        dtype=np.uint64
    )
    return hashes


def _consume_row_indices(
    hashes: npt.NDArray[np.uint64],
    available: dict[int, deque[int]],
    subset_name: str,
) -> npt.NDArray[np.int64]:
    indices: list[int] = []
    for row_hash in hashes:
        matches = available[int(row_hash)]
        if not matches:
            raise ValueError(
                f"Callable splitter returned a {subset_name} row not present in the input"
            )
        indices.append(matches.popleft())
    return np.asarray(indices, dtype=np.int64)


def _array_split_indices(
    result: Sequence[Any],
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
) -> SplitIndices:
    train_X = np.asarray(result[0])
    eval_X = np.asarray(result[1])
    train_y = np.asarray(result[2]).reshape(-1)
    eval_y = np.asarray(result[3]).reshape(-1)
    if (
        train_X.ndim != 2
        or eval_X.ndim != 2
        or train_X.shape[1] != X.shape[1]
        or eval_X.shape[1] != X.shape[1]
        or train_X.shape[0] != train_y.shape[0]
        or eval_X.shape[0] != eval_y.shape[0]
    ):
        raise ValueError("Callable splitter returned invalid train/eval arrays")

    available: dict[int, deque[int]] = defaultdict(deque)
    for index, row_hash in enumerate(_row_hashes(X, y)):
        available[int(row_hash)].append(index)
    train_indices = _consume_row_indices(
        _row_hashes(train_X, train_y), available, "training"
    )
    eval_indices = _consume_row_indices(
        _row_hashes(eval_X, eval_y), available, "evaluation"
    )
    return train_indices, eval_indices


def callable_holdout_indices(
    splitter: Callable[..., Any],
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    groups: npt.ArrayLike | None = None,
) -> SplitIndices:
    feature_values, target_values, normalized_groups = _validate_split_inputs(
        X, y, groups
    )
    result = splitter(feature_values, target_values, normalized_groups.copy())
    if not isinstance(result, (tuple, list)):
        raise ValueError("Callable splitter must return a tuple or list")
    if len(result) == 2:
        split: tuple[Any, Any] = (result[0], result[1])
    elif len(result) == 4:
        split = _array_split_indices(result, feature_values, target_values)
    else:
        raise ValueError(
            "Callable splitter must return train/eval indices or four train/eval arrays"
        )
    return _validated_split(split, normalized_groups, "Callable split")
