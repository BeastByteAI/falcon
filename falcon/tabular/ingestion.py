from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy import typing as npt

from falcon import types as ft
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.type_guessing import determine_column_types
from falcon.types import DatasetSchema, TargetKind
from falcon.utils import logger

TabularData: TypeAlias = str | npt.NDArray[Any] | pd.DataFrame | tuple[Any, Any]


def read_data(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError("Only `.csv` and `.parquet` files are supported")


def _target_kind(task: str) -> TargetKind:
    if task == TABULAR_CLASSIFICATION_TASK:
        return "classification"
    if task == TABULAR_REGRESSION_TASK:
        return "regression"
    raise ValueError(f"Unknown tabular task `{task}`")


def _validate_features(features: ft.ColumnsList | None) -> None:
    if features is not None and len(features) < 1:
        raise ValueError("Features List cannot be empty")


def _dataframe_parts(
    data: pd.DataFrame,
    features: ft.ColumnsList | None,
    target: str | int | None,
) -> tuple[pd.DataFrame, pd.Series[Any], tuple[str, ...], str]:
    _validate_features(features)
    if features is not None and target is None:
        raise ValueError(
            "Either both target and features should be provided or neither of them."
        )
    if data.shape[1] < 2 and features is None:
        raise ValueError("Tabular data must contain at least one feature and a target")

    if target is None:
        target_data: pd.Series[Any] | pd.DataFrame = data.iloc[:, -1]
        target_name = str(data.columns[-1])
    elif isinstance(target, str):
        target_data = data.loc[:, target]
        target_name = target
    else:
        target_data = data.iloc[:, target]
        target_name = str(data.columns[target])
    if isinstance(target_data, pd.DataFrame):
        if target_data.shape[1] != 1:
            raise ValueError("The target should contain only one column.")
        target_data = target_data.iloc[:, 0]

    if features is None:
        if target is None:
            feature_data = data.iloc[:, :-1]
        elif isinstance(target, str):
            feature_data = data.loc[:, data.columns != target]
        else:
            target_position = target % data.shape[1]
            positions = [
                position
                for position in range(data.shape[1])
                if position != target_position
            ]
            feature_data = data.iloc[:, positions]
    elif all(isinstance(feature, str) for feature in features):
        feature_data = data.loc[:, cast(list[str], features)]
    elif all(isinstance(feature, (int, np.integer)) for feature in features):
        feature_data = data.iloc[:, cast(list[int], features)]
    else:
        raise ValueError("Features must contain only column names or only indices")

    column_names = tuple(str(column) for column in feature_data.columns)
    return feature_data, target_data, column_names, target_name


def _array_parts(
    data: npt.NDArray[Any],
    features: ft.ColumnsList | None,
    target: str | int | None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], tuple[str, ...], str]:
    _validate_features(features)
    if data.ndim != 2:
        raise ValueError("Tabular arrays must be two-dimensional")
    if data.shape[1] < 2 and features is None:
        raise ValueError("Tabular data must contain at least one feature and a target")
    if (target is None or features is None) and not (
        target is None and features is None
    ):
        raise ValueError(
            "Either both target and features should be provided or neither of them."
        )
    if features is not None and not all(
        isinstance(feature, (int, np.integer)) for feature in features
    ):
        raise ValueError("Expected list of integers as features, found strings")
    if isinstance(target, str):
        raise ValueError("Expected integer as target, found string")

    if features is None:
        feature_indices = list(range(data.shape[1] - 1))
        target_index = data.shape[1] - 1
    else:
        feature_indices = [int(feature) for feature in features]
        if target is None:
            raise RuntimeError("Target validation did not run")
        target_index = int(target)

    X = data[:, feature_indices]
    y = data[:, target_index]
    column_names = tuple(f"feature_{index}" for index in feature_indices)
    return X, y, column_names, "target"


def _tuple_parts(
    data: tuple[Any, Any],
    features: ft.ColumnsList | None,
    target: str | int | None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], tuple[str, ...], str]:
    if len(data) != 2:
        raise ValueError(
            "When passing data as tuple, it should contain exactly 2 elements: `X` and `y`."
        )
    if features is not None or target is not None:
        logger.warning(
            "When data is passed as tuple of (X, y) all columns are used regardless the values of `features` or `target` arguments."
        )

    raw_X, raw_y = data
    if isinstance(raw_X, pd.DataFrame):
        column_names = tuple(str(column) for column in raw_X.columns)
        X = raw_X.to_numpy(dtype=np.object_)
    else:
        X = np.asarray(raw_X, dtype=np.object_)
        if X.ndim != 2:
            raise ValueError("Feature arrays must be two-dimensional")
        column_names = tuple(f"feature_{index}" for index in range(X.shape[1]))

    if isinstance(raw_y, pd.DataFrame):
        if raw_y.shape[1] != 1:
            raise ValueError("The target should contain only one column.")
        target_name = str(raw_y.columns[0])
        y = raw_y.to_numpy(dtype=np.object_)
    elif isinstance(raw_y, pd.Series):
        target_name = str(raw_y.name) if raw_y.name is not None else "target"
        y = raw_y.to_numpy(dtype=np.object_)
    else:
        target_name = "target"
        y = np.asarray(raw_y, dtype=np.object_)
    return X, y, column_names, target_name


def _validate_shapes(
    X: npt.NDArray[Any], y: npt.NDArray[Any], column_names: tuple[str, ...]
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.object_]]:
    X = np.asarray(X, dtype=np.object_)
    y = np.asarray(y, dtype=np.object_)
    if X.ndim != 2:
        raise ValueError("Feature arrays must be two-dimensional")
    if X.shape[1] == 0:
        raise ValueError("Tabular data must contain at least one feature")
    if len(column_names) != X.shape[1]:
        raise ValueError("Feature names do not match the feature array")
    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError("The target should contain only one column.")
        y = y[:, 0]
    elif y.ndim != 1:
        raise ValueError("The target should contain only one column.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Features and target must contain the same number of rows")
    if X.shape[0] == 0:
        raise ValueError("Tabular data must contain at least one row")
    return X, y


def _drop_missing_targets(
    X: npt.NDArray[np.object_],
    y: npt.NDArray[np.object_],
    row_indices: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
    npt.NDArray[np.int64],
]:
    missing_target = np.asarray(pd.isna(y), dtype=np.bool_)
    missing_count = int(missing_target.sum())
    if missing_count == 1:
        logger.info("Dropped 1 row with a missing target value.")
    elif missing_count > 1:
        logger.info("Dropped %d rows with missing target values.", missing_count)
    if missing_count:
        keep = ~missing_target
        X = X[keep]
        y = y[keep]
        row_indices = row_indices[keep]
    if X.shape[0] == 0:
        raise ValueError("No rows remain after dropping missing target values")
    return X, y, row_indices


def ingest_data_with_row_selection(
    data: TabularData,
    task: str,
    features: ft.ColumnsList | None = None,
    target: str | int | None = None,
) -> tuple[
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
    DatasetSchema,
    npt.NDArray[np.int64],
    int,
]:
    if isinstance(data, str):
        data = read_data(data)

    if isinstance(data, tuple):
        X, y, column_names, target_name = _tuple_parts(data, features, target)
    elif isinstance(data, pd.DataFrame):
        feature_data, target_data, column_names, target_name = _dataframe_parts(
            data, features, target
        )
        X = feature_data.to_numpy(dtype=np.object_)
        y = target_data.to_numpy(dtype=np.object_)
    elif isinstance(data, np.ndarray):
        X, y, column_names, target_name = _array_parts(data, features, target)
    else:
        raise TypeError(
            "Data must be a csv/parquet path, DataFrame, ndarray, or (X, y) tuple"
        )

    X, y = _validate_shapes(X, y, column_names)
    source_row_count = X.shape[0]
    row_indices: npt.NDArray[np.int64] = np.arange(source_row_count, dtype=np.int64)
    X, y, row_indices = _drop_missing_targets(X, y, row_indices)
    column_types = tuple(determine_column_types(X))
    schema = DatasetSchema(
        column_names=column_names,
        column_types=column_types,
        target_name=target_name,
        target_kind=_target_kind(task),
        dimensions=cast(tuple[int, int], X.shape),
    )
    return X, y, schema, row_indices, source_row_count


def ingest_data(
    data: TabularData,
    task: str,
    features: ft.ColumnsList | None = None,
    target: str | int | None = None,
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.object_], DatasetSchema]:
    X, y, schema, _, _ = ingest_data_with_row_selection(
        data,
        task=task,
        features=features,
        target=target,
    )
    return X, y, schema
