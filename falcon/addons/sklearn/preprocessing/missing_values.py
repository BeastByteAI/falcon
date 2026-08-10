from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.proto import onnx_proto
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

MISSING_STRING_TOKENS: tuple[str, ...] = ("nan", "None", "<NA>", "NaT")


def string_values_and_missing_mask(
    X: npt.NDArray[Any],
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.bool_]]:
    values = np.asarray(X, dtype=np.object_)
    if values.ndim not in (1, 2):
        raise ValueError("String features must be one- or two-dimensional")
    missing = np.asarray(pd.isna(values), dtype=np.bool_)
    strings = values.astype(np.str_)
    missing |= np.isin(strings, MISSING_STRING_TOKENS)
    return strings, missing


def numeric_values(X: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    raw_values = np.asarray(X, dtype=np.object_)
    if raw_values.ndim != 2:
        raise ValueError("Numeric features must be two-dimensional")
    missing = np.asarray(pd.isna(raw_values), dtype=np.bool_)
    values = np.empty(raw_values.shape, dtype=np.float64)
    values[missing] = np.nan
    try:
        values[~missing] = raw_values[~missing].astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Numeric features contain a non-numeric value") from error
    return values


class NumericMedianImputer(BaseEstimator, TransformerMixin):
    statistics_: npt.NDArray[np.float64]
    n_features_in_: int

    def fit(self, X: npt.NDArray[Any], y: Any = None) -> NumericMedianImputer:
        values = numeric_values(X)
        all_missing = np.isnan(values).all(axis=0)
        if all_missing.any():
            column = int(np.flatnonzero(all_missing)[0])
            raise ValueError(f"Cannot impute all-missing numeric column {column}")
        self.statistics_ = np.nanmedian(values, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        check_is_fitted(self, ("statistics_", "n_features_in_"))
        values = numeric_values(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Numeric feature count differs from fitted data")
        missing = np.isnan(values)
        filled = np.where(missing, self.statistics_, values)
        return np.concatenate((filled, missing.astype(np.float64)), axis=1).astype(
            np.float32
        )


class NumericCast(BaseEstimator, TransformerMixin):
    """
    Numeric counterpart of `NumericMedianImputer` for pipelines built without
    imputation. Exports to a single `Cast` node so that no `IsNaN`/`Where` pair
    ends up in the graph, and emits no missing indicator column.
    """

    n_features_in_: int

    def fit(self, X: npt.NDArray[Any], y: Any = None) -> NumericCast:
        values = numeric_values(X)
        if np.isnan(values).any():
            column = int(np.flatnonzero(np.isnan(values).any(axis=0))[0])
            raise ValueError(
                f"Numeric column {column} contains missing values, which cannot be "
                "handled while imputation is disabled"
            )
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        check_is_fitted(self, "n_features_in_")
        values = numeric_values(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Numeric feature count differs from fitted data")
        return values.astype(np.float32)


class MissingStringImputer(BaseEstimator, TransformerMixin):
    fill_value: str
    n_features_in_: int

    def __init__(self, fill_value: str) -> None:
        self.fill_value = fill_value

    def fit(self, X: npt.NDArray[Any], y: Any = None) -> MissingStringImputer:
        values, _ = string_values_and_missing_mask(X)
        self.n_features_in_ = values.shape[1] if values.ndim == 2 else 1
        return self

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.str_]:
        check_is_fitted(self, "n_features_in_")
        values, missing = string_values_and_missing_mask(X)
        n_features = values.shape[1] if values.ndim == 2 else 1
        if n_features != self.n_features_in_:
            raise ValueError("String feature count differs from fitted data")
        return np.where(missing, self.fill_value, values).astype(np.str_)


class StringCast(BaseEstimator, TransformerMixin):
    """
    String counterpart of `MissingStringImputer` for pipelines built without
    imputation. Missing values keep their plain string representation
    (`"nan"`, `"None"`, ...) and are encoded as ordinary categories, which keeps
    the exported graph free of the `Equal`/`Or`/`Where` mask.
    """

    n_features_in_: int

    def fit(self, X: npt.NDArray[Any], y: Any = None) -> StringCast:
        values, _ = string_values_and_missing_mask(X)
        self.n_features_in_ = values.shape[1] if values.ndim == 2 else 1
        return self

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.str_]:
        check_is_fitted(self, "n_features_in_")
        values, _ = string_values_and_missing_mask(X)
        n_features = values.shape[1] if values.ndim == 2 else 1
        if n_features != self.n_features_in_:
            raise ValueError("String feature count differs from fitted data")
        return values


def add_string_missing_mask(
    scope: Any,
    container: Any,
    input_name: str,
    name: str,
) -> str:
    comparisons: list[str] = []
    for index, token in enumerate(MISSING_STRING_TOKENS):
        token_name = scope.get_unique_variable_name(f"{name}_token_{index}")
        container.add_initializer(
            token_name,
            onnx_proto.TensorProto.STRING,
            [],
            [token],
        )
        comparison = scope.get_unique_variable_name(f"{name}_equal_{index}")
        container.add_node(
            "Equal",
            [input_name, token_name],
            [comparison],
            name=scope.get_unique_operator_name(f"{name}_equal_{index}"),
            op_domain="",
        )
        comparisons.append(comparison)

    missing = comparisons[0]
    for index, comparison in enumerate(comparisons[1:], start=1):
        combined = scope.get_unique_variable_name(f"{name}_or_{index}")
        container.add_node(
            "Or",
            [missing, comparison],
            [combined],
            name=scope.get_unique_operator_name(f"{name}_or_{index}"),
            op_domain="",
        )
        missing = combined
    return missing


def _numeric_shape_calculator(operator: Any) -> None:
    batch_size = operator.inputs[0].get_first_dimension()
    feature_count = operator.inputs[0].type.shape[1]
    output_features = None if feature_count is None else feature_count * 2
    operator.outputs[0].type = FloatTensorType([batch_size, output_features])


def _numeric_converter(scope: Any, operator: Any, container: Any) -> None:
    transformer: NumericMedianImputer = operator.raw_operator
    check_is_fitted(transformer, ("statistics_", "n_features_in_"))
    input_name = operator.inputs[0].full_name
    output_name = operator.outputs[0].full_name

    missing = scope.get_unique_variable_name("numeric_missing")
    container.add_node(
        "IsNaN",
        [input_name],
        [missing],
        name=scope.get_unique_operator_name("numeric_is_nan"),
        op_domain="",
    )
    statistics = scope.get_unique_variable_name("numeric_medians")
    container.add_initializer(
        statistics,
        onnx_proto.TensorProto.FLOAT,
        [1, transformer.n_features_in_],
        transformer.statistics_.astype(np.float32).reshape(1, -1),
    )
    filled = scope.get_unique_variable_name("imputed_numeric_values")
    container.add_node(
        "Where",
        [missing, statistics, input_name],
        [filled],
        name=scope.get_unique_operator_name("impute_numeric_values"),
        op_domain="",
    )
    indicator = scope.get_unique_variable_name("numeric_missing_indicator")
    container.add_node(
        "Cast",
        [missing],
        [indicator],
        name=scope.get_unique_operator_name("cast_numeric_missing_indicator"),
        op_domain="",
        to=onnx_proto.TensorProto.FLOAT,
    )
    container.add_node(
        "Concat",
        [filled, indicator],
        [output_name],
        name=scope.get_unique_operator_name("append_numeric_missing_indicator"),
        op_domain="",
        axis=1,
    )


def _numeric_cast_shape_calculator(operator: Any) -> None:
    batch_size = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = FloatTensorType(
        [batch_size, operator.inputs[0].type.shape[1]]
    )


def _numeric_cast_converter(scope: Any, operator: Any, container: Any) -> None:
    check_is_fitted(operator.raw_operator, "n_features_in_")
    container.add_node(
        "Cast",
        [operator.inputs[0].full_name],
        [operator.outputs[0].full_name],
        name=scope.get_unique_operator_name("cast_numeric_values"),
        op_domain="",
        to=onnx_proto.TensorProto.FLOAT,
    )


def _string_shape_calculator(operator: Any) -> None:
    operator.outputs[0].type = StringTensorType(operator.inputs[0].type.shape)


def _string_converter(scope: Any, operator: Any, container: Any) -> None:
    transformer: MissingStringImputer = operator.raw_operator
    check_is_fitted(transformer, "n_features_in_")
    input_name = operator.inputs[0].full_name
    output_name = operator.outputs[0].full_name
    missing = add_string_missing_mask(scope, container, input_name, "string_missing")
    fill_value = scope.get_unique_variable_name("string_fill_value")
    container.add_initializer(
        fill_value,
        onnx_proto.TensorProto.STRING,
        [],
        [transformer.fill_value],
    )
    container.add_node(
        "Where",
        [missing, fill_value, input_name],
        [output_name],
        name=scope.get_unique_operator_name("impute_string_values"),
        op_domain="",
    )


def _string_cast_converter(scope: Any, operator: Any, container: Any) -> None:
    check_is_fitted(operator.raw_operator, "n_features_in_")
    container.add_node(
        "Identity",
        [operator.inputs[0].full_name],
        [operator.outputs[0].full_name],
        name=scope.get_unique_operator_name("pass_string_values"),
        op_domain="",
    )


update_registered_converter(
    NumericMedianImputer,
    "FalconNumericMedianImputer",
    _numeric_shape_calculator,
    _numeric_converter,
)
update_registered_converter(
    NumericCast,
    "FalconNumericCast",
    _numeric_cast_shape_calculator,
    _numeric_cast_converter,
)
update_registered_converter(
    MissingStringImputer,
    "FalconMissingStringImputer",
    _string_shape_calculator,
    _string_converter,
)
update_registered_converter(
    StringCast,
    "FalconStringCast",
    _string_shape_calculator,
    _string_cast_converter,
)
