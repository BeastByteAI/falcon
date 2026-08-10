from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy import typing as npt
from skl2onnx import update_registered_converter
from skl2onnx.common.data_types import DoubleTensorType
from skl2onnx.proto import onnx_proto
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from falcon.addons.sklearn.preprocessing.missing_values import (
    add_string_missing_mask,
    string_values_and_missing_mask,
)

DATE_FORMAT = r"%Y-%m-%d"
DATETIME_SPACE_FORMAT = r"%Y-%m-%d %H:%M:%S"
DATETIME_T_FORMAT = r"%Y-%m-%dT%H:%M:%SZ"

_DATE_VARIANT = "date"
_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATETIME_VARIANTS: dict[str, tuple[re.Pattern[str], str, bool]] = {
    "datetime_t_z": (
        re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"),
        "T",
        True,
    ),
    "datetime_t": (
        re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
        "T",
        False,
    ),
    "datetime_space_z": (
        re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}Z"),
        " ",
        True,
    ),
    "datetime_space": (
        re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"),
        " ",
        False,
    ),
}
_DATE_CYCLIC_INDICES = np.asarray([1, 2], dtype=np.int64)
_DATE_CYCLIC_PERIODS = np.asarray([12, 31], dtype=np.float64)
_DATETIME_CYCLIC_INDICES = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
_DATETIME_CYCLIC_PERIODS = np.asarray([12, 31, 24, 60, 60], dtype=np.float64)
# ONNX Runtime evaluates Sin/Cos only for floats; rounding prevents tiny kernel
# differences from sending native and exported tree models down different branches.
_CYCLIC_QUANTIZATION_SCALE = np.float32(100_000)


class DateTimeTokenizer(BaseEstimator, TransformerMixin):
    format: str
    variant_: str
    reference_value_: str
    raw_dim_: int
    out_dim: int

    def __init__(self, format: str) -> None:
        if format not in (DATE_FORMAT, DATETIME_SPACE_FORMAT, DATETIME_T_FORMAT):
            raise ValueError("Selected date format is not supported")
        self.format = format

    def fit(self, X: npt.NDArray[Any], y: Any = None) -> DateTimeTokenizer:
        values, missing = self._single_column_values(X)
        present_values = values[~missing]
        if present_values.size == 0:
            raise ValueError(
                "DateTimeTokenizer requires at least one non-missing value"
            )

        self.variant_ = self._detect_variant(present_values)
        self.reference_value_ = str(present_values[0])
        self.raw_dim_ = 3 if self.variant_ == _DATE_VARIANT else 6
        cyclic_count = 2 if self.variant_ == _DATE_VARIANT else 5
        self.out_dim = self.raw_dim_ + 2 * cyclic_count + 1
        return self

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        check_is_fitted(self, ("variant_", "reference_value_", "raw_dim_", "out_dim"))
        values, missing = self._single_column_values(X)
        if values.size == 0:
            return np.empty((0, self.out_dim), dtype=np.float64)

        present_values = values[~missing]
        if present_values.size:
            variant = self._detect_variant(present_values)
            if variant != self.variant_:
                raise ValueError(
                    "Input does not match the fitted datetime format variant "
                    f"{self.variant_!r}"
                )
        values = np.where(missing, self.reference_value_, values)
        variant = self.variant_

        rows: list[list[str]] = []
        if variant == _DATE_VARIANT:
            rows = [value.split("-") for value in values]
            cyclic_indices = _DATE_CYCLIC_INDICES
            cyclic_periods = _DATE_CYCLIC_PERIODS
        else:
            _, delimiter, has_trailing_z = _DATETIME_VARIANTS[variant]
            for value in values:
                date_value, time_value = value.split(delimiter, maxsplit=1)
                if has_trailing_z:
                    time_value = time_value.removesuffix("Z")
                rows.append(date_value.split("-") + time_value.split(":"))
            cyclic_indices = _DATETIME_CYCLIC_INDICES
            cyclic_periods = _DATETIME_CYCLIC_PERIODS

        components = np.asarray(rows, dtype=np.float64)
        angle_scales = (2 * np.pi) / cyclic_periods
        angles = (components[:, cyclic_indices] * angle_scales).astype(np.float32)
        sin_values = (
            np.round(np.sin(angles) * _CYCLIC_QUANTIZATION_SCALE)
            / _CYCLIC_QUANTIZATION_SCALE
        )
        cos_values = (
            np.round(np.cos(angles) * _CYCLIC_QUANTIZATION_SCALE)
            / _CYCLIC_QUANTIZATION_SCALE
        )
        return np.concatenate(
            (
                components,
                sin_values.astype(np.float64),
                cos_values.astype(np.float64),
                missing.astype(np.float64).reshape(-1, 1),
            ),
            axis=1,
        )

    def _detect_variant(self, values: npt.NDArray[np.str_]) -> str:
        if self.format == DATE_FORMAT:
            if all(_DATE_PATTERN.fullmatch(value) is not None for value in values):
                return _DATE_VARIANT
            raise ValueError("Values must all use the YYYY-MM-DD date format")

        detected: set[str] = set()
        for value in values:
            variant = next(
                (
                    name
                    for name, (pattern, _, _) in _DATETIME_VARIANTS.items()
                    if pattern.fullmatch(value) is not None
                ),
                None,
            )
            if variant is None:
                raise ValueError(
                    "Values must use a supported YYYY-MM-DD datetime format"
                )
            detected.add(variant)

        if len(detected) != 1:
            raise ValueError(
                "DateTimeTokenizer requires a single datetime format variant; "
                "mixed delimiters or trailing-Z usage were found"
            )
        return detected.pop()

    @staticmethod
    def _single_column_values(
        X: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.bool_]]:
        values, missing = string_values_and_missing_mask(X)
        if values.ndim == 1:
            return values, missing
        if values.ndim == 2 and values.shape[1] == 1:
            return values[:, 0], missing[:, 0]
        raise ValueError("DateTimeTokenizer only accepts single-column arrays")


def _dt_shape_calculator(operator: Any) -> None:
    check_is_fitted(operator.raw_operator, ("variant_", "reference_value_", "out_dim"))
    batch_size = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = DoubleTensorType(
        [batch_size, operator.raw_operator.out_dim]
    )


def _add_string_split(
    scope: Any,
    container: Any,
    input_name: str,
    delimiter: str,
    maxsplit: int,
    name: str,
) -> str:
    split_values = scope.get_unique_variable_name(f"{name}_values")
    split_counts = scope.get_unique_variable_name(f"{name}_counts")
    container.add_node(
        "StringSplit",
        [input_name],
        [split_values, split_counts],
        name=scope.get_unique_operator_name(name),
        op_domain="",
        op_version=20,
        delimiter=delimiter,
        maxsplit=maxsplit,
    )
    return split_values


def _add_squeeze(
    scope: Any,
    container: Any,
    input_name: str,
    axes_name: str,
    name: str,
) -> str:
    output_name = scope.get_unique_variable_name(f"{name}_output")
    container.add_node(
        "Squeeze",
        [input_name, axes_name],
        [output_name],
        name=scope.get_unique_operator_name(name),
        op_domain="",
    )
    return output_name


def _add_int64_initializer(
    scope: Any, container: Any, name: str, values: npt.NDArray[np.int64]
) -> str:
    initializer_name = scope.get_unique_variable_name(name)
    container.add_initializer(
        name=initializer_name,
        onnx_type=onnx_proto.TensorProto.INT64,
        shape=list(values.shape),
        # Flattened to a list because onnx below 1.22 takes the length of the content,
        # which a 0-d array holding a scalar initializer cannot supply.
        content=values.reshape(-1).tolist(),
    )
    return initializer_name


def _dt_converter(scope: Any, operator: Any, container: Any) -> None:
    transformer: DateTimeTokenizer = operator.raw_operator
    check_is_fitted(transformer, ("variant_", "reference_value_", "out_dim"))
    input_name = operator.inputs[0].full_name
    output_name = operator.outputs[0].full_name
    missing = add_string_missing_mask(scope, container, input_name, "date_missing")
    reference_value = scope.get_unique_variable_name("date_reference_value")
    container.add_initializer(
        reference_value,
        onnx_proto.TensorProto.STRING,
        [],
        [transformer.reference_value_],
    )
    filled_input = scope.get_unique_variable_name("filled_date_values")
    container.add_node(
        "Where",
        [missing, reference_value, input_name],
        [filled_input],
        name=scope.get_unique_operator_name("fill_missing_date_values"),
        op_domain="",
    )
    squeeze_axis = _add_int64_initializer(
        scope, container, "date_squeeze_axis", np.asarray([1], dtype=np.int64)
    )

    if transformer.variant_ == _DATE_VARIANT:
        split_date = _add_string_split(
            scope, container, filled_input, "-", 2, "split_date"
        )
        raw_strings = _add_squeeze(
            scope, container, split_date, squeeze_axis, "squeeze_date"
        )
        cyclic_indices = _DATE_CYCLIC_INDICES
        cyclic_periods = _DATE_CYCLIC_PERIODS
    else:
        _, delimiter, has_trailing_z = _DATETIME_VARIANTS[transformer.variant_]
        split_datetime = _add_string_split(
            scope, container, filled_input, delimiter, 1, "split_datetime"
        )
        date_and_time = _add_squeeze(
            scope, container, split_datetime, squeeze_axis, "squeeze_datetime"
        )

        date_index = _add_int64_initializer(
            scope, container, "date_index", np.asarray([0], dtype=np.int64)
        )
        time_index = _add_int64_initializer(
            scope, container, "time_index", np.asarray([1], dtype=np.int64)
        )
        date_value = scope.get_unique_variable_name("date_value")
        time_value = scope.get_unique_variable_name("time_value")
        container.add_node(
            "Gather",
            [date_and_time, date_index],
            [date_value],
            name=scope.get_unique_operator_name("gather_date"),
            op_domain="",
            axis=1,
        )
        container.add_node(
            "Gather",
            [date_and_time, time_index],
            [time_value],
            name=scope.get_unique_operator_name("gather_time"),
            op_domain="",
            axis=1,
        )

        if has_trailing_z:
            split_z = _add_string_split(
                scope, container, time_value, "Z", 1, "strip_trailing_z"
            )
            first_token_index = _add_int64_initializer(
                scope,
                container,
                "first_token_index",
                np.asarray(0, dtype=np.int64),
            )
            time_without_z = scope.get_unique_variable_name("time_without_z")
            container.add_node(
                "Gather",
                [split_z, first_token_index],
                [time_without_z],
                name=scope.get_unique_operator_name("gather_time_without_z"),
                op_domain="",
                axis=2,
            )
            time_value = time_without_z

        split_date = _add_string_split(
            scope, container, date_value, "-", 2, "split_date"
        )
        split_time = _add_string_split(
            scope, container, time_value, ":", 2, "split_time"
        )
        date_components = _add_squeeze(
            scope, container, split_date, squeeze_axis, "squeeze_date"
        )
        time_components = _add_squeeze(
            scope, container, split_time, squeeze_axis, "squeeze_time"
        )
        raw_strings = scope.get_unique_variable_name("raw_datetime_components")
        container.add_node(
            "Concat",
            [date_components, time_components],
            [raw_strings],
            name=scope.get_unique_operator_name("concat_datetime_components"),
            op_domain="",
            axis=1,
        )
        cyclic_indices = _DATETIME_CYCLIC_INDICES
        cyclic_periods = _DATETIME_CYCLIC_PERIODS

    integer_components = scope.get_unique_variable_name("integer_date_components")
    container.add_node(
        "Cast",
        [raw_strings],
        [integer_components],
        name=scope.get_unique_operator_name("parse_date_components"),
        op_domain="",
        to=onnx_proto.TensorProto.INT64,
    )
    raw_components = scope.get_unique_variable_name("raw_date_components")
    container.add_node(
        "Cast",
        [integer_components],
        [raw_components],
        name=scope.get_unique_operator_name("cast_date_components"),
        op_domain="",
        to=onnx_proto.TensorProto.DOUBLE,
    )

    cyclic_index_name = _add_int64_initializer(
        scope, container, "cyclic_indices", cyclic_indices
    )
    cyclic_components = scope.get_unique_variable_name("cyclic_components")
    container.add_node(
        "Gather",
        [raw_components, cyclic_index_name],
        [cyclic_components],
        name=scope.get_unique_operator_name("gather_cyclic_components"),
        op_domain="",
        axis=1,
    )

    angle_scales = (2 * np.pi) / cyclic_periods
    angle_scale_name = scope.get_unique_variable_name("cyclic_angle_scales")
    container.add_initializer(
        name=angle_scale_name,
        onnx_type=onnx_proto.TensorProto.DOUBLE,
        shape=list(angle_scales.shape),
        content=angle_scales,
    )
    double_angles = scope.get_unique_variable_name("double_cyclic_angles")
    angles = scope.get_unique_variable_name("cyclic_angles")
    container.add_node(
        "Mul",
        [cyclic_components, angle_scale_name],
        [double_angles],
        name=scope.get_unique_operator_name("scale_cyclic_components"),
        op_domain="",
    )
    container.add_node(
        "Cast",
        [double_angles],
        [angles],
        name=scope.get_unique_operator_name("cast_cyclic_angles"),
        op_domain="",
        to=onnx_proto.TensorProto.FLOAT,
    )
    unrounded_sin = scope.get_unique_variable_name("unrounded_cyclic_sin")
    unrounded_cos = scope.get_unique_variable_name("unrounded_cyclic_cos")
    container.add_node(
        "Sin",
        [angles],
        [unrounded_sin],
        name=scope.get_unique_operator_name("sin_cyclic_components"),
        op_domain="",
    )
    container.add_node(
        "Cos",
        [angles],
        [unrounded_cos],
        name=scope.get_unique_operator_name("cos_cyclic_components"),
        op_domain="",
    )

    quantization_scale_name = scope.get_unique_variable_name(
        "cyclic_quantization_scale"
    )
    container.add_initializer(
        name=quantization_scale_name,
        onnx_type=onnx_proto.TensorProto.FLOAT,
        shape=[],
        # A one-element sequence rather than a 0-d array: onnx below 1.22 takes the
        # length of the content even for a scalar initializer, which an unsized array
        # cannot supply.
        content=[float(_CYCLIC_QUANTIZATION_SCALE)],
    )
    quantized_values: list[str] = []
    for function_name, unrounded_values in (
        ("sin", unrounded_sin),
        ("cos", unrounded_cos),
    ):
        scaled_values = scope.get_unique_variable_name(f"scaled_cyclic_{function_name}")
        rounded_values = scope.get_unique_variable_name(
            f"rounded_cyclic_{function_name}"
        )
        float_values = scope.get_unique_variable_name(f"float_cyclic_{function_name}")
        double_values = scope.get_unique_variable_name(f"double_cyclic_{function_name}")
        container.add_node(
            "Mul",
            [unrounded_values, quantization_scale_name],
            [scaled_values],
            name=scope.get_unique_operator_name(
                f"scale_cyclic_{function_name}_for_rounding"
            ),
            op_domain="",
        )
        container.add_node(
            "Round",
            [scaled_values],
            [rounded_values],
            name=scope.get_unique_operator_name(f"round_cyclic_{function_name}"),
            op_domain="",
        )
        container.add_node(
            "Div",
            [rounded_values, quantization_scale_name],
            [float_values],
            name=scope.get_unique_operator_name(
                f"unscale_cyclic_{function_name}_after_rounding"
            ),
            op_domain="",
        )
        container.add_node(
            "Cast",
            [float_values],
            [double_values],
            name=scope.get_unique_operator_name(f"cast_cyclic_{function_name}"),
            op_domain="",
            to=onnx_proto.TensorProto.DOUBLE,
        )
        quantized_values.append(double_values)

    missing_indicator = scope.get_unique_variable_name("date_missing_indicator")
    container.add_node(
        "Cast",
        [missing],
        [missing_indicator],
        name=scope.get_unique_operator_name("cast_date_missing_indicator"),
        op_domain="",
        to=onnx_proto.TensorProto.DOUBLE,
    )
    container.add_node(
        "Concat",
        [raw_components, *quantized_values, missing_indicator],
        [output_name],
        name=scope.get_unique_operator_name("concat_date_features"),
        op_domain="",
        axis=1,
    )


update_registered_converter(
    DateTimeTokenizer, "FalconDateTimeTokenizer", _dt_shape_calculator, _dt_converter
)
