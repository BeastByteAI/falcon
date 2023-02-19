import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from onnxconverter_common.data_types import StringTensorType
from skl2onnx.proto import onnx_proto
from skl2onnx import update_registered_converter


class DateTimeTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, format: str):
        if format not in (r"%Y-%m-%d"):
            raise ValueError("Selected date format is not supported")
        self.format = format

    def fit(self, X: any):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert (
            len(X.shape) < 2 or X.shape[1] == 1
        ), "DateTimeTokenizer only accepts single column arrays"
        if self.format == r"%Y-%m-%d":
            self.separators = ["-"]
            return np.asarray(np.char.split(X.astype(str), sep="-").tolist()).squeeze()


def _dt_shape_calculator(operator):  # type: ignore
    if operator.raw_operator.format == r"%Y-%m-%d":
        c = 3
    else:
        ValueError("DateTimeTokenizer contains unknown date format")

    operator.outputs[0].type = StringTensorType([None, c])


def _dt_converter(scope, operator, container):  # type: ignore
    in_name = operator.inputs[0].full_name
    out_name = operator.outputs[0].full_name
    tokenizer_name = scope.get_unique_operator_name("dt_token")
    sq_name = scope.get_unique_operator_name("dt_squeeze")
    tokenized = scope.get_unique_operator_name("tokenized")

    axis_name = scope.get_unique_variable_name("axis")
    axis = np.asarray([1], dtype=np.int64)

    proto_dtype = onnx_proto.TensorProto.INT64

    container.add_initializer(
        name=axis_name, onnx_type=proto_dtype, shape=[1], content=axis
    )

    attrs = {
        "pad_value": "0",
        "mark": False,
        "mincharnum": 1,
        "separators": operator.raw_operator.separators,
    }

    container.add_node(
        "Tokenizer",
        [in_name],
        [tokenized],
        name=tokenizer_name,
        op_domain="com.microsoft",
        op_version=1,
        **attrs
    )

    container.add_node(
        "Squeeze", [tokenized, axis_name], [out_name], name=sq_name, op_domain=""
    )


update_registered_converter(
    DateTimeTokenizer, "FalconDateTimeTokenizer", _dt_shape_calculator, _dt_converter
)
