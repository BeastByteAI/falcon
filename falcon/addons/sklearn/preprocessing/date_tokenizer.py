import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from onnxconverter_common.data_types import StringTensorType
from skl2onnx.proto import onnx_proto
from skl2onnx import update_registered_converter
import re


class DateTimeTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, format: str):
        if format not in (r"%Y-%m-%d", r"%Y-%m-%d %H:%M:%S", r"%Y-%m-%dT%H:%M:%SZ"):
            raise ValueError("Selected date format is not supported")
        self.format = format

    def fit(self, X):
        self.fit_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert (
            len(X.shape) < 2 or X.shape[1] == 1
        ), "DateTimeTokenizer only accepts single column arrays"
        if self.format == r"%Y-%m-%d":
            self.out_dim = 3
        elif self.format in (r"%Y-%m-%d %H:%M:%S", r"%Y-%m-%dT%H:%M:%SZ"):
            self.out_dim = 6
        r = re.compile("[0-9]+")
        l = np.apply_along_axis(lambda x: r.findall(str(x)), -1, X).reshape(-1, self.out_dim)
        return l
        
def _dt_shape_calculator(operator):  # type: ignore
    c = operator.raw_operator.out_dim
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
        "tokenexp": r"[0-9]+",
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
