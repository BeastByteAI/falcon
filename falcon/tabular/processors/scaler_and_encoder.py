import numpy as np
from numpy import typing as npt
from falcon.types import Float32Array
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from falcon.abstract import Processor, ONNXConvertible
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from falcon.config import ONNX_OPSET_VERSION
from typing import List, Optional, Type, Any
from falcon.types import SerializedModelTuple


class ScalerAndEncoder(Processor, ONNXConvertible):
    """
    Applies OneHotEncoder on categorical features and StandardScaler on numerical features.
    """
    
    def __init__(
        self, mask: List[bool]
    ) -> None:  # [True -> categorical, False -> numerical]
        """
        Parameters
        ----------
        mask : List[bool]
            Boolean mask with True/False for categorical/numerical features.
        """
        self.booleans_list = mask

    def fit(self, X: npt.NDArray, _: Any = None) -> None:
        """
        Fits the encoder.

        Parameters
        ----------
        X : npt.NDArray
            Data to encode
        _ : Any, optional
            dummy argument to keep compatibility with pipeline training, by default None
        """
        transformers = []
        for i, v in enumerate(self.booleans_list):
            if v == True:
                self.method = OneHotEncoder(
                    categories="auto", sparse=False, handle_unknown="ignore"
                )
            else:
                self.method = StandardScaler(with_mean=True, with_std=True)
            t = (f"input {i}", self.method, [i])
            transformers.append(t)

        self.ct = ColumnTransformer(transformers)
        self.ct.fit(X)

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """
        Applies the encoder.

        Parameters
        ----------
        X : npt.NDArray
            Input data

        Returns
        -------
        npt.NDArray
            Encoded data
        """
        return self.ct.transform(X).astype(dtype=np.float32)

    def get_input_type(self) -> Type:
        """
        Returns
        -------
        Type
            npt.NDArray[np.object_]
        """
        return npt.NDArray[np.object_]

    def get_output_type(self) -> Type:
        """
        Returns
        -------
        Type
            Float32Array
        """
        return Float32Array

    def forward(self, X: npt.NDArray[np.object_]) -> npt.NDArray:
        """
        Equivalent of `.predict()` or `.transform()`.


        Parameters
        ----------
        X : npt.NDArray[np.object_]
            _description_

        Returns
        -------
        npt.NDArray
            _description_
        """
        return self.transform(X)

    def to_onnx(self) -> SerializedModelTuple:
        """
        Serializes the encoder to onnx. 
        Each feature in the original dataset is mapped to its own input node (`float32` for numerical or string for `categorical`)

        Returns
        -------
        SerializedModelTuple
            Tuple of (Converted model serialized to string, number of input nodes, number of output nodes, list of initial types (one per input node), list of initial shapes (one per input node)).
        """
        initial_types = []
        initial_types_str: List[str] = []
        initial_shapes: List[List[Optional[int]]] = []
        for i, t in enumerate(self.booleans_list):
            if t == True:
                tensor = StringTensorType([None, 1])
                initial_types_str.append("STRING")
            else:
                tensor = FloatTensorType([None, 1])
                initial_types_str.append("FLOAT32")
            initial_types.append((f"input{i}", tensor))
            initial_shapes.append([None, 1])
        return (
            convert_sklearn(
                self.ct,
                initial_types=initial_types,
                target_opset=ONNX_OPSET_VERSION,
                options={StandardScaler: {"div": "div_cast"}},
            ).SerializeToString(),
            len(self.booleans_list),
            1,
            initial_types_str,
            initial_shapes,
        )
