import numpy as np
from numpy import typing as npt
from falcon.types import Float32Array
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from falcon.abstract import Processor, ONNXConvertible
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from falcon.config import ONNX_OPSET_VERSION, ML_ONNX_OPSET_VERSION
from typing import List, Optional, Type, Any
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.preprocessing import MaxAbsScaler, OrdinalEncoder
from skl2onnx.sklapi import CastTransformer
from falcon.serialization import SerializedModelRepr

class ScalerAndEncoder(Processor, ONNXConvertible):
    """
    Applies OneHotEncoder/OrdinalEncoder on low/high cardinality categorical features and StandardScaler on numerical features.
    """
    
    def __init__(
        self, mask: List[int]
    ) -> None:  # [1/2 -> low/high cardinality categorical, 0 -> numerical]
        """
        Parameters
        ----------
        mask : List[int]
            boolean mask with True/False for categorical/numerical features
        """
        self.mask = mask

    def fit(self, X: npt.NDArray, _: Any = None) -> None:
        """
        Fits the encoder.

        Parameters
        ----------
        X : npt.NDArray
            data to encode
        _ : Any, optional
            dummy argument to keep compatibility with pipeline training, by default None
        """
        transformers = []
        for i, v in enumerate(self.mask):
            if v == 1:
                method = OneHotEncoder(
                    categories="auto", sparse=False, handle_unknown="ignore"
                )
            elif v == 0:
                method = SKLPipeline(steps = [('cast64', CastTransformer(dtype=np.float64)),('scaler', StandardScaler(with_mean=True, with_std=True)),('cast32', CastTransformer())])
            else:
                method = SKLPipeline(steps=[('ord_enc', OrdinalEncoder(categories="auto", handle_unknown = "use_encoded_value", unknown_value = -1)), ('sc', MaxAbsScaler())])
            
            t = (f"input {i}", method, [i])
            transformers.append(t)

        self.ct = ColumnTransformer(transformers)
        self.ct.fit(X)

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """
        Applies the encoder.

        Parameters
        ----------
        X : npt.NDArray
            input data

        Returns
        -------
        npt.NDArray
            encoded data
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
            data to process

        Returns
        -------
        npt.NDArray
            processed data
        """
        return self.transform(X)

    def to_onnx(self) -> SerializedModelRepr:
        """
        Serializes the encoder to onnx. 
        Each feature in the original dataset is mapped to its own input node (`float32` for numerical or `string` for categorical).

        Returns
        -------
        SerializedModelRepr
        """
        initial_types = []
        initial_types_str: List[str] = []
        initial_shapes: List[List[Optional[int]]] = []
        for i, t in enumerate(self.mask):
            if t == 1 or t == 2:
                tensor = StringTensorType([None, 1])
                initial_types_str.append("STRING")
            else:
                tensor = FloatTensorType([None, 1])
                initial_types_str.append("FLOAT32")
            initial_types.append((f"input{i}", tensor))
            initial_shapes.append([None, 1])
        return SerializedModelRepr(
            convert_sklearn(
                self.ct,
                initial_types=initial_types,
                target_opset={'': ONNX_OPSET_VERSION, 'ai.onnx.ml': ML_ONNX_OPSET_VERSION},
                options={StandardScaler: {"div": "div_cast"}},
            ).SerializeToString(),
            len(self.mask),
            1,
            initial_types_str,
            initial_shapes,
        )
