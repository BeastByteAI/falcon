import numpy as np
from numpy import typing as npt
from falcon.types import Float32Array, ColumnTypes
from sklearn.base import BaseEstimator
from sklearn import __version__ as sklearn_version
from packaging import version
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

    def __init__(self, mask: List[ColumnTypes]) -> None:
        """
        Parameters
        ----------
        mask : List[ColumnTypes]
            provides a type for each column at a given index
        """
        self.mask = mask

    def _get_ohe(self) -> BaseEstimator:
        if version.parse(sklearn_version) < version.parse("1.2.0"):
            not_sparse = {"sparse": False}
        else:
            not_sparse = {"sparse_output": False}

        method = SKLPipeline(
            steps=[
                ("cast_str", CastTransformer(dtype=np.str_)),
                (
                    "ohe",
                    OneHotEncoder(
                        categories="auto", handle_unknown="ignore", **not_sparse
                    ),
                ),
            ]
        )
        return method

    def _get_numeric_scaler(self) -> BaseEstimator:
        return SKLPipeline(
            steps=[
                ("cast64", CastTransformer(dtype=np.float64)),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("cast32", CastTransformer()),
            ]
        )

    def _get_ordinal_encoder(self) -> BaseEstimator:
        return SKLPipeline(
            steps=[
                ("cast_str", CastTransformer(dtype=np.str_)),
                (
                    "ord_enc",
                    OrdinalEncoder(
                        categories="auto",
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                ("sc", MaxAbsScaler()),
            ]
        )

    def fit(self, X: npt.NDArray, y: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Fits the encoder.

        Parameters
        ----------
        X : npt.NDArray
            data to encode
        _ : Any, optional
            dummy argument to keep compatibility with pipeline training
        """
        transformers = []

        for i, v in enumerate(self.mask):
            if v == ColumnTypes.CAT_LOW_CARD:
                method = self._get_ohe()
            elif v == ColumnTypes.NUMERIC_REGULAR:
                method = self._get_numeric_scaler()
            else:
                method = self._get_ordinal_encoder()
            t = (f"input {i}", method, [i])
            transformers.append(t)

        self.ct = ColumnTransformer(transformers)
        self.ct.fit(X)

    def predict(self, X: npt.NDArray, *args: Any, **kwargs: Any) -> npt.NDArray:
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
            object
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

    def forward(
        self, X: npt.NDArray[np.object_], *args: Any, **kwargs: Any
    ) -> npt.NDArray:
        """
        Equivalent of `.predict()` or `.transform()`.


        Parameters
        ----------
        X : npt.NDArray[object]
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
            if t in [ColumnTypes.NUMERIC_REGULAR]:
                tensor = FloatTensorType([None, 1])
                initial_types_str.append("FLOAT32")
            else:
                tensor = StringTensorType([None, 1])
                initial_types_str.append("STRING")
            initial_types.append((f"input{i}", tensor))
            initial_shapes.append([None, 1])
        return SerializedModelRepr(
            convert_sklearn(
                self.ct,
                initial_types=initial_types,
                target_opset={
                    "": ONNX_OPSET_VERSION,
                    "ai.onnx.ml": ML_ONNX_OPSET_VERSION,
                },
                options={StandardScaler: {"div": "div_cast"}},
            ),
            len(self.mask),
            1,
            initial_types_str,
            initial_shapes,
        )
