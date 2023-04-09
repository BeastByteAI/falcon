from typing import Type, Any
import numpy as np
from numpy import typing as npt
from sklearn.base import (
    BaseEstimator as _BaseEstimator,
    RegressorMixin as _RegressorMixin,
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import TensorType, FloatTensorType
from falcon.abstract.model import Model as _Model
from falcon.abstract.onnx_convertible import ONNXConvertible as _ONNXConvertible
from falcon.types import Float32Array, Int64Array
from falcon.serialization import SerializedModelRepr as _SerializedModelRepr
from falcon.config import ONNX_OPSET_VERSION, ML_ONNX_OPSET_VERSION

class SklearnRegressorWrapper(_Model, _ONNXConvertible):
    def __init__(self, regressor: Type[_RegressorMixin], **kwargs: Any):
        if not issubclass(regressor, _RegressorMixin):
            raise TypeError(
                "Regressor must be a subclass of sklearn.base.RegressorMixin"
            )
        self._regressor = regressor
        self._kwargs = kwargs

    def fit(self, X: Any, y: Any, *args: Any, **kwargs: Any) -> None:
        self._model = self._regressor(**self._kwargs)
        self._model.fit(X, y)
        self._shape = [None, *X.shape[1:]]

    def predict(self, X: npt.NDArray, *args: Any, **kwargs: Any) -> npt.NDArray:
        return self._model.predict(X).astype(np.float32)

    def to_onnx(self) -> _SerializedModelRepr:
        """
        Serializes the model to onnx.

        Returns
        -------
        SerializedModelRepr
        """
        initial_type = [("model_input", FloatTensorType(self._shape))]
        onnx_model = convert_sklearn(
            self._model,
            initial_types=initial_type,
            target_opset={"": ONNX_OPSET_VERSION, "ai.onnx.ml": ML_ONNX_OPSET_VERSION},
        )
        n_inputs = len(onnx_model.graph.input)
        n_outputs = len(onnx_model.graph.output)

        return _SerializedModelRepr(
            onnx_model, n_inputs, n_outputs, ["FLOAT32"], [self._shape]
        )