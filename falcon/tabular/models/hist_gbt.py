from sklearn.ensemble import (
    HistGradientBoostingClassifier as SklearnHistGradientBoostingClassifier,
    HistGradientBoostingRegressor as SklearnHistGradientBoostingRegressor,
)
from typing import Callable, Dict, Any, Union
from falcon.abstract import Model, ONNXConvertible
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import TensorType, FloatTensorType
from falcon.types import Float32Array, Int64Array
from falcon.serialization import SerializedModelRepr
from falcon.config import ONNX_OPSET_VERSION, ML_ONNX_OPSET_VERSION
from falcon.abstract.optuna import OptunaMixin
from numpy import typing as npt

class _BaseHistGradientBoosting(Model, ONNXConvertible, OptunaMixin):
    def __init__(self, estimator: Union[SklearnHistGradientBoostingClassifier, SklearnHistGradientBoostingRegressor], **kwargs: Any):
        self.estimator = estimator
    
    def fit(self, X: Float32Array, y: Float32Array, *args: Any, **kwargs: Any) -> None:
        """
        Fits the model

        Parameters
        ----------
        X : Float32Array
            Features
        y : Float32Array
            targets
        """
        self._shape = [None, *X.shape[1:]]
        self.estimator.fit(X, y)
    
    def to_onnx(self) -> SerializedModelRepr:
        """
        Serializes the model to onnx. 

        Returns
        -------
        SerializedModelRepr
        """
        initial_type = [("model_input", FloatTensorType(self._shape))]
        options = self._get_onnx_options()
        onnx_model = convert_sklearn(
            self.estimator,
            initial_types=initial_type,
            target_opset={'': ONNX_OPSET_VERSION, 'ai.onnx.ml': ML_ONNX_OPSET_VERSION},
            options=options,
        )
        n_inputs = len(onnx_model.graph.input)
        n_outputs = len(onnx_model.graph.output)

        return SerializedModelRepr(
            onnx_model,
            n_inputs,
            n_outputs,
            ["FLOAT32"],
            [self._shape]
        )

    def _get_onnx_options(self) -> Dict:
        return {}
    
    def predict(self, X: npt.NDArray, *args: Any, **kwargs: Any) -> npt.NDArray:
        return self.estimator.predict(X)
    
    @classmethod
    def get_search_space(cls, X: npt.NDArray, y: npt.NDArray) -> Union[Callable, Dict]:
        return { 
            "max_iter" : {
                "type": "int",
                "kwargs": {
                    "low": 50, 
                    "high": 500
                }
            },
            "min_samples_leaf" : {
                "type": "int",
                "kwargs": {
                    "low": 5,
                    "high": 20,
                    "step": 5
                }
            },
            "learning_rate" : {
                "type": "float",
                "kwargs": {
                    "low": 0.001,
                    "high": 1.,
                    "log": True
                }
            },
            "l2_regularization": {
                "type": "float",
                "kwargs": {
                    "low": 1e-7,
                    "high": 0.01,
                    "log": True
                }
            }
        }

class HistGradientBoostingRegressor(_BaseHistGradientBoosting):
    """
    Wrapper around `sklearn.ensemble.HistGradientBoostingRegressor`.
    """
    def __init__(self, max_iter: int = 100, min_samples_leaf: int = 20, learning_rate: float = 0.1, l2_regularization: float = 0., random_seed: int = 42, **kwargs: Any):
        """

        Parameters
        ----------
        max_iter : int, optional
            number of decision trees, by default 100
        min_samples_leaf : int, optional
            minimum number of samples per leaf, by default 20
        learning_rate : float, optional
            learning rate, by default 0.1
        l2_regularization : float, optional
            L2 regularization parameter, by default 0.0
        random_seed : int, optional
            by default 42
        """
        estimator = SklearnHistGradientBoostingRegressor(max_iter = max_iter, learning_rate=learning_rate, l2_regularization=l2_regularization, min_samples_leaf=min_samples_leaf, random_state=random_seed)
        super().__init__(estimator=estimator)

    

class HistGradientBoostingClassifier(_BaseHistGradientBoosting):
    """
    Wrapper around `sklearn.ensemble.HistGradientBoostingClassifier`.
    """
    def __init__(self, max_iter: int = 100, min_samples_leaf: int = 20, learning_rate: float = 0.1, l2_regularization: float = 0., random_seed: int = 42, **kwargs: Any):
        """

        Parameters
        ----------
        max_iter : int, optional
            number of decision trees, by default 100
        min_samples_leaf : int, optional
            minimum number of samples per leaf, by default 20
        learning_rate : float, optional
            learning rate, by default 0.1
        l2_regularization : float, optional
            L2 regularization parameter, by default 0.0
        random_seed : int, optional
            by default 42
        """
        estimator = SklearnHistGradientBoostingClassifier(max_iter = max_iter, learning_rate=learning_rate, l2_regularization=l2_regularization, min_samples_leaf=min_samples_leaf, random_state=random_seed)
        super().__init__(estimator=estimator)