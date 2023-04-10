from sklearn.ensemble import (
    StackingRegressor as SklearnStackingRegressor,
    StackingClassifier as SklearnStackingClassifier,
)
from falcon.abstract.onnx_convertible import ONNXConvertible
from falcon.addons.sklearn import (
    BalancedStackingClassifier as SklearnBalancedStackingClassifier,
)
from falcon.abstract import Model
from falcon.types import Float32Array
from sklearn.base import BaseEstimator
from typing import Any, List, Optional, Union, Dict, Tuple, Callable, Type
from numpy import typing as npt
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import TensorType, FloatTensorType
from falcon.config import ONNX_OPSET_VERSION, ML_ONNX_OPSET_VERSION
from falcon.serialization import SerializedModelRepr


class _StackingBase(Model, ONNXConvertible):
    def __init__(
        self,
        cls: Callable,
        estimators: List[Tuple[str, BaseEstimator]],
        final_estimator: BaseEstimator,
        cv: Any = None,
        n_jobs: int = 1,
        passthrough: bool = False,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        self.estimator: BaseEstimator = cls(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=n_jobs,
            passthrough=passthrough,
            **kwargs,
        )

        self._ret_type: Type = (
            np.float32
            if isinstance(self.estimator, SklearnStackingRegressor)
            else np.int64
        )

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

    def predict(self, X: Float32Array, *args: Any, **kwargs: Any) -> Float32Array:
        """
        Predicts the target for the given input

        Parameters
        ----------
        X : Float32Array
            featrues

        Returns
        -------
        Float32Array
            predictions
        """
        prediction: npt.NDArray = self.estimator.predict(X)
        prediction = prediction.astype(dtype=self._ret_type)
        return prediction

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
            # final_types=self._get_onnx_final_types(),
            options=options,
        )
        n_inputs = len(onnx_model.graph.input)
        n_outputs = len(onnx_model.graph.output)

        return SerializedModelRepr(
            onnx_model,
            n_inputs,
            n_outputs,
            ["FLOAT32"],
            [self._shape],
        )

    # def _get_onnx_final_types(self) -> List[Tuple[str, TensorType]]:
    #     return [("stacking_output", FloatTensorType([None, 1]))]

    def _get_onnx_options(self) -> Dict:
        return {}


class StackingClassifier(_StackingBase):
    """
    Small wrapper around `sklearn.ensemble.StackingClassifier`. 
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        final_estimator: BaseEstimator,
        balanced: bool = True,
        cv: Any = 5,
        n_jobs: int = 1,
        passthrough: bool = False,
        verbose: int = 0,
        stack_method: Any = "auto",
        **kwargs: Any,
    ) -> None:
        """
        Small wrapper around `sklearn.ensemble.StackingClassifier`. 
        For more detailed description please refer to sklarn documentation. 


        Parameters
        ----------
        estimators : List[Tuple[str, BaseEstimator]]
            base estimators
        final_estimator : BaseEstimator
            meta estimator, by default LogisticRegression
        balanced : bool, optional
            if True, the classes are balanced by performing random oversampling, by default True
        cv : Any, optional
            number of CV folds, or custom CV object, by default 5
        n_jobs : int, optional
            number of parallel jobs, by default -1
        passthrough : bool, optional
            when True the meta estimator is trained on original data in addition to the predictions of base estimators, by default False
        verbose : int, optional
            verbosity level of underlying sklearn estimator, by default 0
        stack_method : Any, optional
            methods called for each base estimator, by default "auto"
        """
        cls: Callable
        if balanced:
            cls = SklearnBalancedStackingClassifier
        else:
            cls = SklearnStackingClassifier
        super().__init__(
            cls=cls,
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose,
            stack_method=stack_method,
            **kwargs,
        )

    # def _get_onnx_final_types(
    #     self,
    # ) -> List[Tuple[str, TensorType]]:
    #     return [
    #         ("stacking_labels", FloatTensorType([None])),
    #         ("stacking_probs", FloatTensorType([None, None])),
    #     ]

    def _get_onnx_options(self) -> Dict:
        return {id(self.estimator): {"zipmap": False}}


class StackingRegressor(_StackingBase):
    """
    Small wrapper around `sklearn.ensemble.StackingRegressor`. 
    """
    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        final_estimator: BaseEstimator,
        cv: Any = 5,
        n_jobs: int = 1,
        passthrough: bool = False,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Small wrapper around `sklearn.ensemble.StackingRegressor`. 
        For more detailed description please refer to sklarn documentation. 

        Parameters
        ----------
        estimators : List[Tuple[str, BaseEstimator]]
            base estimators
        final_estimator : BaseEstimator
            meta estimator
        cv : Any, optional
            number of CV folds, or custom CV object, by default 5
        n_jobs : int, optional
            number of parallel jobs, by default -1
        passthrough : bool, optional
            when True the meta estimator is trained on original data in addition to the predictions of base estimators, by default False
        verbose : int, optional
            verbosity level of underlying sklearn estimator, by default 0
        """
        super().__init__(
            cls=SklearnStackingRegressor,
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose,
            **kwargs,
        )
