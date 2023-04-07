import numpy as np
from typing import Dict, Any, Type, List, Tuple, Optional
from falcon.types import Float32Array, Int64Array
from falcon.abstract.task_pipeline import Pipeline
from falcon.abstract.learner import Learner
from falcon.abstract.onnx_convertible import ONNXConvertible
from falcon.serialization import SerializedModelRepr
from falcon.tabular.adapters.ts.auxiliary import _wrap_onnx


class TSAdapterLearner(Learner, ONNXConvertible):
    def __init__(
        self,
        task: str,
        dataset_size: Tuple[int, ...],
        mask: List,
        wrapped_pipeline: Type[Pipeline],
        wrapped_pipeline_options: Dict,
    ) -> None:
        self.task = task
        self.mask = mask
        self.dataset_size = dataset_size

        self.wrapped_pipeline = wrapped_pipeline
        self.wrapped_pipeline_options = wrapped_pipeline_options
        self.data_shape: List[Optional[int]] = []
        self.n_inputs: int = 1

    def fit(self, X: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> None:
        mean = np.mean(X, axis=1).reshape(-1, 1)
        y = y.reshape(-1, 1)
        self._pipeline = self.wrapped_pipeline(
            task=self.task,
            mask=self.mask,
            dataset_size=self.dataset_size,
            **self.wrapped_pipeline_options
        )
        self.data_shape = [None, X.shape[1]]
        self._pipeline.fit(X - mean, y - mean)

    def predict(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        mean = np.mean(X, axis=1).reshape(-1, 1)
        pred = self._pipeline.predict(X - mean).reshape(-1, 1)
        pred = pred + mean
        return pred.squeeze(1)

    def to_onnx(self) -> SerializedModelRepr:
        onx = self._pipeline.save()
        _wrap_onnx(onx)
        _last_axis: int = (
            self.data_shape[1] if self.data_shape[1] else 1
        )  # just for type compatibility
        sm = SerializedModelRepr(
            model=onx,
            n_inputs=_last_axis,
            n_outputs=1,
            initial_types=["float32" for _ in range(_last_axis)],
            initial_shapes=[self.data_shape],
        )
        return sm

    def get_input_type(self) -> Type:
        """
        Returns
        -------
        Type
            Float32Array
        """
        return Float32Array

    def get_output_type(self) -> Type:
        """
        Returns
        -------
        Type
            Float32Array for regression, Int64Array for classification
        """
        return Float32Array if self.task == "tabular_regression" else Int64Array
