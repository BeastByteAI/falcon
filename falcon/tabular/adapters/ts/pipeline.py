import numpy as np
from numpy import typing as npt
from typing import Dict, Type, List, Any, Tuple
from falcon.abstract.task_pipeline import Pipeline
from falcon.abstract.learner import Learner
from falcon.abstract.onnx_convertible import ONNXConvertible
from falcon.tabular.adapters.ts.learner import TSAdapterLearner


class TSAdapterPipeline(Pipeline):
    def __init__(
        self,
        task: str,
        dataset_size: Tuple[int],
        mask: List[Any],
        wrapped_pipeline: Type[Pipeline],
        wrapped_pipeline_options: Dict,
        **kwargs: Any
    ) -> None:
        super().__init__(task, dataset_size, mask)
        self.wrapped_pipeline = wrapped_pipeline
        self.wrapped_pipeline_options = wrapped_pipeline_options

    def fit(self, X: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> None:
        self._pipeline = []
        learner = TSAdapterLearner(
            self.task,
            self.dataset_size,
            self.mask,
            self.wrapped_pipeline,
            self.wrapped_pipeline_options,
        )
        self.add_element(learner)
        learner.fit(X, y)

    def predict(self, X: npt.NDArray, *args: Any, **kwargs: Any) -> npt.NDArray:
        return self._pipeline[0].predict(X)
