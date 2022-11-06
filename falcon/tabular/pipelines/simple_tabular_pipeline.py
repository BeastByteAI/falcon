from numpy import typing as npt
import numpy.typing as npt
from typing import List, Any, Optional, Dict, Type
from falcon.abstract import Pipeline, PipelineElement
from falcon.abstract.learner import Learner
from falcon.abstract.onnx_convertible import ONNXConvertible
from falcon.tabular.processors.label_decoder import LabelDecoder
from falcon.tabular.processors.scaler_and_encoder import ScalerAndEncoder
from falcon.tabular.learners.super_learner import SuperLearner
from falcon.utils import print_

class SimpleTabularPipeline(Pipeline):
    """
    Default tabular pipeline.
    """
    def __init__(self, task: str, mask: List[int], learner: Type[Learner] = SuperLearner, learner_kwargs: Optional[Dict] = None):
        """
        Default tabular pipeline. On a high level it simply chains a preprocessor and model learner (by default `SuperLearner`). 
        For classification tasks, the labels are also encoded as integers (while predictions are decoded back to strings).
        Internally, all numerical features are scaled to 0 mean and 1 std. All categorical features are one-hot encoded (this approach might not be suitable for features with very high cardinality). 

        Parameters
        ----------
        task : str
            `tabular_classification` or `tabular_regression`
        mask : List[int]
            list of ints where 1/2 indicates a low/high cardinality categorical feature and 0 indicates a numerical feature
        learner : Learner, optional
            learner class to be used, by default `SuperLearner`
        learner_kwargs : Optional[Dict], optional
            arguments to be passed to the learner, by default None
        """
        super().__init__(task=task)
        
        encoder: PipelineElement = ScalerAndEncoder(mask)
        self.add_element(encoder)
        
        if not learner_kwargs:
            learner_kwargs = {}
        learner_: PipelineElement = learner(task=task, **learner_kwargs)
        self.add_element(learner_)

        if self.task == "tabular_classification":
            self.labels_transformer: LabelDecoder = LabelDecoder()
            self.add_element(self.labels_transformer)

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> None:
        """
        Fits the pipeline by consecutively calling `.fit_pipe()` method of each element in pipeline.
        For tabular classification, `LabelDecoder` is applied to targets before actual training occurs.

        Parameters
        ----------
        X : npt.NDArray
            train featrues
        y : npt.NDArray
            train targets
        """
        print_('Fitting the pipeline...')
        if self.task == "tabular_classification":
            self.labels_transformer.fit(y)
            y = self.labels_transformer.transform(y, inverse=False)
        for p in self._pipeline:
            p.fit_pipe(X, y)
            X = p.forward(X)

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """
        Predicts the label of passed data points.

        Parameters
        ----------
        X : npt.NDArray
            features

        Returns
        -------
        npt.NDArray
            predicted label
        """
        for p in self._pipeline:
            X = p.forward(X)
        return X

    

    