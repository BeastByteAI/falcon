from abc import abstractmethod
from typing import Any, Type, Union, Optional, List, Tuple
from onnx import ModelProto
from falcon.abstract.model import Model
from falcon.abstract.onnx_convertible import ONNXConvertible
from falcon.serialization import SerializedModelRepr, serialize_to_onnx


class PipelineElement(Model):
    """
    Base class for all pipeline elements.
    """

    @abstractmethod
    def get_input_type(self) -> Type:
        """
        Returns
        -------
        Type
            Input types
        """
        pass

    @abstractmethod
    def get_output_type(self) -> Type:
        """
        Returns
        -------
        Type
            Output types
        """
        pass

    def forward(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Equivalent of `predict` method that is used for elements chaining inside pipeline during inference.

        Parameters
        ----------
        X : Any
            featrues

        Returns
        -------
        Any
            predictions
        """
        return self.predict(X)

    def fit_pipe(self, X: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Equivalent of `fit` method that is used for elements chaining inisde pipeline during training.

        Parameters
        ----------
        X : Any
            features
        y : Any
            targets

        Returns
        -------
        Any
            usually None
        """
        self.fit(X, y)


class Pipeline(Model):
    """
    Base class for all pipelines.
    """

    def __init__(
        self, task: str, dataset_size: Tuple[int, ...], mask: List[Any], **kwargs: Any
    ) -> None:
        self.task = task
        self._pipeline: List[PipelineElement] = []
        self.dataset_size = dataset_size
        self.mask = mask

    def add_element(self, element: PipelineElement) -> None:
        """
        Adds element to pipeline. The input type of added element should match the output type of the last element in the pipeline.

        Parameters
        ----------
        element : PipelineElement
            element to be added to the end of the pipeline
        """
        if (
            len(self._pipeline) > 1
            and element.get_input_type() != self._pipeline[-1].get_output_type()
        ):
            raise RuntimeError(
                "The element cannot be added to pipeline due to input type missmatch."
            )
        if element is self:
            raise ValueError("Cannot add self to the pipeline")
        self._pipeline.append(element)

    def save(self, feature_names: Optional[List] = None) -> ModelProto:
        """
        Exports the pipeline to ONNX ModelProto

        Parameters
        ----------
        feature_names : Optional[List], optional
            feature names, by default None
        Returns
        -------
        ModelProto
            Pipeline as ONNX ModelProto
        """
        serialized_pipeline_elements: List[SerializedModelRepr] = []
        for p in self._pipeline:
            if isinstance(p, ONNXConvertible):
                serialized_pipeline_elements.append(p.to_onnx())
            else:
                raise RuntimeError("Encountered non convertible pipeline element")

        serialized_model = serialize_to_onnx(
            serialized_pipeline_elements,
            task=self.task,
            init_types=self.mask,
            init_feature_names=feature_names,
        )
        return serialized_model
