from abc import abstractmethod
from typing import Any, Type, Union, Optional, List, Tuple
from falcon.abstract.model import Model
from falcon.abstract.onnx_convertible import ONNXConvertible
from falcon.serialization import SerializedModelRepr, serialize_to_falcon, serialize_to_onnx


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

    def forward(self, X: Any) -> Any:
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
        self.predict(X)

    def fit_pipe(self, X: Any, y: Any) -> Any:
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

    def __init__(self, task: str, **kwargs: Any) -> None:
        self.task = task
        self._pipeline: List[PipelineElement] = []

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

    def save(self, format: str = "auto") -> Tuple[bytes, str]:
        """
        Serializes the model to string. For more details please refer to the documentation of `TaskManager.save_model(...)` method.

        Parameters
        ----------
        format : str, optional
           `auto`, `falcon` or `onnx`, by default `auto`

        Returns
        -------
        Tuple[SerializedModelRepr, str]
            Serialized model, format


        """
        if format not in {"falcon", "onnx"}:
            raise ValueError(
                f"expected one of [onnx, falcon] as output format, got {format}"
            )
        serialized_pipeline_elements: List[SerializedModelRepr] = []
        for p in self._pipeline:
            if isinstance(p, ONNXConvertible):
                serialized_pipeline_elements.append(p.to_onnx())
            else:
                if format == "onnx":
                    raise RuntimeError("This model cannot be serialized to onnx")
                elif format == "auto":
                    format = "falcon"
        if format == "falcon":
            serialized_model = serialize_to_falcon(serialized_pipeline_elements)
        else:
            format = "onnx"
            serialized_model = serialize_to_onnx(
                serialized_pipeline_elements
            ).SerializeToString()
        return serialized_model, format
