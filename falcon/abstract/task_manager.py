from __future__ import annotations
from abc import ABC, abstractmethod
from numpy import typing as npt
from .task_pipeline import Pipeline
from typing import Dict, Optional, Any, Callable, Type
from falcon.serialization import SerializedModelRepr

class TaskManager(ABC):
    """
    Base class for all Task Managers.
    """
    def __init__(
        self,
        task: str,
        data: Any,
        pipeline: Optional[Type[Pipeline]] = None,
        pipeline_options: Optional[Dict] = None,
        extra_pipeline_options: Optional[Dict] = None,
        features: Any = None,
        target: Any = None,
    ):
        """

        Parameters
        ----------
        task : str
            current task
        data : Any
            data to be used for training
        pipeline : Optional[Type[Pipeline]], optional
            pipeline class to be used, by default None
        pipeline_options : Optional[Dict], optional
            arguments to be passed to pipeline instead of default ones, by default None
        extra_pipeline_options : Optional[Dict], optional
            arguments to be passed to pipeline in addition to default ones, by default None
        features : Any, optional
            featrues to be used for training, by default None
        target : Any, optional
            targets to be used for training, by default None
        """
        self.task: str = task
        self.features = features
        self.target = target
        self._data = self._prepare_data(data)
        self._extra_pipeline_options = extra_pipeline_options
        self._create_pipeline(pipeline=pipeline, options=pipeline_options)

    @abstractmethod
    def train(self, **kwargs: Any) -> TaskManager:
        """
        Trains the underlying pipeline.

        Returns
        -------
        TaskManager
            self
        """
        pass

    @abstractmethod
    def _prepare_data(self, data: Any) -> Any:
        """
        Initial data preparation (e.g. reading from file).
        Warning: initial data preparation (e.g. reading, cleaning) and data preprocessing (e.g. scaling, encoding) are two distinct steps. The later one is performed inside the pipeline.

        Parameters
        ----------
        data : Any
            training data

        Returns
        -------
        Any
            prepared data
        """
        pass

    @property
    @abstractmethod
    def default_pipeline(self) -> Type[Pipeline]:
        """
        Default pipeline class. Can be chosen dynamically.
        """
        pass

    @property
    def default_pipeline_options(self) -> Dict:
        """
        Default pipeline options. Can be chosen dynamically.
        """
        pass

    def predict(self, X: Any) -> Any:
        """
        Calls predict methods of the pipeline.

        Parameters
        ----------
        X : Any
            features

        Returns
        -------
        Any
            predictions
        """
        return self._pipeline.predict(X)

    def _create_pipeline(
        self, pipeline: Optional[Type[Pipeline]], options: Optional[Dict]
    ) -> None:
        """
        Initializes the pipeline.

        Parameters
        ----------
        pipeline : Optional[Type[Pipeline]]
            pipeline class
        options : Optional[Dict]
            pipeline options
        """

        # if pipeline is not None and options is None:
        #     self._pipeline = pipeline(task=self.task)
        if pipeline is None:
            pipeline = self.default_pipeline
        if options is None:
            options = self.default_pipeline_options
            if self._extra_pipeline_options is not None:
                for k, v in self._extra_pipeline_options.items():
                    options[k] = v
        self._pipeline = pipeline(task=self.task, **options)

    def save_model(self, format: str = "auto", filename: Optional[str] = None) -> bytes:
        """
        Serializes and saves the model.

        Parameters
        ----------
        format : str, optional
            "auto", "onnx" or "falcon"; "falcon" format should only be used in rare cases when converting to onnx is not possible, by default "auto"
        filename : Optional[str], optional
            filename for the model file, by default None. If filename is not specified, the model is not saved on disk and only returned as bytes object

        Returns
        -------
        bytes
            serialized model as bytes
        """
        if format not in {"falcon", "onnx", "auto"}:
            raise ValueError(
                f"expected one of [onnx, falcon] as output format, got {format}"
            )
        serialized_model, format = self._pipeline.save(format=format)
        if filename is not None:
            if not filename.endswith(f".{format}"):
                filename += f".{format}"
            with open(filename, "wb+") as f:
                f.write(serialized_model)
        return serialized_model

    @abstractmethod
    def evaluate(self, test_data: Any) -> Any:
        """
        Evaluates the performance of a trained pipeline.

        Parameters
        ----------
        test_data : Any
            data to be used for evaluation
            

        Returns
        -------
        Any
            evaluation metric or None
        """
        pass

    @abstractmethod
    def performance_summary(self, test_data: Any) -> Any: 
        """
        Prints the performance summary of the trained pipeline.
    

        Parameters
        ----------
        test_data : Any
            test set, optional

        Returns
        -------
        Any
            relevant metrics or None
        """
        pass