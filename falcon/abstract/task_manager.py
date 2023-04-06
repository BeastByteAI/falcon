from __future__ import annotations
from abc import ABC, abstractmethod
from numpy import typing as npt
from .task_pipeline import Pipeline
from typing import Dict, Optional, Any, Callable, Type, List
from falcon.serialization import SerializedModelRepr
from onnx import ModelProto
from onnx import save_model as onnx_save_model


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
        self.dataset_size = ()
        self.feature_names_to_save: List[Any] = []
        self._data = self._prepare_data(data)
        if self.dataset_size is None: 
            raise RuntimeError('It seems like prepare_data() method did not set dataset_size attribute.')
        self._extra_pipeline_options: Optional[Dict] = extra_pipeline_options
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
    @abstractmethod
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
        self._pipeline: Pipeline = pipeline(task=self.task, dataset_size = self.dataset_size, **options)

    def save_model(self, filename: Optional[str] = None, **kwargs: Any) -> ModelProto:
        """
        Serializes and saves the model.

        Parameters
        ----------
        filename : Optional[str], optional
            filename for the model file, by default None. If filename is not specified, the model is not saved on disk and only returned as bytes object
        Returns
        -------
        ModelProto
            ONNX ModelProto of the model
        """

        serialized_model = self._pipeline.save(feature_names=self.feature_names_to_save)
        if filename is not None:
            if not filename.endswith(f".onnx"):
                filename += f".onnx"
            onnx_save_model(serialized_model, filename, save_as_external_data=True, all_tensors_to_one_file=True, location=f"{filename}.tensors", size_threshold=0, convert_attribute=True)
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