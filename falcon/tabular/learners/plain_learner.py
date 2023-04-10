from falcon.abstract import Learner
from falcon.abstract import Model, ONNXConvertible
from typing import Type, Optional, Any, Dict, Union
from falcon.tabular.models.hist_gbt import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from falcon.types import Float32Array, Int64Array
from imblearn.over_sampling import RandomOverSampler
from falcon.serialization import SerializedModelRepr

class PlainLearner(Learner, ONNXConvertible):
    """
    PlainLearner trains a model using provided or default hyperparameters.
    """
    def __init__(self, task: str, model_class: Optional[Type] = None, hyperparameters: Optional[Dict] = None, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        task : str
            'tabular_classification' or 'tabular_regression'
        model_class : Optional[Type], optional
            the class of the model to train, by default None;
            if None, HistGradientBoosting is used
        hyperparameters: Dict, optional
            the dictionary of hyperparameters for model training
        """
        self.task = task
        self.model_class: Type[Any]
        self.hyperparameters = hyperparameters if hyperparameters else {}
        if model_class is None: 
            if task == "tabular_classification": 
                self.model_class = HistGradientBoostingClassifier
            elif task == "tabular_regression":
                self.model_class = HistGradientBoostingRegressor 
            else:
                ValueError("Not supported task")
        else: 
            self.model_class = model_class
        if not issubclass(self.model_class, Model):
            raise ValueError('Model class should be a subclass of falcon.abstract.Model')
        if not issubclass(self.model_class, ONNXConvertible):
            raise ValueError('PlainLearner only supports ONNXConvertible models')

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

    def fit(self, X: Float32Array, y: Float32Array, *args: Any, **kwargs: Any) -> None:
        """
        Fits the model and trains the final model using them.
        For classification tasks, the dataset will be balanced by upsampling the minority class(es).

        Parameters
        ----------
        X : Float32Array
            features
        y : Float32Array
            targets
        """

        if self.task == 'tabular_classification':
            X, y = RandomOverSampler().fit_resample(
                X, y
            )
        model = self.model_class(**self.hyperparameters)
        model.fit(X,y)
        self.model = model
    
    def predict(self, X: Float32Array, *args: Any, **kwargs: Any) -> Union[Float32Array, Int64Array]:
        return self.model.predict(X)
    
    # didn't work without forward method
    
    
    def fit_pipe(self, X: Float32Array, y: Float32Array, *args: Any, **kwargs: Any) -> None:
        """
        Equivalent to `.fit(X, y)`

        Parameters
        ----------
        X : Float32Array
            features
        y : Float32Array
            targets
        """
        self.fit(X, y)
    
    def to_onnx(self) -> SerializedModelRepr:
        """
        Serializes the underlying model to onnx by calling its `.to_onnx()` method.

        Returns
        -------
        SerializedModelRepr
        """
        return self.model.to_onnx()
        
    