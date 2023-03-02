from sklearn.base import (
    BaseEstimator as _BaseEstimator,
    ClassifierMixin as _ClassifierMixin,
    RegressorMixin as _RegressorMixin,
)
from typing import Optional, Union, Dict
import pandas as pd
from numpy import typing as npt
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from falcon.main import initialize
from falcon.task_configurations import get_task_configuration
from datetime import datetime


class _FalconBaseEstimator(_BaseEstimator):
    def __init__(
        self, config: Union[str, Dict] = "SuperLearner", make_eval_set: bool = False
    ) -> None:
        """
        Parameters
        ----------
        config : Union[str, Dict], optional
            configuration to be used, by default "SuperLearner"
        make_eval_set : bool, optional
            determines if an evaluation set should be created, by default False
        """
        self.config = config
        self.make_eval_set = make_eval_set
    
    def _get_tags(self) -> Dict:
        tags =  super()._get_tags()
        if "string" not in tags['X_types']:
            tags['X_types'].append("string")
        tags['non_deterministic'] = True
        return tags

    def _get_task_config(self) -> Dict:
        if isinstance(self.config, dict):
            config = self.config
        elif isinstance(self.config, str):
            config = get_task_configuration("tabular_classification", self.config)
        else: 
            raise ValueError("Invalid configuration")
        return config

    def predict(self, X: Union[pd.DataFrame, npt.NDArray]) -> npt.NDArray:
        check_is_fitted(self)
        X = check_array(X, dtype=None)
        y = self.manager_.predict(X)
        return y

    def save_model(self, filename: Optional[str]) -> None:
        """
        Saves model in onnx format

        Parameters
        ----------
        filename : str, optional
            filename of the saved model
        """
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d.%H%M%S")
            filename = f"falcon_{ts}.onnx"
        print("Saving the model ...")
        self.manager_.save_model(format="onnx", filename=filename)
        print(f"The model was saved as `{filename}`")


class FalconTabularClassifier(_FalconBaseEstimator, _ClassifierMixin):
    """
    Falcon sklearn wrapper to be used for tabular classification tasks.
    Alternatively, `FalconClassifier` can be used as an alias.
    """

    def fit(
        self, X: Union[pd.DataFrame, npt.NDArray], y: Union[pd.DataFrame, npt.NDArray]
    ) -> _FalconBaseEstimator:
        """
        Fits the classifier

        Parameters
        ----------
        X : Union[pd.DataFrame, npt.NDArray]
            data
        y : Union[pd.DataFrame, npt.NDArray]
            labels
        """
        X, y = check_X_y(X, y, dtype=None)
        self.classes_ = unique_labels(y)
        config = self._get_task_config()
        self.n_features_in_ = X.shape[1]
        self.manager_ = initialize(task="tabular_classification", data=(X, y), **config)
        self.manager_.train(pre_eval=False, make_eval_set=self.make_eval_set)
        self.manager_.performance_summary(None)
        return self


class FalconTabularRegressor(_FalconBaseEstimator, _RegressorMixin):
    """
    Falcon sklearn wrapper to be used for tabular regression tasks.
    Alternatively, `FalconRegressor` can be used as an alias.
    """

    def fit(
        self, X: Union[pd.DataFrame, npt.NDArray], y: Union[pd.DataFrame, npt.NDArray]
    ) -> _FalconBaseEstimator:
        """
        Fits the regressor

        Parameters
        ----------
        X : Union[pd.DataFrame, npt.NDArray]
            data
        y : Union[pd.DataFrame, npt.NDArray]
            labels
        """
        X, y = check_X_y(X, y, dtype=None)
        config = self._get_task_config()
        self.n_features_in_ = X.shape[1]
        self.manager_ = initialize(task="tabular_regression", data=(X, y), **config)
        self.manager_.train(pre_eval=False, make_eval_set=self.make_eval_set)
        self.manager_.performance_summary(None)
        return self


FalconClassifier = FalconTabularClassifier
FalconRegressor = FalconTabularRegressor
