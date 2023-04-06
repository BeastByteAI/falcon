from __future__ import annotations
from falcon.abstract import TaskManager, Pipeline
from falcon.tabular.pipelines.simple_tabular_pipeline import SimpleTabularPipeline
from falcon.tabular.utils import convert_to_np_obj
from .reporting import print_classification_report, print_regression_report
from falcon.tabular.utils import *
from falcon.type_guessing import determine_column_types
from falcon import types as ft
from falcon.types import ColumnTypes
from typing import Union, Optional, List, Tuple, Type, Dict, Any
from numpy import typing as npt
import pandas as pd
from falcon.utils import print_, set_verbosity_level
from sklearn.model_selection import train_test_split
import os
import pandas as pd


class TabularTaskManager(TaskManager):
    """
    Default task manager for tabular data.
    """

    def __init__(
        self,
        task: str,
        data: Union[str, npt.NDArray, pd.DataFrame, Tuple],
        pipeline: Optional[Type[Pipeline]] = None,
        pipeline_options: Optional[Dict] = None,
        extra_pipeline_options: Optional[Dict] = None,
        features: Optional[ft.ColumnsList] = None,
        target: Optional[Union[str, int]] = None,
        eval_strategy: Optional[Union[str, BaseCrossValidator, Callable]] = "auto",
        **options: Any,
    ) -> None:
        """

        Parameters
        ----------
        task : str
            `tabular_classification` or `tabular_regression`
        data : Union[str, npt.NDArray, pd.DataFrame, Tuple]
            path to data file or pandas dataframe or numpy array or tuple (X,y)
        pipeline: Optional[Type[Pipeline]]
            class to be used as pipeline, by default None.
            If None, `SimpleTabularPipeline` will be used
        pipeline_options : Optional[Dict], optional
            arguments to be passed to the pipeline, by default None.
            These options will overwrite the ones from `default_pipeline_options` attribute.
        extra_pipeline_options : Optional[Dict], optional
            arguments to be passed to the pipeline, by default None.
            These options will be passed in addition to the ones from `default_pipeline_options` attribute.
            This argument is ignored if `pipeline_options` is not None
        features : Optional[ft.ColumnsList], optional
            names or indices of columns to be used as features, by default None.
            If None, all columns except the last one will be used.
            If `target` argument is not None, features should be passed explicitly as well
        target : Optional[Union[str, int]], optional
            name or index of column to be used as target, by default None.
            If None, the last column will be used as target.
            If `features` argument is not None, target should be specified explicitly as well
        eval_strategy : Optional[Union[str, BaseCrossValidator, Callable]], optional
            evaluation strategy, can be one of {'auto', 'holdout' 'cv', BaseCrossValidator, Callable} by default 'auto'.
            If 'auto', uses 5 fold CV for small datasets and holdout for large ones.
            If 'holdout', uses holdout strategy with 25% of data for validation.
            If 'cv', uses 5 fold CV.
            If BaseCrossValidator, uses the specified cross-validator.
            If Callable, uses the specified function to split data into train and validation sets.
            If None, no evaluation will be performed.
        """
        print_(f"\nInitializing a new TabularTaskManager for task `{task}`")
        self._data: Tuple[npt.NDArray, npt.NDArray, List[ColumnTypes]]
        # self._pipeline: Pipeline
        super().__init__(
            task=task,
            data=data,
            pipeline=pipeline,
            pipeline_options=pipeline_options,
            extra_pipeline_options=extra_pipeline_options,
            features=features,
            target=target,
        )

        self._eval_set: Optional[Tuple] = None
        self._stored_cv_score: Optional[Dict] = None
        self.eval_strategy = eval_strategy
        if not self._validate_eval_strategy():
            raise ValueError(
                f"Invalid value for `eval_strategy` argument: {self.eval_strategy}"
            )

    def _validate_eval_strategy(self) -> bool:
        if self.eval_strategy is None:
            return True
        if self.eval_strategy in ("auto", "cv", "holdout"):
            return True
        if isinstance(self.eval_strategy, BaseCrossValidator):
            return True
        if callable(self.eval_strategy):
            return True
        return False

    def _infer_feature_names(self, data: Any) -> None:
        print(type(data))
        if (
            isinstance(data, pd.DataFrame)
            and self.features is None
            and self.target is not None
        ):
            self.features: Optional[Union[List[str], List[int]]] = [c for c in data.columns if c != self.target]


    def _prepare_data(
        self, data: Union[str, npt.NDArray, pd.DataFrame, Tuple], training: bool = True
    ) -> Tuple[npt.NDArray, npt.NDArray, List[ColumnTypes]]:
        """
        Initial data preparation:
        1) optional: read data from the specified location;
        2) split into features and targets. By default it is assumed that the last column is the target;
        3) clean data;
        4) determine numerical and categorical features (create categorical mask).

        Parameters
        ----------
        data : Union[str, npt.NDArray, pd.DataFrame, Tuple]
            path to data file or pandas dataframe or numpy array or Tuple(X,y)

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray, List[ColumnTypes]]
            tuple of features, target and type mask for features
        """
        if isinstance(data, str):
            data = read_data(data)
            self._infer_feature_names(data)
        if isinstance(data, tuple):
            if self.features is not None or self.target is not None:
                print(
                    "When data is passed as tuple of (X, y) all columns are used regardless the values of `features` or `target` arguments."
                )
            if len(data) != 2:
                raise ValueError(
                    "When passing data as tuple, it should contain exactly 2 elements: `X` and `y`."
                )
            X, y = data
            if isinstance(X, pd.DataFrame):
                self.feature_names_to_save = list(X.columns)
            if len(y.shape) > 3 or len(X.shape) > 3:
                raise ValueError("Invalid data shape.")
            if len(y.shape) > 1 and y.shape[-1] != 1:
                raise ValueError("The target should contain only one column.")
            X, y = convert_to_np_obj(X), convert_to_np_obj(y)
        else:
            self._infer_feature_names(data)
            X, y = split_features(data, features=self.features, target=self.target)
            if self.features is None and isinstance(data, pd.DataFrame):
                self.feature_names_to_save = list(data.columns[:-1])
            elif self.features is not None:
                self.feature_names_to_save = self.features
        X, y = clean_data_split(X, y)
        self.dataset_size = X.shape
        mask: List[ColumnTypes]
        if training:
            mask = determine_column_types(X)
        else:
            mask = []
        if len(y.shape) == 2:
            y = y.ravel()
        return X, y, mask

    @property
    def default_pipeline(self) -> Type[Pipeline]:
        """
        Default pipeline class.
        """

        return SimpleTabularPipeline

    @property
    def default_pipeline_options(self) -> Dict:
        """
        Default options for pipeline.
        """
        options: Dict[str, Any] = {"mask": self._data[2]}
        return options

    def _cross_validate(self) -> None:
        cv = (
            self.eval_strategy
            if isinstance(self.eval_strategy, BaseCrossValidator)
            else None
        )
        scores = tab_cv_score(
            self._pipeline, self._data[0], self._data[1], self.task, cv=cv
        )
        scores["N_SAMPLES"] = self.dataset_size[0]
        self._stored_cv_score = scores

    def train(self, **kwargs: Any) -> TabularTaskManager:
        """
        Invokes the training procedure of an underlying pipeline.

        Returns
        -------
        TabularTaskManager
            `self`
        """
        print_("Beginning training")
        if self.eval_strategy is not None:
            eval_strategy = self.eval_strategy
            split_fn = None
            if eval_strategy == "auto":
                if self.dataset_size[0] < 2500:
                    eval_strategy = "cv"
                else:
                    eval_strategy = "holdout"

            if eval_strategy == "holdout":
                split_fn = lambda X, y: train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    stratify=y if self.task == "tabular_classification" else None,
                )
            elif callable(eval_strategy) and not isinstance(
                eval_strategy, BaseCrossValidator
            ):
                split_fn = eval_strategy

            if callable(split_fn):
                if self._eval_set is None:
                    X_train, X_eval, y_train, y_eval = split_fn(
                        self._data[0], self._data[1]
                    )
                    self._data = (X_train, y_train, self._data[2])
                    self._eval_set = (X_eval, y_eval)
                else:
                    print_("Evaluation set is already available.")
            elif eval_strategy == "cv" or isinstance(eval_strategy, BaseCrossValidator):
                print_("Starting cross validation")
                self._cross_validate()
                print_("Finished cross-validation")

        print_("Beginning the main training phase")
        self._pipeline.fit(self._data[0], self._data[1])
        print_("Finished training")
        return self

    def predict(self, data: Union[str, npt.NDArray, pd.DataFrame]) -> npt.NDArray:
        """
        Performs prediction on new data.

        Parameters
        ----------
        data : Union[str, npt.NDArray, pd.DataFrame]
             path to data file or pandas dataframe or numpy array

        Returns
        -------
        npt.NDArray
            predictions
        """
        if isinstance(data, str):
            data = read_data(data)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.object_)
        return self._pipeline.predict(data)

    def predict_stored_subset(self, subset: str = "train") -> npt.NDArray:
        """
        Makes a prediction on a stored subset (`train` or `eval`).

        Parameters
        ----------
        subset : str, optional
            subset to predict on (train or eval), by default 'train'

        Returns
        -------
        npt.NDArray
            predicted values
        """
        if subset == "train":
            return self.predict(self._data[0])
        elif subset == "eval":
            if self._eval_set is None:
                raise RuntimeError("validation set is not available")
            return self.predict(self._eval_set[0])
        else:
            raise ValueError("subset should be either `train` or `eval`")

    def performance_summary(
        self, test_data: Optional[Union[str, npt.NDArray, pd.DataFrame, Tuple]] = None
    ) -> dict:
        """
        Prints a performance summary of the model.
        The summary always includes metrics calculated for the train set.
        If the train/eval split was done during training, the summary includes metrics calculated on eval set.
        If test set is provided as an argument, the performance includes metrics calculated on test set.

        Parameters
        ----------
        test_data : Optional[Union[str, npt.NDArray, pd.DataFrame, Tuple]]
            data to be used as test set, by default None

        Returns
        -------
        dict
            metrics for each subset
        """
        metrics_ = {}
        report_fn = (
            print_classification_report
            if self.task == "tabular_classification"
            else print_regression_report
        )
        y_hat_train = self.predict_stored_subset("train")
        metrics_["train"] = report_fn(self._data[1], y_hat_train, silent=True)
        if self._eval_set is not None:
            y_hat_eval = self.predict_stored_subset("eval")
            metrics_["eval"] = report_fn(self._eval_set[1], y_hat_eval, silent=True)
        if self._stored_cv_score is not None:
            metrics_["eval_cv"] = self._stored_cv_score
        if test_data is not None:
            metrics_["test"] = self.evaluate(test_data, silent=True)
        df = pd.DataFrame.from_dict(metrics_, orient="index")
        print("\n", df, "\n")
        return metrics_

    def evaluate(
        self,
        test_data: Union[str, npt.NDArray, pd.DataFrame, Tuple],
        silent: bool = False,
    ) -> Dict:
        """
        Perfoms and prints the evaluation report on the given dataset.

        Parameters
        ----------
        test_data : Union[str, npt.NDArray, pd.DataFrame, Tuple]
            dataset to be used for evaluation
        silent: bool
            controls whether the metrics are printed on screen, by default False
        """
        print("The evaluation report will be provided here")
        X, y, _ = self._prepare_data(test_data, training=False)
        y_hat = self.predict(X)
        if self.task == "tabular_classification":
            return print_classification_report(y, y_hat, silent=silent)
        else:
            return print_regression_report(y, y_hat, silent=silent)
