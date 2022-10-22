from __future__ import annotations
from falcon.abstract import TaskManager, Pipeline
from falcon.tabular.pipelines.simple_tabular_pipeline import SimpleTabularPipeline
from .reporting import print_classification_report, print_regression_report
from falcon.tabular.utils import *
from falcon import types as ft
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
        data: Union[str, npt.NDArray, pd.DataFrame],  # TODO add support for Tuple[X, y]
        pipeline: Optional[Type[Pipeline]] = None,
        pipeline_options: Optional[Dict] = None,
        extra_pipeline_options: Optional[Dict] = None,
        features: Optional[ft.ColumnsList] = None,
        target: Optional[Union[str, int]] = None,
        **options: Any,
    ) -> None:
        """

        Parameters
        ----------
        task : str
            `tabular_classification` or `tabular_regression`
        data : Union[str, npt.NDArray, pd.DataFrame]
            Path to data file or pandas dataframe or numpy array.
        pipeline: Optional[Type[Pipeline]] 
            class to be used as pipeline, by default None
            if None, SimpleTabularPipeline will be used
        pipeline_options : Optional[Dict], optional
            Arguments to be passed to the pipeline, by default None.
            These options will overwrite the ones from `default_pipeline_options` attribute.
        extra_pipeline_options : Optional[Dict], optional
            Arguments to be passed to the pipeline, by default None.
            These options will be passed in addition to the ones from `default_pipeline_options` attribute.
            This argument is ignored if `pipeline_options` is not None.
        features : Optional[ft.ColumnsList], optional
            Names or indices of columns to be used as features, by default None.
            If None, all columns except the last one will be used.
            If `target` argument is not None, features should be passed explicitly as well.
        target : Optional[Union[str, int]], optional
            Name or index of column to be used as target, by default None.
            If None, the last column will be used as target.
            If `features` argument is not None, target should be specified explicitly as well.
        """
        print_(f"\nInitializing a new TabularTaskManager for task `{task}`")

        super().__init__(
            task=task,
            data=data,
            pipeline=pipeline,
            pipeline_options=pipeline_options,
            extra_pipeline_options=extra_pipeline_options,
            features=features,
            target=target,
        )

        self._eval_set = None

    def _prepare_data(
        self, data: Union[str, npt.NDArray, pd.DataFrame], training: bool = True
    ) -> Tuple[npt.NDArray, npt.NDArray, List[bool]]:
        """
        Initial data preparation. 
            1) Optional: read data from the specified location
            2) Split into features and targets. By default it is assumed that the last column is the target.
            3) Clean data
            4) Determine numerical and categorical features (create categorical mask).

        Parameters
        ----------
        data : Union[str, npt.NDArray, pd.DataFrame]
            Path to data file or pandas dataframe or numpy array.

        Returns
        -------
        Tuple[npt.NDArray, npt.NDArray, List[bool]]
            Tuple of features, target and categorical mask for features.
        """
        if isinstance(data, str):
            data = read_data(data)
        X, y = split_features(data, features=self.features, target=self.target)
        X, y = clean_data_split(X, y)
        mask: List[bool]
        if training:
            mask = get_cat_mask(X)
        else: 
            mask = []
        if len(y.shape) == 2:
            y = y.ravel()
        return X, y, mask

    @property
    def default_pipeline(self) -> Type[Pipeline]:
        """
        default pipeline class
        """

        return SimpleTabularPipeline

    @property
    def default_pipeline_options(self) -> Dict:
        """
        default options for pipeline
        """
        options = {"mask": self._data[2]}
        return options
    
    def _pre_eval_pipeline(self) -> float: 
        print_("Pre-evaluation of model performance")
        # TODO make `no_print` context manager
        old_verbosity_level = int(os.getenv("FALCON_VERBOSITYLEVEL", "1"))
        set_verbosity_level(0)
        if (
            self._data[0].shape[0] * self._data[0].shape[1] < 50000
            or self._data[0].shape[0] < 500
        ):
            score = tab_cv_score(
                self._pipeline, self._data[0], self._data[1], self.task
            )
            avg_score = float(np.mean(score))
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self._data[0], self._data[1], test_size=0.25
            )
            copied_pipeline = deepcopy(self._pipeline)
            copied_pipeline.fit(X_train, y_train)
            pred = copied_pipeline.predict(X_test)
            avg_score = calculate_model_score(y_test, pred, self.task)
        set_verbosity_level(old_verbosity_level)
        return avg_score


    def train(self, make_eval_subset: bool = True, pre_eval: bool = False, **kwargs: Any) -> TabularTaskManager:
        """
        Invokes the training procedure of an underlying pipeline. Print an expected model performance if available.

        Parameters
        ----------
        pre_eval : bool
            If True, first estimate model perfromance via 10 folds CV for small datasets or 25% test split for large datasets, by default False.
            Setting pre_eval = True is not reccomended as it pre-evaluates the pipeline as a whole which has lots of random elements therefore the results might be non reproducable.
        make_eval_subset: bool
            Controls whether a dedicated eval set should be allocated for performance report, by default True. 
            If True, overwrites the value of `pre_eval` to False.
        Returns
        -------
        TabularTaskManager
            self
        """
        print_("Beginning training")
        if make_eval_subset:
            if self._eval_set is None:
                X_train, X_eval, y_train, y_eval = train_test_split(
                    self._data[0], self._data[1], test_size=0.25
                )
                self._data = (X_train, y_train, self._data[2])
                self._eval_set = (X_eval, y_eval)
            else: 
                print('Evaluation set is already available.')
        elif pre_eval:
            print("Setting `pre_eval = True` is not reccomended as it pre evaluates the whole pipeline, can be lenghty and innacurate")
            preeval_score = self._pre_eval_pipeline()
            print(f"Pre evaluation score: {preeval_score}")
        print_("Beginning the main training phase")
        self._pipeline.fit(self._data[0], self._data[1])
        print_("Finished training")
        return self

    def predict(self, data: Union[str, npt.NDArray, pd.DataFrame]) -> npt.NDArray:
        """
        Perform prediction on new data.

        Parameters
        ----------
        data : Union[str, npt.NDArray, pd.DataFrame]
             Path to data file or pandas dataframe or numpy array.

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

    def predict_stored_subset(self, subset = 'train') -> npt.NDArray:
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
        if subset == 'train': 
            return self.predict(self._data[0])
        elif subset == 'eval':
            if self._eval_set is None: 
                raise RuntimeError('validation set is not available') 
            return self.predict(self._eval_set[0])
        else:
            raise ValueError('subset should be either `train` or `eval`')

    def performance_summary(self, test_data: Optional[Union[str, npt.NDArray, pd.DataFrame]]) -> dict:
        """
        Prints a performance summary of the model. 
        The summary always includes metrics calculated for the train set. 
        If the train/eval split was done during training, the summary includes metrics calculated on eval set. 
        If test set is provided as an argument, the performance includes metrics calculated on test set.

        Parameters
        ----------
        test_data : Optional[Union[str, npt.NDArray, pd.DataFrame]]
            data to be used as test set, by default None.

        Returns
        -------
        dict
            metrics for each subset
        """
        metrics_ = {}
        report_fn = print_classification_report if self.task == 'tabular_classification' else print_regression_report
        y_hat_train = self.predict_stored_subset('train')
        y_hat_eval = self.predict_stored_subset('eval')
        metrics_['train'] = report_fn(self._data[1], y_hat_train, silent = True)
        if self._eval_set is not None: 
            metrics_['eval'] = report_fn(self._eval_set[1], y_hat_eval, silent = True)
        if test_data is not None: 
            metrics_['test'] = self.evaluate(test_data, silent = True)
        df = pd.DataFrame.from_dict(metrics_, orient = 'index')
        print("\n", df, "\n")
        return metrics_

    def evaluate(self, test_data: Union[str, npt.NDArray, pd.DataFrame], silent = False) -> None:
        """
        Perfoms and prints the evaluation report on the given dataset.

        Parameters
        ----------
        test_data : Union[str, npt.NDArray, pd.DataFrame]
            Dataset to be used for evaluation.
        silent: bool
            Controls whether the metrics are printed on screen, by default False.
        """
        print("The evaluation report will be provided here")
        X, y, _ = self._prepare_data(test_data, training = False)
        y_hat = self.predict(X)
        if self.task == "tabular_classification":
            return print_classification_report(y, y_hat, silent = silent)
        else:
            return print_regression_report(y, y_hat, silent = silent)
