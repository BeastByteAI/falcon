import pandas as pd
from typing import Union, List, Dict, Tuple
import numpy as np
from falcon.task_configurations import get_task_configuration
from falcon.tabular.adapters.ts.auxiliary import _create_window, _split_fn
from falcon.tabular.adapters.ts.pipeline import TSAdapterPipeline
from falcon.abstract import TaskManager


class TSAdapter:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        window_size: int = 8,
        adapt_for: str = "tabular_regression",
        config: str = "PlainLearner",
        eval_size: float = 0.2,
    ):
        self.dataframe = dataframe
        self.window_size = window_size
        if adapt_for not in ("tabular_classification", "tabular_regression"):
            raise ValueError(
                "adapt_for should be one of (tabular_classification, tabular_regression"
            )
        if adapt_for == "tabular_classification":
            raise NotImplementedError("tabular_classification is not yet implemented")
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("invalid dataframe type, only pd.DataFrame is supported")
        if target not in dataframe.columns:
            raise ValueError(f"Provided target {target} was not found")
        if window_size <= 1 or window_size >= dataframe.shape[0] - 1:
            raise ValueError(
                "Invalid window size. Minumum value is 2, maximum value is (the number of rows in the dataframe - 1)"
            )
        self.target = target
        self._adapt_for = adapt_for
        self.config = config
        if eval_size <= 0.0 or eval_size >= 1.0:
            raise ValueError("eval_size should be in the range (O., 1.)")
        self.eval_size = eval_size
        self._manager = None

    def adapt(self, target_function: str = "AutoML") -> Dict:
        if target_function not in ("AutoML", "initialize"):
            raise ValueError("target_function should be one of (AutoML, initialize)")
        data = self.dataframe[self.target].astype(np.float32)
        df = pd.DataFrame({"y": data})
        df = _create_window(df, self.window_size)
        config = get_task_configuration(self._adapt_for, self.config)
        eval_strategy_fn = lambda X, y: _split_fn(X, y, self.eval_size)
        wrapped_pipeline = config["pipeline"]
        wrapped_pipeline_options = config["extra_pipeline_options"]
        config["pipeline"] = TSAdapterPipeline
        config["extra_pipeline_options"] = {
            "wrapped_pipeline": wrapped_pipeline,
            "wrapped_pipeline_options": wrapped_pipeline_options,
        }
        config["eval_strategy"] = eval_strategy_fn
        if target_function == "AutoML":
            config = {"config": config}
            config["train_data"] = df
        else:
            config["data"] = df
        config["task"] = self._adapt_for
        config["features"] = list(df.columns[:-1])
        config["target"] = "y"
        return config

    def bind(self, manager: TaskManager) -> None:
        self._manager = manager

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray], forecast_period=3
    ) -> np.ndarray:
        if self._manager is None:
            raise ValueError("Manager is not bound. Please call .bind() method first")
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        predictions = np.zeros(shape=(forecast_period,))
        for i in range(forecast_period):
            print(X, X.shape)
            pred = self._manager.predict(X)[0]
            predictions[i] = pred
            X = np.roll(X, -1)
            X[0, -1] = pred
        return predictions
