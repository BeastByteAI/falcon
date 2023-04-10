import pandas as pd
from typing import Union, List, Dict, Tuple, Optional
import numpy as np
from falcon.task_configurations import get_task_configuration
from falcon.tabular.adapters.ts.auxiliary import _create_window, _split_fn
from falcon.tabular.adapters.ts.pipeline import TSAdapterPipeline
from falcon.tabular.adapters.ts.plot_errors import _plot_errors
from falcon.abstract import TaskManager
from sklearn.metrics import mean_absolute_error, r2_score


class TSAdapter:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        window_size: int = 8,
        adapt_for: str = "tabular_regression",
        config: Union[str, Dict] = "PlainLearner",
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
        self._manager: Optional[TaskManager] = None

    def adapt(self, target_function: str = "AutoML") -> Dict:
        if target_function not in ("AutoML", "initialize"):
            raise ValueError("target_function should be one of (AutoML, initialize)")
        data = self.dataframe[self.target].astype(np.float32)
        df = pd.DataFrame({"y": data})
        df = _create_window(df, self.window_size)
        if isinstance(self.config, str):
            config = get_task_configuration(self._adapt_for, self.config)
        else:
            config = self.config
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

    def evaluate(
        self, forecast_period: int = 1, visualize: bool = False
    ) -> Optional[pd.DataFrame]:
        if self._manager is None:
            raise ValueError("Manager is not bound. Please call .bind() method first")
        if not hasattr(self._manager, "_eval_set"):
            print("Cannot evaluate. No evaluation set was provided")
            return None
        _train = self._manager._data
        _eval = self._manager._eval_set  # type: ignore
        _train = _train[1].squeeze()

        forecast_period_m1 = forecast_period - 1
        predictions = np.zeros(shape=(len(_eval[1]) - forecast_period_m1,))

        trunc = forecast_period_m1 if forecast_period_m1 > 0 else -len(_eval[1])
        for i, datapoint in enumerate(_eval[0][:-trunc]):
            datapoint = datapoint.reshape(1, -1)
            predictions[i] = self.predict(datapoint, forecast_period=forecast_period)[
                -1
            ]

        metrics = {
            "FORECAST_WINDOW": [forecast_period],
            "N_FORECASTS": [len(predictions)],
            "MAE": [mean_absolute_error(_eval[1][forecast_period_m1:], predictions)],
            "R2": [r2_score(_eval[1][forecast_period_m1:], predictions)],
        }
        df_metrics = pd.DataFrame(metrics)
        print(df_metrics)
        if visualize:
            _plot_errors(_train, _eval[1], predictions)
        return df_metrics

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray], forecast_period: int = 3
    ) -> np.ndarray:
        if self._manager is None:
            raise ValueError("Manager is not bound. Please call .bind() method first")
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = X.copy()
        predictions = np.zeros(shape=(forecast_period,))
        for i in range(forecast_period):
            # print(X, X.shape)
            pred = self._manager.predict(X)[0]
            predictions[i] = pred
            X = np.roll(X, -1)
            X[0, -1] = pred
        return predictions
