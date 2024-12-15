from typing import Any, Union, Dict, List, Type, Optional
import pandas as pd

try:
    from fnnx.runtime import Runtime as _Runtime
    from fnnx.handlers.local import LocalHandler as _LocalHandler
except ImportError:
    _Runtime = None
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK

import numpy as np


class Runtime:
    """
    Runtime for ONNX models based on PHONNX.
    """

    def __init__(self, model_path: str):
        if _Runtime is None:
            raise ImportError("FNNX is not installed.")
        self.runtime = _Runtime(model_path)
        handler: _LocalHandler = self.runtime.handler

        self._input_names = list(handler.input_specs.keys())

        producer_tags = handler.manifest.get("producer_tags", [])

        self.task = self._detect_task(producer_tags)

        if self.task is None:
            raise RuntimeError("Could not detect task from model tags.")

    def _detect_task(self, producer_tags: List[str]) -> Optional[str]:
        for tag in producer_tags:
            if tag.startswith(f"falcon.beastbyte.ai::{TABULAR_CLASSIFICATION_TASK}"):
                return TABULAR_CLASSIFICATION_TASK
            elif tag.startswith(f"falcon.beastbyte.ai::{TABULAR_REGRESSION_TASK}"):
                return TABULAR_REGRESSION_TASK

    def _predict(
        self,
        X: Union[np.ndarray | pd.DataFrame | Dict[str, np.ndarray]],
    ) -> List[np.ndarray]:
        """
        Runs the model.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame | Dict[str, np.ndarray]
            model

        Returns
        -------
        List[np.ndarray]
            model predictions
        """

        if isinstance(X, pd.DataFrame):
            X = X.to_dict(orient="list")
        elif isinstance(X, np.ndarray):
            X = {name: X[:, i] for i, name in enumerate(self._input_names)}

        for k in X.keys():
            X[k] = np.asarray(X[k]).reshape(-1, 1)

        return self.runtime.compute(X, {})

    def predict(
        self, X: Union[np.ndarray | pd.DataFrame | Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Runs the model.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame | Dict[str, np.ndarray]
            model

        Returns
        -------
        np.ndarray
            model predictions
        """
        return self._predict(X)["y_pred"]

    def predict_proba(
        self, X: Union[np.ndarray | pd.DataFrame | Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Predicts probabilities of classes.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame | Dict[str, np.ndarray]
            model

        Returns
        -------
        np.ndarray
            model predicted probabilities
        """
        if self.task != TABULAR_CLASSIFICATION_TASK:
            raise RuntimeError(f"Called predict_proba on a model for {self.task} task.")
        return self._predict(X)["probabilities"]
