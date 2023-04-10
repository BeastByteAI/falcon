import os
import sys
import warnings
from falcon.runtime import ONNXRuntime
from typing import List, Optional, Dict
from typing import List, Tuple, Optional
from numpy import typing as npt
import numpy as np
from typing import Any, Dict, Union


def run_model(model_path: str, X: npt.NDArray) -> Union[List[npt.NDArray], np.ndarray]:
    """
    Runs input data through the saved model.

    Parameters
    ----------
    model_path : str
        model path
    X : npt.NDArray
        model inputs

    Returns
    -------
    Union[List[npt.NDArray], np.ndarray]
        model predictions
    """
    if model_path.endswith("onnx"):
        return run_onnx(model_path, X, "final")
    else:
        raise ValueError("Invalid model path")


def run_onnx(
    model: Union[bytes, str], X: npt.NDArray, outputs: str = "final"
) -> List[npt.NDArray]:
    runtime = ONNXRuntime(model=model)
    return runtime.run(X, outputs=outputs)


def set_verbosity_level(level: int = 1) -> None:
    if level not in {0, 1}:
        level = 0
    os.environ["FALCON_VERBOSITY_LEVEL"] = str(level)


def print_(*args: Any) -> None:
    verbosity_level = os.getenv("FALCON_VERBOSITY_LEVEL", "1")
    if verbosity_level == "1":
        for a in args:
            print(a)


def disable_warnings() -> None:
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
    return None


def set_eval_strategy(
    eval_strategy: Any, manager_configuration_: Dict, test_data: Any = None
) -> None:
    if "eval_strategy" in manager_configuration_.keys():
        if not eval_strategy == "dynamic":
            manager_configuration_["eval_strategy"] = eval_strategy
    else:
        if eval_strategy == "dynamic":
            eval_strategy = "auto" if test_data is None else None
        manager_configuration_["eval_strategy"] = eval_strategy
