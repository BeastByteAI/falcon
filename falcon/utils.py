import os
import sys
import warnings
from typing import List, Optional, Dict
from typing import List, Tuple, Optional
from numpy import typing as npt
import numpy as np
from typing import Any, Dict, Union



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
