from copy import deepcopy
from typing import Dict
from falcon.tabular.configurations import (
    TABULAR_CLASSIFICATION_CONFIGURATIONS,
    TABULAR_REGRESSION_CONFIGURATIONS,
)


def get_task_configuration(task: str, configuration_name: str) -> Dict:
    if task == "tabular_classification":
        configs = TABULAR_CLASSIFICATION_CONFIGURATIONS
    elif task == "tabular_regression":
        configs = TABULAR_REGRESSION_CONFIGURATIONS
    else:
        raise ValueError("Unknown task")

    if configuration_name not in configs.keys():
        raise ValueError(f"Task {task} has no config with name `{configuration_name}`")
    
    return deepcopy(configs[configuration_name])