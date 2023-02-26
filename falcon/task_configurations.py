from copy import deepcopy
from typing import Dict, List, Type
import os
from falcon.constants import (
    TABULAR_CLASSIFICATION_TASK as _TAB_CLF_TASK,
    TABULAR_REGRESSION_TASK as _TAB_REGR_TASK,
)
from falcon.tabular.configurations import (
    TABULAR_CLASSIFICATION_CONFIGURATIONS as _TAB_CLF_CONF,
    TABULAR_REGRESSION_CONFIGURATIONS as _TAB_REGR_CONF,
)
from falcon.abstract.task_manager import TaskManager as _TaskManager
from falcon.tabular.tabular_manager import TabularTaskManager as _TabularTaskManager

_PREFIX = "falcon_ml_"


def _prevent_load() -> bool:
    return bool(os.getenv("FALCON_PREVENT_EXTENSION_AUTO_LOAD", False))


class TaskConfigurationRegistry:

    _CONFIGURATIONS = {}

    @classmethod
    def register_task(cls, task: str, task_manager: Type[_TaskManager]) -> None:
        if task not in cls._CONFIGURATIONS.keys():
            if not issubclass(task_manager, _TaskManager):
                raise ValueError(
                    "Invalid task manager. Task manager should be a subclass of `falcon.base.manager.TaskManager`"
                )
            cls._CONFIGURATIONS[task] = {"manager": task_manager, "configs": {}}
        else:
            print(f"Task {task} already exists and will not be registered again.")

    @classmethod
    def get_registered_tasks(cls) -> List[str]:
        return cls._CONFIGURATIONS.keys()

    @classmethod
    def is_known_task(cls, task: str) -> bool:
        return task in cls._CONFIGURATIONS.keys()

    @classmethod
    def register_configurations(
        cls, task: str, config: Dict, silent: bool = False
    ) -> None:
        if not cls.is_known_task(task):
            raise ValueError(
                f"The task {task} does not exist. Please register it first using TaskConfigurationRegistry.register_task method."
            )
        if not silent:
            print(f"Registered {list(config.keys())} for task {task}")
        cls._CONFIGURATIONS[task]["configs"].update(deepcopy(config))

    @classmethod
    def get_configuration(
        cls, task: str, configuration_name: str, allow_extensions_discovery: bool = True
    ) -> Dict:
        if not cls.is_known_task(task):
            raise ValueError(f"Unknown task `{task}`")
        elif configuration_name not in cls._CONFIGURATIONS[task]["configs"].keys():
            should_load = (
                allow_extensions_discovery
                and not _prevent_load()
                and "::" in configuration_name
            )
            if should_load:

                extension_name = configuration_name.split("::")[0]
                print(
                    f"Extension `{_PREFIX + extension_name.lower()}` does not seem to be loaded. Will try to load automatically."
                )
                cls.load_extension(extension_name=extension_name)
                return cls.get_configuration(task, configuration_name, False)
            raise ValueError(f"Configuration `{configuration_name}` does not exist")
        return deepcopy(cls._CONFIGURATIONS[task]["configs"][configuration_name])

    @classmethod
    def get_registered_config_names(cls, task: str) -> List[str]:
        if not cls.is_known_task(task):
            raise ValueError(f"Unknown task `{task}`")
        return cls._CONFIGURATIONS[task]["configs"].keys()

    @classmethod
    def get_task_manager(cls, task: str) -> Type[_TaskManager]:
        if not cls.is_known_task(task):
            raise ValueError(f"Unknown task `{task}`")
        return cls._CONFIGURATIONS[task]["manager"]

    @classmethod
    def load_extension(cls, extension_name: str):
        extension_name = extension_name.lower()
        print(f"Attempting to load {_PREFIX + extension_name}...")
        try:
            __import__(_PREFIX + extension_name).self_register()
        except ModuleNotFoundError:
            print(
                f"Seems like the extension `{extension_name}` is not installed. Try installing it first using `pip install {_PREFIX+extension_name}`."
            )


TaskConfigurationRegistry.register_task(_TAB_CLF_TASK, _TabularTaskManager)
TaskConfigurationRegistry.register_task(_TAB_REGR_TASK, _TabularTaskManager)

TaskConfigurationRegistry.register_configurations(_TAB_CLF_TASK, _TAB_CLF_CONF, silent = True)
TaskConfigurationRegistry.register_configurations(_TAB_REGR_TASK, _TAB_REGR_CONF, silent = True)

# for backward compatibility
get_task_configuration = TaskConfigurationRegistry.get_configuration
