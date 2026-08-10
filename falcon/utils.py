import logging
from typing import Any

logger = logging.getLogger("falcon")
if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
    logger.addHandler(logging.NullHandler())

_VERBOSITY_LOG_LEVELS: dict[int, int] = {
    0: logging.WARNING,
    1: logging.INFO,
}


def set_verbosity_level(level: int = 1) -> None:
    logger.setLevel(_VERBOSITY_LOG_LEVELS.get(level, logging.WARNING))


def set_eval_strategy(
    eval_strategy: Any,
    manager_configuration_: dict[str, Any],
    test_data: Any = None,
) -> None:
    if "eval_strategy" in manager_configuration_:
        if eval_strategy != "dynamic":
            manager_configuration_["eval_strategy"] = eval_strategy
    else:
        if eval_strategy == "dynamic":
            eval_strategy = "auto" if test_data is None else None
        manager_configuration_["eval_strategy"] = eval_strategy
