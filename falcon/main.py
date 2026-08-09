from __future__ import annotations

import datetime
from typing import Any, Literal

from falcon.config import EvalStrategy, RunConfig
from falcon.predictor import UNSPECIFIED, Predictor, _Unspecified
from falcon.tabular.ingestion import TabularData
from falcon.tabular.splitting import GroupBy
from falcon.utils import logger

_REMOVED_KWARGS = {
    "manager_configuration": (
        "`manager_configuration` was removed in 0.9 — use `preset` or "
        "`config=RunConfig(...)`"
    ),
    "pipeline": "`pipeline` was removed in 0.9 — use `preset` or `RunConfig`",
    "pipeline_options": (
        "`pipeline_options` was removed in 0.9 — use `preset` or `RunConfig`"
    ),
    "extra_pipeline_options": (
        "`extra_pipeline_options` was removed in 0.9 — use `preset` or `RunConfig`"
    ),
}


def _reject_unknown_kwargs(kwargs: dict[str, Any]) -> None:
    if not kwargs:
        return
    name = next(iter(kwargs))
    if name in _REMOVED_KWARGS:
        raise TypeError(_REMOVED_KWARGS[name])
    raise TypeError(f"AutoML() got an unexpected keyword argument `{name}`")


def AutoML(
    task: str,
    train_data: TabularData,
    test_data: TabularData | None = None,
    features: list[str] | list[int] | None = None,
    target: str | int | None = None,
    save_model: bool = True,
    eval_strategy: EvalStrategy | Literal["dynamic"] = "dynamic",
    config: RunConfig | str | None = None,
    preset: str = "balanced",
    time_limit: float | None | _Unspecified = UNSPECIFIED,
    random_state: int | _Unspecified = UNSPECIFIED,
    group_by: GroupBy | None = None,
    **kwargs: Any,
) -> Predictor:
    _reject_unknown_kwargs(kwargs)
    if not isinstance(save_model, bool):
        raise TypeError("save_model must be a boolean")

    run_config: RunConfig | None
    resolved_preset = preset
    if isinstance(config, str):
        if preset != "balanced" and preset != config:
            raise ValueError(
                "Pass a preset name through either `preset` or `config`, not both"
            )
        resolved_preset = config
        run_config = None
    elif config is None or isinstance(config, RunConfig):
        run_config = config
    else:
        if isinstance(config, dict):
            raise TypeError(
                "Dictionary configs were removed in 0.9 — use `config=RunConfig(...)`"
            )
        raise TypeError("config must be a RunConfig or preset name")

    resolved_eval_strategy: EvalStrategy
    if eval_strategy == "dynamic":
        resolved_eval_strategy = None if test_data is not None else "auto"
    else:
        resolved_eval_strategy = eval_strategy
    predictor_options: dict[str, Any] = {
        "task": task,
        "preset": resolved_preset,
        "config": run_config,
        "eval_strategy": resolved_eval_strategy,
    }
    if not isinstance(time_limit, _Unspecified):
        predictor_options["time_limit"] = time_limit
    if not isinstance(random_state, _Unspecified):
        predictor_options["random_state"] = random_state

    predictor = Predictor(**predictor_options)
    predictor.fit(
        train_data,
        features=features,
        target=target,
        group_by=group_by,
    )
    predictor._performance_summary(test_data)
    if save_model:
        timestamp = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
        filename = f"falcon_{predictor.task}_{timestamp}.fnnx"
        logger.info("Saving the model ...")
        predictor.save(filename)
        logger.info("The model was saved as `%s`", filename)
    return predictor


__all__ = ["AutoML"]
