from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from falcon.config import EvalStrategy, PortfolioSource, RunConfig
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.utils import logger

_EXTENSION_PREFIX = "falcon_ml_"
_LEGACY_PRESET_PREFIXES = ("SuperLearner", "OptunaLearner", "PlainLearner")


def _prevent_extension_load() -> bool:
    return bool(os.getenv("FALCON_PREVENT_EXTENSION_AUTO_LOAD", False))


class PresetRegistry:
    _PRESETS: dict[str, dict[str, RunConfig]] = {
        TABULAR_CLASSIFICATION_TASK: {},
        TABULAR_REGRESSION_TASK: {},
    }

    @classmethod
    def register_presets(
        cls,
        task: str,
        presets: Mapping[str, RunConfig],
        silent: bool = False,
    ) -> None:
        if task not in cls._PRESETS:
            raise ValueError(f"Unknown task `{task}`")
        if not presets:
            raise ValueError("At least one preset must be provided")
        if any(not name for name in presets):
            raise ValueError("Preset names must not be empty")
        invalid_configs = [
            name
            for name, config in presets.items()
            if not isinstance(config, RunConfig)
        ]
        if invalid_configs:
            raise TypeError(
                f"Preset `{invalid_configs[0]}` must be registered with a RunConfig"
            )
        cls._PRESETS[task].update(
            {name: config.replaced() for name, config in presets.items()}
        )
        if not silent:
            logger.info("Registered presets %s for task %s", list(presets), task)

    @classmethod
    def get_preset(
        cls,
        task: str,
        preset_name: str,
        allow_extensions_discovery: bool = True,
    ) -> RunConfig:
        if task not in cls._PRESETS:
            raise ValueError(f"Unknown task `{task}`")
        if preset_name in cls._PRESETS[task]:
            return cls._PRESETS[task][preset_name]
        should_load = (
            allow_extensions_discovery
            and not _prevent_extension_load()
            and "::" in preset_name
        )
        if should_load:
            extension_name = preset_name.split("::", maxsplit=1)[0]
            logger.info(
                "Extension `%s` does not seem to be loaded. Will try to load "
                "automatically.",
                _EXTENSION_PREFIX + extension_name.lower(),
            )
            cls.load_extension(extension_name)
            return cls.get_preset(task, preset_name, False)

        available = ", ".join(cls.get_registered_preset_names(task))
        if preset_name.startswith(_LEGACY_PRESET_PREFIXES):
            raise ValueError(
                f"Preset `{preset_name}` was removed in 0.9. Available presets: "
                f"{available}. Use `preset` or `config=RunConfig(...)`."
            )
        raise ValueError(
            f"Preset `{preset_name}` does not exist. Available presets: {available}."
        )

    @classmethod
    def get_registered_preset_names(cls, task: str) -> tuple[str, ...]:
        if task not in cls._PRESETS:
            raise ValueError(f"Unknown task `{task}`")
        return tuple(cls._PRESETS[task])

    @classmethod
    def load_extension(cls, extension_name: str) -> None:
        normalized_name = extension_name.lower()
        module_name = _EXTENSION_PREFIX + normalized_name
        logger.info("Attempting to load %s...", module_name)
        try:
            __import__(module_name).self_register()
        except ModuleNotFoundError:
            logger.warning(
                "Seems like the extension `%s` is not installed. Try installing it "
                "first using `pip install %s`.",
                normalized_name,
                module_name,
            )


class _Unset:
    pass


_UNSET = _Unset()


def resolve_run_config(
    task: str,
    preset: str = "balanced",
    *,
    config: RunConfig | None = None,
    time_limit: float | None | _Unset = _UNSET,
    random_state: int | _Unset = _UNSET,
    eval_strategy: EvalStrategy | _Unset = _UNSET,
) -> RunConfig:
    preset_config = PresetRegistry.get_preset(task, preset)
    if config is not None and not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    resolved = preset_config if config is None else preset_config.merged_with(config)
    overrides: dict[str, Any] = {}
    if not isinstance(time_limit, _Unset):
        overrides["time_limit"] = time_limit
    if not isinstance(random_state, _Unset):
        overrides["random_state"] = random_state
    if not isinstance(eval_strategy, _Unset):
        overrides["eval_strategy"] = eval_strategy
    return resolved.replaced(**overrides)


def _builtin_presets() -> dict[str, RunConfig]:
    return {
        "fast": RunConfig(
            candidate_sources=(PortfolioSource(max_candidates=1),),
            ensemble_enabled=False,
            ensemble_max_iterations=1,
            plateau_enabled=False,
            oof_folds=2,
        ),
        "balanced": RunConfig(
            candidate_sources=(PortfolioSource(max_candidates=4),),
            ensemble_max_iterations=50,
            plateau_enabled=True,
            plateau_patience=2,
            oof_folds=5,
        ),
        "best": RunConfig(
            candidate_sources=(PortfolioSource(),),
            ensemble_max_iterations=100,
            plateau_enabled=False,
            oof_folds=10,
        ),
    }


PresetRegistry.register_presets(
    TABULAR_CLASSIFICATION_TASK,
    _builtin_presets(),
    silent=True,
)
PresetRegistry.register_presets(
    TABULAR_REGRESSION_TASK,
    _builtin_presets(),
    silent=True,
)

get_run_config = PresetRegistry.get_preset

__all__ = ["PresetRegistry", "get_run_config", "resolve_run_config"]
