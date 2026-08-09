from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from types import ModuleType

import pytest

import falcon
from falcon.config import (
    DATASET_AWARE_ORDERING_DEFAULT,
    PortfolioSource,
    RunConfig,
)
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.presets import PresetRegistry, resolve_run_config
from falcon.tabular.candidates import EstimatorSpec


def _spec(name: str) -> EstimatorSpec:
    return EstimatorSpec(name=name, family="hist_gradient_boosting")


def test_run_config_is_public_package_api() -> None:
    assert falcon.RunConfig is RunConfig
    assert RunConfig().dataset_aware_ordering is DATASET_AWARE_ORDERING_DEFAULT


def test_run_config_is_frozen_and_normalizes_sequences() -> None:
    source = PortfolioSource(specs=[_spec("first"), _spec("second")])  # type: ignore[arg-type]
    config = RunConfig(candidate_sources=[source])

    assert isinstance(source.specs, tuple)
    assert isinstance(config.candidate_sources, tuple)
    with pytest.raises(FrozenInstanceError):
        config.random_state = 7  # type: ignore[misc]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"candidate_sources": []}, "candidate_sources"),
        ({"ensemble_max_iterations": 0}, "ensemble_max_iterations"),
        ({"plateau_patience": 0}, "plateau_patience"),
        ({"plateau_tolerance": -0.1}, "plateau_tolerance"),
        ({"plateau_tolerance": False}, "plateau_tolerance"),
        ({"oof_folds": 1}, "oof_folds"),
        ({"eval_strategy": "invalid"}, "eval_strategy"),
        ({"time_limit": 0.0}, "time_limit"),
        ({"time_limit": float("inf")}, "time_limit"),
        ({"random_state": -1}, "random_state"),
        ({"dataset_aware_ordering": 1}, "dataset_aware_ordering"),
        ({"calibrate": 1}, "calibrate"),
        ({"impute_missing": 1}, "impute_missing"),
        ({"conformal_alpha": 0.0}, "conformal_alpha"),
        ({"conformal_alpha": 1.0}, "conformal_alpha"),
        ({"class_weight": "auto"}, "class_weight"),
        ({"decision_metric": "accuracy"}, "decision_metric"),
    ],
)
def test_run_config_rejects_invalid_settings(
    options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RunConfig(**options)  # type: ignore[arg-type]


def test_imbalance_settings_default_to_opt_in_weighting_and_a_tuned_rule() -> None:
    config = RunConfig()

    assert config.class_weight == "none"
    assert config.decision_metric == "balanced_accuracy"
    assert config.prior_correct


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, True),
        ({"class_weight": "balanced"}, False),
        ({"decision_metric": None}, False),
        ({"class_weight": "balanced", "decision_metric": None}, False),
    ],
)
def test_prior_correction_needs_unweighted_training_and_a_tuned_rule(
    options: dict[str, object],
    expected: bool,
) -> None:
    assert RunConfig(**options).prior_correct is expected  # type: ignore[arg-type]


def test_replaced_keeps_untouched_defaults_out_of_the_provided_fields() -> None:
    config = RunConfig(oof_folds=3).replaced(random_state=7)

    assert config._provided_fields == frozenset({"oof_folds", "random_state"})
    assert config.oof_folds == 3
    assert config.random_state == 7


def test_portfolio_source_resolves_for_the_requested_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int | None]] = []
    specs = (_spec("first"), _spec("second"), _spec("third"))

    def portfolio(
        task: str, *, n_classes: int | None = None
    ) -> tuple[EstimatorSpec, ...]:
        calls.append((task, n_classes))
        return specs

    monkeypatch.setattr("falcon.tabular.candidates.default_portfolio", portfolio)

    resolved = PortfolioSource(max_candidates=2).get_candidates(
        TABULAR_CLASSIFICATION_TASK,
        n_classes=3,
    )

    assert resolved == specs[:2]
    assert calls == [(TABULAR_CLASSIFICATION_TASK, 3)]


@pytest.mark.parametrize("task", [TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK])
def test_builtin_presets_have_the_expected_training_profiles(task: str) -> None:
    fast = PresetRegistry.get_preset(task, "fast")
    balanced = PresetRegistry.get_preset(task, "balanced")
    best = PresetRegistry.get_preset(task, "best")

    def portfolio(config: RunConfig) -> PortfolioSource:
        source = config.candidate_sources[0]
        assert isinstance(source, PortfolioSource)
        return source

    assert portfolio(fast).max_candidates == 1
    assert not fast.ensemble_enabled
    assert not fast.plateau_enabled

    assert balanced.ensemble_enabled
    assert balanced.plateau_enabled
    assert portfolio(balanced).max_candidates is not None

    assert best.ensemble_enabled
    assert not best.plateau_enabled
    assert portfolio(best).max_candidates is None
    assert best.oof_folds > balanced.oof_folds
    assert best.ensemble_max_iterations > balanced.ensemble_max_iterations


def test_run_config_resolution_applies_documented_precedence() -> None:
    custom = RunConfig(
        eval_strategy=None,
        random_state=11,
        time_limit=30.0,
    )

    resolved = resolve_run_config(
        TABULAR_REGRESSION_TASK,
        "fast",
        config=custom,
        eval_strategy="holdout",
        random_state=19,
        time_limit=5.0,
    )

    assert not resolved.ensemble_enabled
    assert resolved.ensemble_max_iterations == 1
    assert resolved.eval_strategy == "holdout"
    assert resolved.random_state == 19
    assert resolved.time_limit == 5.0

    enabled = resolve_run_config(
        TABULAR_REGRESSION_TASK,
        "fast",
        config=RunConfig(ensemble_enabled=True),
    )
    assert enabled.ensemble_enabled


def test_explicit_none_overrides_config_values() -> None:
    resolved = resolve_run_config(
        TABULAR_REGRESSION_TASK,
        config=RunConfig(eval_strategy="cv", time_limit=30.0),
        eval_strategy=None,
        time_limit=None,
    )

    assert resolved.eval_strategy is None
    assert resolved.time_limit is None


@pytest.mark.parametrize(
    "legacy_name",
    ["SuperLearner", "SuperLearner.mini", "OptunaLearner.hgbt", "PlainLearner"],
)
def test_new_registry_rejects_legacy_presets_with_migration_guidance(
    legacy_name: str,
) -> None:
    with pytest.raises(ValueError) as error:
        PresetRegistry.get_preset(TABULAR_CLASSIFICATION_TASK, legacy_name)

    message = str(error.value)
    assert "removed in 0.9" in message
    assert all(name in message for name in ("fast", "balanced", "best"))
    assert "RunConfig" in message


def test_extension_preset_is_discovered_and_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "falcon_ml_example"
    preset_name = "EXAMPLE::tiny"
    module = ModuleType(module_name)
    extension_config = RunConfig(ensemble_enabled=False)

    def self_register() -> None:
        PresetRegistry.register_presets(
            TABULAR_REGRESSION_TASK,
            {preset_name: extension_config},
            silent=True,
        )

    module.self_register = self_register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.delenv("FALCON_PREVENT_EXTENSION_AUTO_LOAD", raising=False)

    try:
        assert (
            PresetRegistry.get_preset(TABULAR_REGRESSION_TASK, preset_name)
            == extension_config
        )
    finally:
        PresetRegistry._PRESETS[TABULAR_REGRESSION_TASK].pop(preset_name, None)


def test_unknown_preset_lists_available_names() -> None:
    with pytest.raises(ValueError) as error:
        PresetRegistry.get_preset(TABULAR_REGRESSION_TASK, "unknown")

    message = str(error.value)
    assert all(name in message for name in ("fast", "balanced", "best"))
