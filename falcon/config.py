from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from numpy import typing as npt
from sklearn.model_selection import BaseCrossValidator

from falcon.types import DatasetSchema

if TYPE_CHECKING:
    from falcon.tabular.candidates import EstimatorSpec

ONNX_OPSET_VERSION = 21
ML_ONNX_OPSET_VERSION = 4
# `onnx` stamps exported models with its own newest IR version, which runtimes reject
# outright if they are older. Nothing we emit needs more than the IR that pairs with
# opset 21, and pinning it keeps models loadable by the oldest onnxruntime we support:
# Python 3.10 caps onnxruntime at 1.23 (max IR 11) while installing onnx 1.22 (IR 13).
ONNX_IR_VERSION = 10
# This is promoted only after the benchmark comparison gate passes.
DATASET_AWARE_ORDERING_DEFAULT = False

EvalStrategy: TypeAlias = (
    Literal["auto", "holdout", "cv"] | BaseCrossValidator | Callable[..., Any] | None
)
ClassWeight: TypeAlias = Literal["none", "balanced"]
DECISION_METRICS = frozenset({"balanced_accuracy", "f1", "mcc"})
_Setting = TypeVar("_Setting")


class _Unset:
    pass


_UNSET = _Unset()


def _value_or_default(value: _Setting | _Unset, default: _Setting) -> _Setting:
    return default if isinstance(value, _Unset) else value


def _validate_positive_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


@runtime_checkable
class CandidateSource(Protocol):
    def get_candidates(
        self,
        task: str,
        *,
        X: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
        groups: npt.ArrayLike | None = None,
        n_classes: int | None = None,
        n_splits: int = 5,
        time_limit: float | None = None,
        random_state: int = 42,
        schema: DatasetSchema | None = None,
        dataset_aware_ordering: bool = False,
        config: RunConfig | None = None,
    ) -> tuple[EstimatorSpec, ...]: ...


@dataclass(frozen=True)
class PortfolioSource:
    specs: tuple[EstimatorSpec, ...] | None = None
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if self.specs is not None:
            specs = tuple(self.specs)
            if not specs:
                raise ValueError("PortfolioSource specs must not be empty")
            object.__setattr__(self, "specs", specs)
        if self.max_candidates is not None:
            _validate_positive_integer(self.max_candidates, "max_candidates")

    def get_candidates(
        self,
        task: str,
        *,
        X: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
        groups: npt.ArrayLike | None = None,
        n_classes: int | None = None,
        n_splits: int = 5,
        time_limit: float | None = None,
        random_state: int = 42,
        schema: DatasetSchema | None = None,
        dataset_aware_ordering: bool = False,
        config: RunConfig | None = None,
    ) -> tuple[EstimatorSpec, ...]:
        del X, groups, n_splits, time_limit, config
        if self.specs is None:
            from falcon.tabular.candidates import default_portfolio

            specs = default_portfolio(task, n_classes=n_classes)
        else:
            specs = self.specs
        if dataset_aware_ordering:
            if y is None or schema is None:
                raise ValueError(
                    "Dataset-aware ordering requires targets and a dataset schema"
                )
            from falcon.tabular.portfolio_ordering import (
                extract_dataset_meta_features,
                reorder_portfolio,
            )

            meta_features = extract_dataset_meta_features(y, schema)
            specs = reorder_portfolio(
                specs,
                meta_features,
                task,
                random_state=random_state,
            )
        if self.max_candidates is None:
            return specs
        return specs[: self.max_candidates]


@dataclass(frozen=True)
class HPOSource:
    family: str
    n_trials: int = 20
    top_n: int = 1
    time_budget_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("family must be a non-empty string")
        _validate_positive_integer(self.n_trials, "n_trials")
        _validate_positive_integer(self.top_n, "top_n")
        if self.top_n > self.n_trials:
            raise ValueError("top_n must not exceed n_trials")
        if (
            not _is_finite_number(self.time_budget_fraction)
            or not 0 < self.time_budget_fraction < 1
        ):
            raise ValueError("time_budget_fraction must be between zero and one")

    def get_candidates(
        self,
        task: str,
        *,
        X: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
        groups: npt.ArrayLike | None = None,
        n_classes: int | None = None,
        n_splits: int = 5,
        time_limit: float | None = None,
        random_state: int = 42,
        schema: DatasetSchema | None = None,
        dataset_aware_ordering: bool = False,
        config: RunConfig | None = None,
    ) -> tuple[EstimatorSpec, ...]:
        del n_classes, schema, dataset_aware_ordering
        if X is None or y is None:
            raise ValueError("HPO candidate generation requires training data")
        from falcon.tabular.hpo import generate_hpo_candidates

        resolved = RunConfig() if config is None else config
        return generate_hpo_candidates(
            task,
            X,
            y,
            groups=groups,
            family=self.family,
            n_trials=self.n_trials,
            top_n=self.top_n,
            n_splits=n_splits,
            time_limit=time_limit,
            time_budget_fraction=self.time_budget_fraction,
            random_state=random_state,
            class_weight=resolved.class_weight,
            prior_correct=resolved.prior_correct,
        )


@dataclass(frozen=True, init=False)
class RunConfig:
    candidate_sources: tuple[CandidateSource, ...] = field(
        default_factory=lambda: (PortfolioSource(),)
    )
    ensemble_enabled: bool = True
    ensemble_max_iterations: int = 100
    plateau_enabled: bool = True
    plateau_patience: int = 3
    plateau_tolerance: float = 1e-4
    oof_folds: int = 5
    eval_strategy: EvalStrategy = "auto"
    time_limit: float | None = None
    random_state: int = 42
    dataset_aware_ordering: bool = DATASET_AWARE_ORDERING_DEFAULT
    calibrate: bool = False
    conformal_alpha: float | None = None
    impute_missing: bool = True
    class_weight: ClassWeight = "none"
    decision_metric: str | None = "balanced_accuracy"
    _provided_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        candidate_sources: Sequence[CandidateSource] | _Unset = _UNSET,
        ensemble_enabled: bool | _Unset = _UNSET,
        ensemble_max_iterations: int | _Unset = _UNSET,
        plateau_enabled: bool | _Unset = _UNSET,
        plateau_patience: int | _Unset = _UNSET,
        plateau_tolerance: float | _Unset = _UNSET,
        oof_folds: int | _Unset = _UNSET,
        eval_strategy: EvalStrategy | _Unset = _UNSET,
        time_limit: float | None | _Unset = _UNSET,
        random_state: int | _Unset = _UNSET,
        dataset_aware_ordering: bool | _Unset = _UNSET,
        calibrate: bool | _Unset = _UNSET,
        conformal_alpha: float | None | _Unset = _UNSET,
        impute_missing: bool | _Unset = _UNSET,
        class_weight: ClassWeight | _Unset = _UNSET,
        decision_metric: str | None | _Unset = _UNSET,
    ) -> None:
        settings = {
            "candidate_sources": candidate_sources,
            "ensemble_enabled": ensemble_enabled,
            "ensemble_max_iterations": ensemble_max_iterations,
            "plateau_enabled": plateau_enabled,
            "plateau_patience": plateau_patience,
            "plateau_tolerance": plateau_tolerance,
            "oof_folds": oof_folds,
            "eval_strategy": eval_strategy,
            "time_limit": time_limit,
            "random_state": random_state,
            "dataset_aware_ordering": dataset_aware_ordering,
            "calibrate": calibrate,
            "conformal_alpha": conformal_alpha,
            "impute_missing": impute_missing,
            "class_weight": class_weight,
            "decision_metric": decision_metric,
        }
        object.__setattr__(
            self,
            "candidate_sources",
            _value_or_default(candidate_sources, (PortfolioSource(),)),
        )
        object.__setattr__(
            self,
            "ensemble_enabled",
            _value_or_default(ensemble_enabled, True),
        )
        object.__setattr__(
            self,
            "ensemble_max_iterations",
            _value_or_default(ensemble_max_iterations, 100),
        )
        object.__setattr__(
            self,
            "plateau_enabled",
            _value_or_default(plateau_enabled, True),
        )
        object.__setattr__(
            self,
            "plateau_patience",
            _value_or_default(plateau_patience, 3),
        )
        object.__setattr__(
            self,
            "plateau_tolerance",
            _value_or_default(plateau_tolerance, 1e-4),
        )
        object.__setattr__(self, "oof_folds", _value_or_default(oof_folds, 5))
        object.__setattr__(
            self,
            "eval_strategy",
            _value_or_default(eval_strategy, "auto"),
        )
        object.__setattr__(
            self,
            "time_limit",
            _value_or_default(time_limit, None),
        )
        object.__setattr__(
            self,
            "random_state",
            _value_or_default(random_state, 42),
        )
        object.__setattr__(
            self,
            "dataset_aware_ordering",
            _value_or_default(
                dataset_aware_ordering,
                DATASET_AWARE_ORDERING_DEFAULT,
            ),
        )
        object.__setattr__(
            self,
            "calibrate",
            _value_or_default(calibrate, False),
        )
        object.__setattr__(
            self,
            "conformal_alpha",
            _value_or_default(conformal_alpha, None),
        )
        object.__setattr__(
            self,
            "impute_missing",
            _value_or_default(impute_missing, True),
        )
        object.__setattr__(
            self,
            "class_weight",
            _value_or_default(class_weight, "none"),
        )
        object.__setattr__(
            self,
            "decision_metric",
            _value_or_default(decision_metric, "balanced_accuracy"),
        )
        object.__setattr__(
            self,
            "_provided_fields",
            frozenset(
                name
                for name, value in settings.items()
                if not isinstance(value, _Unset)
            ),
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        candidate_sources = tuple(self.candidate_sources)
        if not candidate_sources:
            raise ValueError("candidate_sources must not be empty")
        if not all(isinstance(source, CandidateSource) for source in candidate_sources):
            raise ValueError(
                "candidate_sources must contain only CandidateSource instances"
            )
        object.__setattr__(self, "candidate_sources", candidate_sources)

        for name, value in (
            ("ensemble_enabled", self.ensemble_enabled),
            ("plateau_enabled", self.plateau_enabled),
            ("dataset_aware_ordering", self.dataset_aware_ordering),
            ("calibrate", self.calibrate),
            ("impute_missing", self.impute_missing),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

        _validate_positive_integer(
            self.ensemble_max_iterations,
            "ensemble_max_iterations",
        )
        _validate_positive_integer(self.plateau_patience, "plateau_patience")
        if not _is_finite_number(self.plateau_tolerance) or self.plateau_tolerance < 0:
            raise ValueError("plateau_tolerance must be a finite non-negative value")
        _validate_positive_integer(self.oof_folds, "oof_folds", minimum=2)

        if self.eval_strategy is None:
            valid_strategy = True
        elif isinstance(self.eval_strategy, str):
            valid_strategy = self.eval_strategy in {"auto", "holdout", "cv"}
        else:
            valid_strategy = isinstance(
                self.eval_strategy, BaseCrossValidator
            ) or callable(self.eval_strategy)
        if not valid_strategy:
            raise ValueError(
                "eval_strategy must be 'auto', 'holdout', 'cv', None, a callable, "
                "or a BaseCrossValidator"
            )
        if self.time_limit is not None and (
            not _is_finite_number(self.time_limit) or self.time_limit <= 0
        ):
            raise ValueError("time_limit must be a finite value greater than zero")
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer")
        if self.conformal_alpha is not None and (
            not _is_finite_number(self.conformal_alpha)
            or not 0 < self.conformal_alpha < 1
        ):
            raise ValueError("conformal_alpha must be between zero and one")
        if self.class_weight not in {"none", "balanced"}:
            raise ValueError("class_weight must be either 'none' or 'balanced'")
        if (
            self.decision_metric is not None
            and self.decision_metric not in DECISION_METRICS
        ):
            raise ValueError(
                "decision_metric must be None or one of "
                f"{', '.join(sorted(DECISION_METRICS))}"
            )

    @property
    def prior_correct(self) -> bool:
        """Whether OOF selection scores decisions at `argmax p/pi` instead of `argmax p`."""
        return self.class_weight != "balanced" and self.decision_metric is not None

    def replaced(self, **overrides: Any) -> RunConfig:
        """Like `dataclasses.replace`, but carrying `_provided_fields` forward.

        `replace` passes every field explicitly, which would otherwise mark an
        untouched default as an explicit choice.
        """
        updated = replace(self, **overrides)
        object.__setattr__(
            updated,
            "_provided_fields",
            self._provided_fields | frozenset(overrides),
        )
        return updated

    def merged_with(self, overrides: RunConfig) -> RunConfig:
        return self.replaced(
            **{name: getattr(overrides, name) for name in overrides._provided_fields},
        )


__all__ = [
    "CandidateSource",
    "ClassWeight",
    "DATASET_AWARE_ORDERING_DEFAULT",
    "DECISION_METRICS",
    "EvalStrategy",
    "HPOSource",
    "ML_ONNX_OPSET_VERSION",
    "ONNX_IR_VERSION",
    "ONNX_OPSET_VERSION",
    "PortfolioSource",
    "RunConfig",
]
