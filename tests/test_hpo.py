from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt

from falcon import Predictor
from falcon.config import HPOSource, PortfolioSource, RunConfig
from falcon.runtime import Runtime
from falcon.tabular import hpo
from falcon.tabular.candidates import EstimatorSpec, GreedyWeightedEnsemble

optuna = pytest.importorskip("optuna")

if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import Trial


def _regression_frame(n_rows: int = 60) -> pd.DataFrame:
    values = np.linspace(-3.0, 3.0, n_rows)
    return pd.DataFrame(
        {
            "first": values,
            "second": np.square(values),
            "target": 2.5 * values - 0.4 * np.square(values),
        }
    )


def test_hpo_source_uses_grouped_cv_default_trial_pruner_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = np.column_stack(
        (
            np.repeat(np.arange(12), 2),
            np.tile(np.asarray([0.0, 1.0]), 12),
        )
    ).astype(np.float32)
    y = (1.5 * X[:, 0] - X[:, 1]).astype(np.float32)
    groups = np.repeat(np.arange(12), 2)
    observed_splits: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = []
    studies: list[Study] = []
    optimize_timeouts: list[float | None] = []
    original_splitter = hpo.out_of_fold_indices
    original_create_study = optuna.create_study
    original_optimize = optuna.study.Study.optimize

    def capture_splits(
        features: npt.NDArray[Any],
        targets: npt.NDArray[Any],
        task: str,
        group_values: npt.ArrayLike | None = None,
        *,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        splits = original_splitter(
            features,
            targets,
            task,
            group_values,
            n_splits=n_splits,
            random_state=random_state,
        )
        observed_splits.extend(splits)
        return splits

    def capture_study(**kwargs: Any) -> Study:
        study = original_create_study(**kwargs)
        studies.append(study)
        return study

    def capture_optimize(
        study: Study,
        objective: Callable[[Trial], float],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        timeout = kwargs.get("timeout")
        optimize_timeouts.append(None if timeout is None else float(timeout))
        original_optimize(study, objective, *args, **kwargs)

    monkeypatch.setattr(hpo, "out_of_fold_indices", capture_splits)
    monkeypatch.setattr(optuna, "create_study", capture_study)
    monkeypatch.setattr(optuna.study.Study, "optimize", capture_optimize)

    specs = HPOSource(
        family="linear",
        n_trials=2,
        top_n=2,
        time_budget_fraction=0.4,
    ).get_candidates(
        "tabular_regression",
        X=X,
        y=y,
        groups=groups,
        n_splits=4,
        time_limit=10.0,
        random_state=17,
    )

    assert len(observed_splits) == 4
    for train_indices, eval_indices in observed_splits:
        assert set(groups[train_indices]).isdisjoint(groups[eval_indices])
    assert optimize_timeouts == [4.0]
    assert isinstance(studies[0].pruner, optuna.pruners.MedianPruner)
    assert studies[0].trials[0].params == {"alpha": 1.0}
    assert len(specs) == 2
    assert len({tuple(spec.parameters.items()) for spec in specs}) == 2
    assert all(spec.name.startswith("hpo_linear_trial_") for spec in specs)


def test_hpo_candidates_join_portfolio_oof_ensemble_and_export(
    tmp_path: Path,
) -> None:
    frame = _regression_frame()
    config = RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(EstimatorSpec("portfolio-ridge", "linear", {"alpha": 20.0}),)
            ),
            HPOSource(family="linear", n_trials=2),
        ),
        ensemble_enabled=True,
        ensemble_max_iterations=5,
        plateau_enabled=False,
        oof_folds=3,
        eval_strategy=None,
    )
    predictor = Predictor("tabular_regression", config=config).fit(
        frame,
        features=["first", "second"],
        target="target",
    )
    leaderboard = predictor.leaderboard()

    assert list(leaderboard["candidate"])[0] == "portfolio-ridge"
    assert any(
        name.startswith("hpo_linear_trial_") for name in leaderboard["candidate"]
    )
    assert predictor._learner is not None
    assert isinstance(predictor._learner.model, GreedyWeightedEnsemble)
    assert predictor._learner.model.score >= float(leaderboard["score"].max())

    artifact_path = tmp_path / "hpo-ensemble.fnnx"
    predictor.save(artifact_path)
    inputs = frame[["first", "second"]].iloc[:8]
    np.testing.assert_allclose(
        Runtime(str(artifact_path)).predict(inputs),
        predictor.predict(inputs),
        rtol=1e-5,
        atol=1e-5,
    )


def test_hpo_only_source_trains_without_ensembling() -> None:
    frame = _regression_frame(40)
    config = RunConfig(
        candidate_sources=(HPOSource(family="linear", n_trials=2, top_n=2),),
        ensemble_enabled=False,
        plateau_enabled=False,
        oof_folds=2,
        eval_strategy=None,
    )

    predictor = Predictor("tabular_regression", config=config).fit(
        frame,
        features=["first", "second"],
        target="target",
    )

    leaderboard = predictor.leaderboard()
    assert len(leaderboard) == 2
    assert all(
        name.startswith("hpo_linear_trial_") for name in leaderboard["candidate"]
    )
    assert sorted(leaderboard["weight"]) == [0.0, 1.0]
    assert predictor.predict(frame[["first", "second"]].iloc[:5]).shape == (5,)
    assert predictor.save()


def test_hpo_classification_candidate_produces_probabilities() -> None:
    values = np.linspace(-2.0, 2.0, 48)
    frame = pd.DataFrame(
        {
            "first": values,
            "second": np.sin(values),
            "target": np.where(values > 0, "positive", "negative"),
        }
    )
    config = RunConfig(
        candidate_sources=(HPOSource(family="linear", n_trials=1),),
        ensemble_enabled=False,
        plateau_enabled=False,
        oof_folds=2,
        eval_strategy=None,
    )

    predictor = Predictor("tabular_classification", config=config).fit(
        frame,
        features=["first", "second"],
        target="target",
    )
    probabilities = predictor.predict_proba(frame[["first", "second"]].iloc[:6])

    assert probabilities.shape == (6, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_hpo_source_forwards_the_run_config_weighting_and_scoring_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def capture(*args: Any, **kwargs: Any) -> tuple[EstimatorSpec, ...]:
        captured.append(kwargs)
        return (EstimatorSpec("hpo_stub", "linear", {"alpha": 1.0}),)

    monkeypatch.setattr(hpo, "generate_hpo_candidates", capture)
    frame = _regression_frame(20)
    X = frame[["first", "second"]].to_numpy(dtype=np.float64)
    y = frame["target"].to_numpy(dtype=np.float64)
    source = HPOSource(family="linear", n_trials=1)

    source.get_candidates("tabular_regression", X=X, y=y)
    source.get_candidates(
        "tabular_regression",
        X=X,
        y=y,
        config=RunConfig(class_weight="balanced"),
    )

    assert captured[0]["class_weight"] == "none"
    assert captured[0]["prior_correct"] is True
    assert captured[1]["class_weight"] == "balanced"
    assert captured[1]["prior_correct"] is False


def test_hpo_trials_train_with_the_configured_class_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from falcon.tabular import candidates

    weighted_calls: list[int] = []
    original = candidates.compute_sample_weight

    def counting_compute_sample_weight(**kwargs: Any) -> Any:
        weighted_calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(
        candidates,
        "compute_sample_weight",
        counting_compute_sample_weight,
    )
    values = np.linspace(-2.0, 2.0, 40)
    frame = pd.DataFrame(
        {
            "first": values,
            "second": np.sin(values),
            "target": np.where(values > 0.6, "positive", "negative"),
        }
    )
    config = RunConfig(
        candidate_sources=(HPOSource(family="linear", n_trials=1),),
        ensemble_enabled=False,
        plateau_enabled=False,
        oof_folds=2,
        eval_strategy=None,
    )

    Predictor("tabular_classification", config=config).fit(
        frame,
        features=["first", "second"],
        target="target",
    )
    assert not weighted_calls

    Predictor(
        "tabular_classification",
        config=RunConfig(
            candidate_sources=config.candidate_sources,
            ensemble_enabled=False,
            plateau_enabled=False,
            oof_folds=2,
            eval_strategy=None,
            class_weight="balanced",
        ),
    ).fit(frame, features=["first", "second"], target="target")
    assert weighted_calls


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"family": ""}, "family"),
        ({"family": "linear", "n_trials": 0}, "n_trials"),
        ({"family": "linear", "top_n": 0}, "top_n"),
        ({"family": "linear", "n_trials": 1, "top_n": 2}, "top_n"),
        ({"family": "linear", "time_budget_fraction": 0.0}, "time_budget_fraction"),
        ({"family": "linear", "time_budget_fraction": 1.0}, "time_budget_fraction"),
    ],
)
def test_hpo_source_rejects_invalid_configuration(
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HPOSource(**options)  # type: ignore[arg-type]
