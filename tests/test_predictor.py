from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt

import falcon
from falcon import AutoML, Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.runtime import Runtime
from falcon.tabular.candidates import EstimatorSpec
from falcon.types import ColumnTypes
from tests.fnnx_conformance import assert_fnnx_conforms, extract_fnnx_graph


def _linear_config(
    *,
    task: str,
    eval_strategy: str | None = None,
    ensemble: bool = False,
) -> RunConfig:
    parameters: dict[str, object]
    if task == "tabular_classification":
        parameters = {"C": 1.0, "max_iter": 200}
    else:
        parameters = {"alpha": 1.0}
    specs = [EstimatorSpec("linear-primary", "linear", parameters)]
    if ensemble:
        secondary = {**parameters, "C": 0.5} if "C" in parameters else {"alpha": 2.0}
        specs.append(EstimatorSpec("linear-secondary", "linear", secondary))
    return RunConfig(
        candidate_sources=(PortfolioSource(specs=tuple(specs)),),
        ensemble_enabled=ensemble,
        ensemble_max_iterations=5,
        plateau_enabled=False,
        oof_folds=3,
        eval_strategy=eval_strategy,
    )


def _classification_frame(n_rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    first = rng.normal(size=n_rows)
    second = rng.normal(size=n_rows)
    return pd.DataFrame(
        {
            "first": first,
            "second": second,
            "target": np.where(first + 0.2 * second > 0, "yes", "no"),
        }
    )


def _mixed_frame(task: str) -> tuple[pd.DataFrame, list[str]]:
    sample_count = 128
    sample_indices = np.arange(sample_count)
    features: dict[str, npt.NDArray[Any]] = {
        "numeric": np.linspace(-4.0, 7.0, sample_count).astype(np.object_),
        "categorical_low": np.asarray(
            [f"group-{index % 4}" for index in sample_indices], dtype=np.object_
        ),
        "categorical_high": np.asarray(
            [f"category-{index:03d}" for index in sample_indices], dtype=np.object_
        ),
        "date": np.asarray(
            (
                pd.Timestamp("2020-01-01") + pd.to_timedelta(sample_indices, unit="D")
            ).strftime("%Y-%m-%d"),
            dtype=np.object_,
        ),
        "datetime": np.asarray(
            (
                pd.Timestamp("2020-01-01") + pd.to_timedelta(sample_indices, unit="h")
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            dtype=np.object_,
        ),
        "text": np.asarray(
            [
                "falcon document sample "
                f"{index} contains several useful words about "
                f"{'alpha' if index % 2 else 'beta'}"
                for index in sample_indices
            ],
            dtype=np.object_,
        ),
    }
    for values in features.values():
        values[[5, 37]] = np.nan
    if task == "tabular_classification":
        target: npt.NDArray[Any] = np.where(
            sample_indices % 2, "positive", "negative"
        ).astype(np.object_)
    else:
        target = (0.4 * sample_indices + np.sin(sample_indices)).astype(np.object_)
    target[-1] = np.nan
    return pd.DataFrame({**features, "target": target}), list(features)


def test_predictor_is_the_public_api_and_holdout_does_not_mutate_training_data() -> (
    None
):
    frame = _classification_frame()
    predictor = Predictor(
        "tabular_classification",
        config=_linear_config(task="tabular_classification"),
        eval_strategy="holdout",
        random_state=7,
    )

    result = predictor.fit(frame, features=["first", "second"], target="target")

    assert result is predictor
    assert falcon.Predictor is Predictor
    assert predictor._training_data is not None
    assert predictor._fit_indices is not None
    assert predictor._eval_indices is not None
    assert predictor._training_data.X.shape == (len(frame), 2)
    assert len(predictor._fit_indices) + len(predictor._eval_indices) == len(frame)
    assert set(predictor._training_data.groups[predictor._fit_indices]).isdisjoint(
        predictor._training_data.groups[predictor._eval_indices]
    )
    assert set(predictor._performance_metrics) == {"train", "eval"}


@pytest.mark.parametrize(
    ("task", "preset"),
    [
        pytest.param("tabular_classification", "balanced", id="classification"),
        pytest.param("tabular_regression", "fast", id="regression"),
    ],
)
def test_mixed_type_predictor_round_trip(
    task: str,
    preset: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame, features = _mixed_frame(task)
    with caplog.at_level("INFO", logger="falcon"):
        predictor = Predictor(
            task,
            preset=preset,
            eval_strategy=None,
            random_state=17,
        ).fit(frame, features=features, target="target")
    assert predictor._training_data is not None
    assert predictor._training_data.X.shape == (len(frame) - 1, len(features))
    assert pd.isna(predictor._training_data.X).any()
    assert predictor._training_data.schema.column_types == (
        ColumnTypes.NUMERIC_REGULAR,
        ColumnTypes.CAT_LOW_CARD,
        ColumnTypes.CAT_HIGH_CARD,
        ColumnTypes.DATE_YMD_ISO8601,
        ColumnTypes.DATETIME_YMDHMS_ISO8601,
        ColumnTypes.TEXT_UTF8,
    )
    assert "Dropped 1 row with a missing target" in caplog.text

    inputs = frame.loc[[5, 37, 0], features].copy()
    inputs.loc[0, "categorical_low"] = "Z"
    inputs.loc[0, "categorical_high"] = "Z"
    artifact_path = tmp_path / f"mixed-{task}.fnnx"
    bundle = predictor.save(artifact_path)
    runtime = Runtime(str(artifact_path))

    assert bundle == artifact_path.read_bytes()
    assert predictor.evaluate(frame)["N_SAMPLES"] == len(frame) - 1
    assert_fnnx_conforms(extract_fnnx_graph(bundle))
    if task == "tabular_classification":
        np.testing.assert_array_equal(
            runtime.predict(inputs), predictor.predict(inputs)
        )
        native_probabilities = predictor.predict_proba(inputs)
        np.testing.assert_allclose(
            runtime.predict_proba(inputs),
            native_probabilities,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            native_probabilities.sum(axis=1),
            1.0,
            atol=1e-6,
        )
    else:
        assert len(predictor.leaderboard()) == 1
        np.testing.assert_allclose(
            runtime.predict(inputs),
            predictor.predict(inputs),
            rtol=1e-5,
            atol=1e-4,
        )


def test_predictor_classification_methods_and_artifact_probabilities_match(
    tmp_path: Path,
) -> None:
    frame = _classification_frame()
    predictor = Predictor(
        "tabular_classification",
        config=_linear_config(
            task="tabular_classification",
            ensemble=True,
        ),
    ).fit(frame, features=["first", "second"], target="target")
    inputs = frame[["first", "second"]].iloc[:12]

    native_predictions = predictor.predict(inputs)
    native_probabilities = predictor.predict_proba(inputs)
    evaluation = predictor.evaluate(frame)
    leaderboard = predictor.leaderboard()
    artifact_path = tmp_path / "predictor.fnnx"
    bundle = predictor.save(artifact_path)
    runtime = Runtime(str(artifact_path))

    assert evaluation["N_SAMPLES"] == len(frame)
    assert list(leaderboard["candidate"]) == [
        "linear-primary",
        "linear-secondary",
    ]
    assert len(predictor.feature_importance(n_repeats=2)) == 2
    assert bundle == artifact_path.read_bytes()
    np.testing.assert_array_equal(runtime.predict(inputs), native_predictions)
    np.testing.assert_allclose(
        runtime.predict_proba(inputs),
        native_probabilities,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(native_probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_time_budget_keeps_the_first_candidate_usable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    X = np.arange(120, dtype=np.float64).reshape(60, 2)
    y = 3 * X[:, 0] - X[:, 1]
    config = _linear_config(
        task="tabular_regression",
        eval_strategy=None,
        ensemble=True,
    )
    time_limit = float(np.finfo(float).eps)

    started_at = time.monotonic()
    with caplog.at_level("INFO", logger="falcon"):
        predictor = Predictor(
            "tabular_regression",
            config=config,
            time_limit=time_limit,
        ).fit((X, y))
    elapsed = time.monotonic() - started_at
    leaderboard = predictor.leaderboard()

    assert len(leaderboard) == 1
    assert elapsed <= float(leaderboard.iloc[0]["fit_time"]) + time_limit + 2.0
    assert "time limit is insufficient for the first candidate" in caplog.text
    assert predictor.predict(X[:4]).shape == (4,)
    assert predictor.save()


def test_balanced_preset_stops_on_a_reproducible_plateau(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from falcon.tabular import candidates

    specs = tuple(
        EstimatorSpec(f"linear-{index}", "linear", {"alpha": float(index + 1)})
        for index in range(4)
    )
    monkeypatch.setattr(
        candidates,
        "default_portfolio",
        lambda task, n_classes=None: specs,
    )
    X = np.arange(120, dtype=np.float64).reshape(60, 2)
    y = np.zeros(len(X), dtype=np.float64)

    def fit_predictor() -> Predictor:
        return Predictor(
            "tabular_regression",
            preset="balanced",
            eval_strategy=None,
            random_state=13,
        ).fit((X, y))

    with caplog.at_level("INFO", logger="falcon"):
        first = fit_predictor()
        second = fit_predictor()

    first_candidates = list(first.leaderboard()["candidate"])
    second_candidates = list(second.leaderboard()["candidate"])
    assert first_candidates == second_candidates
    assert len(first_candidates) < len(specs)
    assert "OOF score plateau" in caplog.text
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert first.save()


def test_predictor_is_reproducible_under_a_fixed_seed() -> None:
    frame = _classification_frame()
    config = _linear_config(
        task="tabular_classification",
        eval_strategy=None,
        ensemble=True,
    )

    predictions = [
        Predictor(
            "tabular_classification",
            config=config,
            random_state=19,
        )
        .fit(frame, features=["first", "second"], target="target")
        .predict(frame[["first", "second"]])
        for _ in range(2)
    ]

    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_named_groups_cover_holdout_and_oof_splits_and_remain_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from falcon.tabular import candidates

    spec = EstimatorSpec("linear", "linear", {"max_iter": 200})
    monkeypatch.setattr(
        candidates,
        "default_portfolio",
        lambda task, n_classes=None: (spec,),
    )
    original_out_of_fold_indices = candidates.out_of_fold_indices
    captured_splits: list[
        tuple[
            npt.NDArray[np.int64],
            list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
        ]
    ] = []

    def recording_out_of_fold_indices(
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        task: str,
        groups: npt.ArrayLike | None = None,
        *,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        splits = original_out_of_fold_indices(
            X,
            y,
            task,
            groups,
            n_splits=n_splits,
            random_state=random_state,
        )
        assert groups is not None
        captured_splits.append((np.asarray(groups).copy(), splits))
        return splits

    monkeypatch.setattr(
        candidates,
        "out_of_fold_indices",
        recording_out_of_fold_indices,
    )
    account_ids = np.repeat(np.arange(12), 3)
    frame = pd.DataFrame(
        {
            "account": [f"account-{account_id}" for account_id in account_ids],
            "value": np.arange(len(account_ids), dtype=np.float64),
            "target": np.where(account_ids % 2, "positive", "negative"),
        }
    )

    predictor = Predictor(
        "tabular_classification",
        preset="balanced",
        eval_strategy="holdout",
    ).fit(frame, target="target", group_by="account")

    assert predictor._training_data is not None
    assert predictor._fit_indices is not None
    assert predictor._eval_indices is not None
    training_data = predictor._training_data
    assert training_data.schema.column_names == ("account", "value")
    assert set(training_data.groups[predictor._fit_indices]).isdisjoint(
        training_data.groups[predictor._eval_indices]
    )
    assert len(captured_splits) == 1
    oof_groups, splits = captured_splits[0]
    for train_indices, eval_indices in splits:
        assert set(oof_groups[train_indices]).isdisjoint(oof_groups[eval_indices])
    assert predictor.predict(frame[["account", "value"]].iloc[:3]).shape == (3,)


def test_no_evaluation_strategy_uses_all_rows_and_still_predicts_and_saves() -> None:
    X = np.arange(120, dtype=np.float64).reshape(60, 2)
    y = 3 * X[:, 0] - X[:, 1]
    predictor = Predictor(
        "tabular_regression",
        config=_linear_config(task="tabular_regression", eval_strategy=None),
    ).fit((X, y))

    assert predictor._fit_indices is not None
    assert len(predictor._fit_indices) == len(X)
    assert set(predictor._performance_metrics) == {"train"}
    assert predictor.predict(X[:4]).shape == (4,)
    assert predictor.save()
    assert len(predictor.feature_importance(n_repeats=2)) == X.shape[1]
    with pytest.raises(RuntimeError, match="classification"):
        predictor.predict_proba(X[:4])


def test_automl_dynamic_evaluation_uses_test_data_and_returns_predictor() -> None:
    frame = _classification_frame()
    train = frame.iloc[:70].copy()
    test = frame.iloc[70:].copy()

    predictor = AutoML(
        task="tabular_classification",
        train_data=train,
        test_data=test,
        features=["first", "second"],
        target="target",
        config=_linear_config(
            task="tabular_classification",
            eval_strategy="holdout",
        ),
        save_model=False,
    )

    assert isinstance(predictor, Predictor)
    assert predictor.config.eval_strategy is None
    assert set(predictor._performance_metrics) == {"train", "test"}


def test_removed_automl_kwargs_and_legacy_presets_have_migration_errors() -> None:
    frame = _classification_frame(30)
    with pytest.raises(TypeError) as error:
        AutoML(
            task="tabular_classification",
            train_data=frame,
            manager_configuration={},
            save_model=False,
        )

    message = str(error.value)
    assert "manager_configuration" in message
    assert "removed" in message
    assert "RunConfig" in message

    with pytest.raises(TypeError, match="manager_configuration.*removed"):
        Predictor(
            "tabular_classification",
            manager_configuration={},
        )

    with pytest.raises(ValueError) as preset_error:
        Predictor("tabular_classification", preset="SuperLearner")
    assert all(
        preset in str(preset_error.value) for preset in ("fast", "balanced", "best")
    )
