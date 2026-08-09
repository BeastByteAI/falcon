from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.runtime import Runtime
from falcon.tabular.candidates import EstimatorSpec
from tests.fnnx_conformance import assert_fnnx_conforms, extract_fnnx_graph


def _calibration_config(*, calibrate: bool) -> RunConfig:
    return RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(
                    EstimatorSpec(
                        "linear",
                        "linear",
                        {"C": 10_000.0, "max_iter": 500},
                    ),
                )
            ),
        ),
        ensemble_enabled=False,
        plateau_enabled=False,
        oof_folds=5,
        eval_strategy="holdout",
        calibrate=calibrate,
        # Temperature alone preserves labels; a tuned decision rule fitted on
        # calibrated versus raw OOF probabilities legitimately would not.
        decision_metric=None,
    )


def _noisy_classification_frame() -> tuple[pd.DataFrame, list[str]]:
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        class_sep=0.5,
        flip_y=0.2,
        random_state=41,
    )
    features = [f"feature_{index}" for index in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=features)
    frame["target"] = np.where(y, "yes", "no")
    return frame, features


def test_temperature_calibration_preserves_labels_and_round_trips(
    tmp_path: Path,
) -> None:
    frame, features = _noisy_classification_frame()
    uncalibrated = Predictor(
        "tabular_classification",
        config=_calibration_config(calibrate=False),
        random_state=17,
    ).fit(frame, features=features, target="target")
    calibrated = Predictor(
        "tabular_classification",
        config=_calibration_config(calibrate=True),
        random_state=17,
    ).fit(frame, features=features, target="target")

    assert uncalibrated._eval_indices is not None
    assert calibrated._eval_indices is not None
    np.testing.assert_array_equal(
        calibrated._eval_indices,
        uncalibrated._eval_indices,
    )
    evaluation = frame.iloc[calibrated._eval_indices]
    evaluation_X = evaluation[features]
    uncalibrated_probabilities = uncalibrated.predict_proba(evaluation_X)
    calibrated_probabilities = calibrated.predict_proba(evaluation_X)
    calibrated_predictions = calibrated.predict(evaluation_X)

    np.testing.assert_array_equal(
        calibrated_predictions,
        uncalibrated.predict(evaluation_X),
    )
    assert not np.allclose(
        calibrated_probabilities,
        uncalibrated_probabilities,
        rtol=1e-5,
        atol=1e-6,
    )
    assert calibrated.classes_ is not None
    calibrated_loss = log_loss(
        evaluation["target"],
        calibrated_probabilities,
        labels=calibrated.classes_,
    )
    uncalibrated_loss = log_loss(
        evaluation["target"],
        uncalibrated_probabilities,
        labels=calibrated.classes_,
    )
    assert calibrated_loss <= uncalibrated_loss + 1e-6

    artifact_path = tmp_path / "calibrated.fnnx"
    bundle = calibrated.save(artifact_path)
    graph = extract_fnnx_graph(bundle)
    assert_fnnx_conforms(graph)
    calibration_ops = {
        node.op_type
        for node in graph.model.graph.node
        if "falcon_temperature" in node.name
    }
    assert calibration_ops == {"Clip", "Div", "Log", "Softmax"}

    runtime = Runtime(str(artifact_path))
    np.testing.assert_array_equal(
        runtime.predict(evaluation_X),
        calibrated_predictions,
    )
    np.testing.assert_allclose(
        runtime.predict_proba(evaluation_X),
        calibrated_probabilities,
        rtol=1e-5,
        atol=1e-6,
    )


def test_calibration_oof_splits_keep_groups_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from falcon.tabular import candidates

    original_out_of_fold_indices = candidates.out_of_fold_indices
    captured: list[
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
        captured.append((np.asarray(groups, dtype=np.int64), splits))
        return splits

    monkeypatch.setattr(
        candidates,
        "out_of_fold_indices",
        recording_out_of_fold_indices,
    )
    group_ids = np.repeat(np.arange(12), 3)
    frame = pd.DataFrame(
        {
            "account": [f"account-{group_id}" for group_id in group_ids],
            "value": np.arange(len(group_ids), dtype=np.float64),
            "target": np.where(group_ids % 2, "yes", "no"),
        }
    )
    config = RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(EstimatorSpec("linear", "linear", {"max_iter": 200}),)
            ),
        ),
        ensemble_enabled=False,
        plateau_enabled=False,
        oof_folds=3,
        eval_strategy=None,
        calibrate=True,
    )

    Predictor("tabular_classification", config=config).fit(
        frame,
        target="target",
        group_by="account",
    )

    assert len(captured) == 1
    groups, splits = captured[0]
    for train_indices, eval_indices in splits:
        assert set(groups[train_indices]).isdisjoint(groups[eval_indices])


def test_regression_rejects_probability_calibration() -> None:
    with pytest.raises(ValueError, match="classification"):
        Predictor(
            "tabular_regression",
            config=RunConfig(calibrate=True),
        )
