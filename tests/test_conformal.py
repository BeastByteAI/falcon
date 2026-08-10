from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt
from sklearn.datasets import make_regression

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.runtime import Runtime
from falcon.tabular.candidates import EstimatorSpec
from falcon.tabular.conformal import fit_conformal_quantile
from tests.fnnx_conformance import assert_fnnx_conforms, extract_fnnx_graph


def _conformal_config(*, alpha: float = 0.1) -> RunConfig:
    return RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(
                    EstimatorSpec("ridge", "linear", {"alpha": 1.0}),
                    EstimatorSpec("ridge-regularized", "linear", {"alpha": 10.0}),
                )
            ),
        ),
        ensemble_enabled=True,
        ensemble_max_iterations=5,
        plateau_enabled=False,
        oof_folds=5,
        eval_strategy="holdout",
        conformal_alpha=alpha,
    )


def _regression_frame() -> tuple[pd.DataFrame, list[str]]:
    X, y = make_regression(
        n_samples=800,
        n_features=8,
        n_informative=6,
        noise=18.0,
        random_state=41,
    )
    features = [f"feature_{index}" for index in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=features)
    frame["target"] = y
    return frame, features


def _manifest(bundle: bytes) -> dict[str, Any]:
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:") as archive:
        member = archive.extractfile("manifest.json")
        assert member is not None
        return json.load(member)


def test_conformal_interval_round_trips_and_has_expected_coverage(
    tmp_path: Path,
) -> None:
    frame, features = _regression_frame()
    predictor = Predictor(
        "tabular_regression",
        config=_conformal_config(),
        random_state=17,
    ).fit(frame, features=features, target="target")
    assert predictor._eval_indices is not None
    evaluation = frame.iloc[predictor._eval_indices]
    evaluation_X = evaluation[features]

    artifact_path = tmp_path / "conformal.fnnx"
    bundle = predictor.save(artifact_path)
    graph = extract_fnnx_graph(bundle)
    assert_fnnx_conforms(graph)
    assert [output["name"] for output in _manifest(bundle)["outputs"]] == [
        "y_pred",
        "y_lower",
        "y_upper",
    ]
    conformal_ops = {
        node.op_type
        for node in graph.model.graph.node
        if "falcon_conformal" in node.name
    }
    assert conformal_ops == {"Add", "Sub"}

    runtime = Runtime(str(artifact_path))
    predictions = runtime.predict(evaluation_X)
    lower, upper = runtime.predict_interval(evaluation_X)

    np.testing.assert_allclose(
        predictions,
        predictor.predict(evaluation_X),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.all(lower <= predictions)
    assert np.all(predictions <= upper)
    coverage = np.mean(
        (evaluation["target"].to_numpy() >= lower)
        & (evaluation["target"].to_numpy() <= upper)
    )
    assert 0.84 <= coverage <= 0.96


def test_conformal_quantile_uses_finite_sample_correction() -> None:
    predictions = np.zeros(4, dtype=np.float32)
    targets = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    assert fit_conformal_quantile(predictions, targets, alpha=0.4) == 3.0


def test_conformal_splits_keep_groups_disjoint(
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
    group_ids = np.repeat(np.arange(16), 3)
    frame = pd.DataFrame(
        {
            "account": [f"account-{group_id}" for group_id in group_ids],
            "value": np.arange(len(group_ids), dtype=np.float64),
            "target": group_ids.astype(np.float64),
        }
    )
    config = RunConfig(
        candidate_sources=(
            PortfolioSource(specs=(EstimatorSpec("ridge", "linear", {"alpha": 1.0}),)),
        ),
        ensemble_enabled=False,
        plateau_enabled=False,
        oof_folds=4,
        eval_strategy=None,
        conformal_alpha=0.1,
    )

    Predictor("tabular_regression", config=config).fit(
        frame,
        target="target",
        group_by="account",
    )

    assert len(captured) == 1
    groups, splits = captured[0]
    for train_indices, eval_indices in splits:
        assert set(groups[train_indices]).isdisjoint(groups[eval_indices])


def test_classification_rejects_conformal_intervals() -> None:
    with pytest.raises(ValueError, match="regression"):
        Predictor(
            "tabular_classification",
            config=RunConfig(conformal_alpha=0.1),
        )
