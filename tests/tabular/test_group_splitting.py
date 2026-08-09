from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt
from sklearn.model_selection import BaseCrossValidator

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.sklapi import FalconClassifier
from falcon.tabular.candidates import EstimatorSpec
from falcon.tabular.splitting import (
    callable_holdout_indices,
    cross_validation_indices,
    holdout_indices,
    resolve_groups,
)


def _assert_groups_do_not_straddle(
    groups: npt.NDArray[np.int64],
    train_indices: npt.NDArray[np.int64],
    eval_indices: npt.NDArray[np.int64],
) -> None:
    assert set(groups[train_indices]).isdisjoint(groups[eval_indices])


def _duplicate_classification_data() -> tuple[
    npt.NDArray[np.object_], npt.NDArray[np.int64]
]:
    group_ids = np.repeat(np.arange(12), 2)
    X = np.column_stack((group_ids, group_ids % 3)).astype(np.object_)
    y = (group_ids % 2).astype(np.int64)
    return X, y


def _single_linear_config() -> RunConfig:
    return RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(EstimatorSpec("linear", "linear", {"max_iter": 200}),)
            ),
        ),
        ensemble_enabled=False,
    )


def test_duplicate_rows_never_straddle_default_holdout_or_cv() -> None:
    X, y = _duplicate_classification_data()
    groups = resolve_groups(X, ("group", "category"))

    train_indices, eval_indices = holdout_indices(X, y, task="tabular_classification")
    _assert_groups_do_not_straddle(groups, train_indices, eval_indices)

    for train_indices, eval_indices in cross_validation_indices(
        X, y, task="tabular_classification"
    ):
        _assert_groups_do_not_straddle(groups, train_indices, eval_indices)


def test_named_groups_override_automatic_groups_and_remain_features() -> None:
    frame = pd.DataFrame(
        {
            "account": np.repeat([f"account-{index}" for index in range(8)], 4),
            "value": np.arange(32, dtype=np.float64),
            "target": np.tile([0, 0, 1, 1], 8),
        }
    )
    predictor = Predictor(
        "tabular_classification",
        config=_single_linear_config(),
        eval_strategy="holdout",
    ).fit(frame, group_by="account")

    assert predictor._training_data is not None
    assert predictor._fit_indices is not None
    assert predictor._eval_indices is not None
    assert predictor._training_data.schema.column_names == ("account", "value")
    groups = predictor._training_data.groups
    _assert_groups_do_not_straddle(
        groups,
        predictor._fit_indices,
        predictor._eval_indices,
    )


def test_sklearn_fit_accepts_named_groups() -> None:
    frame = pd.DataFrame(
        {
            "account": np.repeat([f"account-{index}" for index in range(8)], 4),
            "value": np.arange(32, dtype=np.float64),
        }
    )
    target = np.tile([0, 0, 1, 1], 8)
    estimator = FalconClassifier(
        preset=_single_linear_config(),
        eval_strategy="holdout",
    )

    estimator.fit(frame, target, group_by="account")

    predictor = estimator.predictor_
    assert predictor._training_data is not None
    assert predictor._fit_indices is not None
    assert predictor._eval_indices is not None
    groups = predictor._training_data.groups
    assert set(groups[predictor._fit_indices]).isdisjoint(
        groups[predictor._eval_indices]
    )


def test_explicit_groups_are_respected_after_ingestion_drops_rows() -> None:
    frame = pd.DataFrame(
        {
            "value": np.arange(13, dtype=np.float64),
            "target": [*np.tile([0, 1], 6), np.nan],
        }
    )
    explicit_groups = np.asarray([*np.repeat(np.arange(6), 2), 99])
    predictor = Predictor(
        "tabular_classification",
        config=_single_linear_config(),
        eval_strategy="holdout",
    ).fit(frame, group_by=explicit_groups)

    assert predictor._training_data is not None
    assert predictor._fit_indices is not None
    assert predictor._eval_indices is not None
    assert predictor._training_data.schema.column_names == ("value",)
    groups = predictor._training_data.groups
    assert groups.size == 12
    assert len(np.unique(groups)) == 6
    _assert_groups_do_not_straddle(
        groups,
        predictor._fit_indices,
        predictor._eval_indices,
    )


def test_classification_cv_is_stratified_by_group() -> None:
    X, y = _duplicate_classification_data()
    groups = resolve_groups(X, ("group", "category"), group_by="group")

    for train_indices, eval_indices in cross_validation_indices(
        X,
        y,
        task="tabular_classification",
        groups=groups,
        n_splits=4,
    ):
        _assert_groups_do_not_straddle(groups, train_indices, eval_indices)
        assert set(y[train_indices]) == {0, 1}
        assert set(y[eval_indices]) == {0, 1}


class _CapturingGroupCV(BaseCrossValidator):
    def __init__(self) -> None:
        self.groups_seen: npt.NDArray[Any] | None = None

    def get_n_splits(
        self,
        X: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> int:
        return 2

    def split(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
    ) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        assert groups is not None
        self.groups_seen = groups.copy()
        unique_groups = np.unique(groups)
        for eval_groups in np.array_split(unique_groups, 2):
            eval_mask = np.isin(groups, eval_groups)
            yield np.flatnonzero(~eval_mask), np.flatnonzero(eval_mask)


def test_custom_cross_validator_receives_groups() -> None:
    X, y = _duplicate_classification_data()
    groups = resolve_groups(X, ("group", "category"))
    cv = _CapturingGroupCV()

    splits = cross_validation_indices(
        X, y, task="tabular_classification", groups=groups, cv=cv
    )

    assert len(splits) == 2
    assert cv.groups_seen is not None
    np.testing.assert_array_equal(cv.groups_seen, groups)


def test_callable_splitter_receives_groups_and_is_validated() -> None:
    X, y = _duplicate_classification_data()
    groups = resolve_groups(X, ("group", "category"))
    captured_groups: npt.NDArray[Any] | None = None

    def split(
        split_X: npt.NDArray[Any],
        split_y: npt.NDArray[Any],
        split_groups: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        nonlocal captured_groups
        captured_groups = split_groups.copy()
        return holdout_indices(
            split_X,
            split_y,
            task="tabular_classification",
            groups=split_groups,
        )

    train_indices, eval_indices = callable_holdout_indices(split, X, y, groups)

    assert captured_groups is not None
    np.testing.assert_array_equal(captured_groups, groups)
    _assert_groups_do_not_straddle(groups, train_indices, eval_indices)

    def invalid_split(
        split_X: npt.NDArray[Any],
        split_y: npt.NDArray[Any],
        split_groups: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        del split_X, split_y, split_groups
        return np.arange(0, len(X), 2), np.arange(1, len(X), 2)

    with pytest.raises(ValueError, match="places group .* on both sides"):
        callable_holdout_indices(invalid_split, X, y, groups)
