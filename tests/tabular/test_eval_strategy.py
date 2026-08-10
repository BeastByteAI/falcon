from __future__ import annotations

from typing import Any, NoReturn

import numpy as np
import pytest
from numpy import typing as npt
from sklearn.model_selection import KFold

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.tabular.candidates import EstimatorSpec


class _BrokenKFold(KFold):
    def split(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError("pytest :: Broken KFold")


def _broken_split(*args: Any, **kwargs: Any) -> NoReturn:
    raise ValueError("pytest :: Broken split")


def _config() -> RunConfig:
    return RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(EstimatorSpec("linear", "linear", {"max_iter": 200}),)
            ),
        ),
        ensemble_enabled=False,
        oof_folds=3,
    )


def _classification_data(
    n_rows: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_rows, 2))
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def test_auto_eval_strategy() -> None:
    small_X, small_y = _classification_data(250)
    small = Predictor(
        "tabular_classification",
        config=_config(),
        eval_strategy="auto",
    ).fit((small_X, small_y))

    assert "eval_cv" in small._performance_metrics
    assert "eval" not in small._performance_metrics

    large_X, large_y = _classification_data(2_500)
    large = Predictor(
        "tabular_classification",
        config=_config(),
        eval_strategy="auto",
    ).fit((large_X, large_y))

    assert "eval" in large._performance_metrics
    assert "eval_cv" not in large._performance_metrics


@pytest.mark.parametrize(
    ("strategy", "expected_key"),
    [("cv", "eval_cv"), ("holdout", "eval")],
)
def test_named_eval_strategy(strategy: str, expected_key: str) -> None:
    X, y = _classification_data(100)
    predictor = Predictor(
        "tabular_classification",
        config=_config(),
        eval_strategy=strategy,
    ).fit((X, y))

    assert expected_key in predictor._performance_metrics


def test_custom_cv_eval_strategy() -> None:
    X, y = _classification_data(100)
    predictor = Predictor(
        "tabular_classification",
        config=_config(),
        eval_strategy=_BrokenKFold(n_splits=5, shuffle=True, random_state=42),
    )

    with pytest.raises(ValueError, match="pytest :: Broken KFold"):
        predictor.fit((X, y))


def test_custom_holdout_eval_strategy() -> None:
    X, y = _classification_data(100)
    predictor = Predictor(
        "tabular_classification",
        config=_config(),
        eval_strategy=_broken_split,
    )

    with pytest.raises(ValueError, match="pytest :: Broken split"):
        predictor.fit((X, y))


def test_no_eval_strategy() -> None:
    X, y = _classification_data(100)
    predictor = Predictor(
        "tabular_classification",
        config=_config(),
        eval_strategy=None,
    ).fit((X, y))

    assert set(predictor._performance_metrics) == {"train"}
    assert predictor.predict(X).shape == (len(X),)
    assert predictor.save()
