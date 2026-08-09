from __future__ import annotations

import numpy as np
import pytest

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.tabular.candidates import EstimatorSpec
from falcon.tabular.portfolio_ordering import (
    DatasetMetaFeatures,
    extract_dataset_meta_features,
    reorder_portfolio,
)
from falcon.types import ColumnTypes, DatasetSchema, TargetKind


def _schema(
    column_types: tuple[ColumnTypes, ...],
    *,
    n_rows: int,
    target_kind: TargetKind = "classification",
) -> DatasetSchema:
    return DatasetSchema(
        column_names=tuple(f"feature_{index}" for index in range(len(column_types))),
        column_types=column_types,
        target_name="target",
        target_kind=target_kind,
        dimensions=(n_rows, len(column_types)),
    )


def _known_specs() -> tuple[EstimatorSpec, ...]:
    names = (
        "xgboost_default",
        "catboost_default",
        "catboost_zeroshot_r177",
        "lightgbm_zeroshot_large",
        "lightgbm_default",
        "extra_trees_zeroshot",
        "random_forest_zeroshot",
        "linear_default",
    )
    return tuple(
        EstimatorSpec(name, "linear", {"alpha": float(index + 1)})
        for index, name in enumerate(names)
    )


def test_extract_dataset_meta_features_uses_schema_and_class_distribution() -> None:
    schema = _schema(
        (
            ColumnTypes.NUMERIC_REGULAR,
            ColumnTypes.CAT_LOW_CARD,
            ColumnTypes.CAT_HIGH_CARD,
            ColumnTypes.TEXT_UTF8,
            ColumnTypes.DATE_YMD_ISO8601,
            ColumnTypes.NUMERIC_REGULAR,
        ),
        n_rows=8,
    )

    meta_features = extract_dataset_meta_features(
        np.asarray([0, 0, 0, 0, 0, 0, 1, 1]),
        schema,
    )

    assert meta_features == DatasetMetaFeatures(
        n_rows=8,
        n_features=6,
        class_balance=0.25,
        categorical_fraction=2 / 6,
        text_fraction=1 / 6,
    )

    regression_schema = _schema(
        (ColumnTypes.NUMERIC_REGULAR,),
        n_rows=8,
        target_kind="regression",
    )
    regression = extract_dataset_meta_features(np.arange(8), regression_schema)
    assert regression.class_balance is None


def test_reordering_varies_by_dataset_and_is_seed_deterministic() -> None:
    specs = _known_specs()
    small_wide = DatasetMetaFeatures(1_100, 125, None, 0.0, 0.0)
    large_narrow = DatasetMetaFeatures(35_000, 18, None, 0.15, 0.0)

    small_order = reorder_portfolio(
        specs,
        small_wide,
        TABULAR_REGRESSION_TASK,
        random_state=17,
    )
    repeated_small_order = reorder_portfolio(
        specs,
        small_wide,
        TABULAR_REGRESSION_TASK,
        random_state=17,
    )
    large_order = reorder_portfolio(
        specs,
        large_narrow,
        TABULAR_REGRESSION_TASK,
        random_state=17,
    )

    assert small_order == repeated_small_order
    assert small_order != large_order
    assert small_order[0].name == "catboost_default"
    assert large_order[0].name == "catboost_zeroshot_r177"


def test_reordering_keeps_unknown_slots_and_falls_back_outside_corpus() -> None:
    known = _known_specs()
    unknown = EstimatorSpec("extension_candidate", "linear")
    specs = (known[0], unknown, *known[1:])
    in_range = DatasetMetaFeatures(1_100, 125, None, 0.0, 0.0)

    reordered = reorder_portfolio(
        specs,
        in_range,
        TABULAR_REGRESSION_TASK,
        random_state=42,
    )

    assert reordered[1] is unknown
    out_of_range = DatasetMetaFeatures(50, 125, None, 0.0, 0.0)
    assert (
        reorder_portfolio(
            specs,
            out_of_range,
            TABULAR_REGRESSION_TASK,
            random_state=42,
        )
        == specs
    )


def test_predictor_orders_before_portfolio_limit_and_respects_toggle() -> None:
    specs = _known_specs()

    def selected_candidate(
        n_rows: int,
        n_features: int,
        *,
        enabled: bool,
    ) -> str:
        values = np.arange(n_rows * n_features, dtype=np.float64).reshape(
            n_rows, n_features
        )
        target = values[:, 0] * 0.5 - values[:, -1]
        config = RunConfig(
            candidate_sources=(PortfolioSource(specs=specs, max_candidates=1),),
            dataset_aware_ordering=enabled,
            ensemble_enabled=False,
            eval_strategy=None,
            random_state=23,
        )
        predictor = Predictor(
            TABULAR_REGRESSION_TASK,
            config=config,
        ).fit((values, target))
        return str(predictor.leaderboard().iloc[0]["candidate"])

    small_wide = selected_candidate(300, 120, enabled=True)
    large_narrow = selected_candidate(30_000, 12, enabled=True)

    assert small_wide == "catboost_default"
    assert large_narrow == "catboost_zeroshot_r177"
    assert selected_candidate(300, 120, enabled=True) == small_wide
    assert selected_candidate(50, 120, enabled=True) == specs[0].name
    assert selected_candidate(300, 120, enabled=False) == specs[0].name


def test_classification_profile_requires_class_balance() -> None:
    specs = _known_specs()
    incomplete = DatasetMetaFeatures(1_000, 100, None, 0.0, 0.0)

    with pytest.raises(ValueError, match="class_balance"):
        reorder_portfolio(
            specs,
            incomplete,
            TABULAR_CLASSIFICATION_TASK,
            random_state=42,
        )
