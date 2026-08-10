from __future__ import annotations

import json
import logging
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.tabular.candidates import EstimatorSpec
from falcon.tabular.ingestion import ingest_data
from falcon.type_guessing import determine_column_types
from falcon.types import ColumnTypes
from tests.fnnx_conformance import extract_fnnx_graph


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": np.arange(12, dtype=np.float64),
            "category": [f"group-{index % 3}" for index in range(12)],
            "outcome": [index % 2 for index in range(12)],
        }
    )


def _assert_named_schema(data: pd.DataFrame | str) -> None:
    X, y, schema = ingest_data(data, task="tabular_classification")

    assert X.shape == (12, 2)
    assert y.shape == (12,)
    assert schema.column_names == ("amount", "category")
    assert schema.column_types == (
        ColumnTypes.NUMERIC_REGULAR,
        ColumnTypes.CAT_LOW_CARD,
    )
    assert schema.target_name == "outcome"
    assert schema.target_kind == "classification"
    assert schema.dimensions == (12, 2)


def test_ingestion_produces_schema_for_dataframe_and_file_inputs(
    tmp_path: Path,
) -> None:
    frame = _training_frame()
    csv_path = tmp_path / "training.csv"
    parquet_path = tmp_path / "training.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    _assert_named_schema(frame)
    _assert_named_schema(str(csv_path))
    _assert_named_schema(str(parquet_path))


def test_ingestion_produces_schema_for_array_and_tuple_inputs() -> None:
    frame = _training_frame()
    combined = frame.to_numpy()
    dataframe_tuple = (frame[["amount", "category"]], frame["outcome"])
    array_tuple = (combined[:, :2], combined[:, 2])

    X, y, array_schema = ingest_data(combined, task="tabular_classification")
    dataframe_X, dataframe_y, dataframe_schema = ingest_data(
        dataframe_tuple, task="tabular_classification"
    )
    tuple_X, tuple_y, tuple_schema = ingest_data(
        array_tuple, task="tabular_classification"
    )

    assert X.shape == dataframe_X.shape == tuple_X.shape == (12, 2)
    assert y.shape == dataframe_y.shape == tuple_y.shape == (12,)
    assert array_schema.column_names == ("feature_0", "feature_1")
    assert tuple_schema.column_names == ("feature_0", "feature_1")
    assert dataframe_schema.column_names == ("amount", "category")
    assert dataframe_schema.target_name == "outcome"
    assert array_schema.column_types == dataframe_schema.column_types
    assert tuple_schema.column_types == dataframe_schema.column_types


def test_type_guessing_ignores_missing_values() -> None:
    numeric = np.asarray([[*range(11), np.nan]], dtype=np.float64).T
    dates = np.asarray([["2024-01-01"], [None], ["2024-03-15"]], dtype=np.object_)

    assert determine_column_types(numeric) == [ColumnTypes.NUMERIC_REGULAR]
    assert determine_column_types(dates) == [ColumnTypes.DATE_YMD_ISO8601]


def test_ingestion_drops_missing_targets_and_keeps_missing_feature_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pd.DataFrame(
        {
            "numeric": [*range(12), np.nan, 13],
            "date": ["2024-01-01"] * 14,
            "target": [*[index % 2 for index in range(13)], np.nan],
        }
    )

    with caplog.at_level(logging.INFO, logger="falcon"):
        X, y, schema = ingest_data(frame, task="tabular_classification")

    assert X.shape == (13, 2)
    assert y.shape == (13,)
    assert schema.dimensions == (13, 2)
    assert schema.column_types == (
        ColumnTypes.NUMERIC_REGULAR,
        ColumnTypes.DATE_YMD_ISO8601,
    )
    assert pd.isna(X[:, 0]).sum() == 1
    assert "Dropped 1 row with a missing target" in caplog.text


def test_saved_artifact_embeds_schema_and_uses_sanitized_unique_input_names() -> None:
    sample_indices = np.arange(24, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "feature-a": sample_indices,
            "feature a": sample_indices**2,
            "target value": sample_indices * 0.5,
        }
    )
    config = RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(
                    EstimatorSpec(
                        "schema-test",
                        "hist_gradient_boosting",
                        {"max_iter": 8, "min_samples_leaf": 2},
                    ),
                )
            ),
        ),
        ensemble_enabled=False,
        eval_strategy=None,
    )
    predictor = Predictor("tabular_regression", config=config).fit(frame)

    bundle = predictor.save()
    graph = extract_fnnx_graph(bundle)
    with tarfile.open(fileobj=BytesIO(bundle), mode="r:") as archive:
        manifest_member = archive.extractfile("manifest.json")
        assert manifest_member is not None
        manifest = json.load(manifest_member)

    assert [input_.name for input_ in graph.model.graph.input] == [
        "feature_a",
        "feature_a_1",
    ]
    assert manifest["schema"] == {
        "columns": [
            {"name": "feature-a", "type": "NUMERIC_REGULAR"},
            {"name": "feature a", "type": "NUMERIC_REGULAR"},
        ],
        "target": {"name": "target value", "kind": "regression"},
        "dimensions": {"rows": 24, "features": 2},
    }
