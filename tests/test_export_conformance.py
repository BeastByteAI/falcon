from __future__ import annotations

import json
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pandas as pd
import pytest
from numpy import typing as npt
from onnx import TensorProto, helper

from falcon import Predictor
from falcon.config import (
    ONNX_IR_VERSION,
    ONNX_OPSET_VERSION,
    PortfolioSource,
    RunConfig,
)
from falcon.constants import DEFAULT_PRODUCER_NAME
from falcon.runtime import Runtime
from falcon.serialization import FNNXSerializer, SerializedModelRepr
from falcon.tabular.candidates import EstimatorSpec
from falcon.types import ColumnTypes
from tests.fnnx_conformance import (
    ExtractedFNNXGraph,
    assert_fnnx_conforms,
    assert_onnx_is_valid,
    assert_standard_node_domains,
    assert_standard_opset_declarations,
    extract_fnnx_graph,
    read_bundle_member,
)

_TASKS = ("tabular_classification", "tabular_regression")
_COLUMN_TYPES = {
    "numeric": ColumnTypes.NUMERIC_REGULAR,
    "categorical_low": ColumnTypes.CAT_LOW_CARD,
    "categorical_high": ColumnTypes.CAT_HIGH_CARD,
    "text": ColumnTypes.TEXT_UTF8,
    "date": ColumnTypes.DATE_YMD_ISO8601,
    "datetime": ColumnTypes.DATETIME_YMDHMS_ISO8601,
}
_ALL_CASES = [
    pytest.param(task, column_name, id=f"{task}-{column_name}")
    for task in _TASKS
    for column_name in _COLUMN_TYPES
]
_NODE_DOMAIN_CASES = _ALL_CASES
_ROUND_TRIP_CASES = _ALL_CASES
_BRANCHING_OPS = frozenset({"Where", "IsNaN", "Equal", "Or", "If", "Loop", "Scan"})


@dataclass(frozen=True)
class RoundTripArtifact:
    graph: ExtractedFNNXGraph
    native_predictions: npt.NDArray[Any]
    runtime_predictions: npt.NDArray[Any]
    native_probabilities: npt.NDArray[Any] | None
    runtime_probabilities: npt.NDArray[Any] | None


ArtifactFactory = Callable[[str, str], RoundTripArtifact]


def _conformance_config(task: str) -> RunConfig:
    parameters: dict[str, object]
    if task == "tabular_classification":
        parameters = {"max_iter": 200}
    else:
        parameters = {"alpha": 1.0}
    return RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(
                    EstimatorSpec(
                        "conformance-test",
                        "linear",
                        parameters,
                    ),
                )
            ),
        ),
        ensemble_enabled=False,
        eval_strategy=None,
    )


def _make_training_frame(task: str, column_name: str) -> pd.DataFrame:
    sample_count = 128
    sample_indices = np.arange(sample_count)
    if column_name == "numeric":
        feature = np.linspace(-4.0, 7.0, sample_count)
    elif column_name == "categorical_low":
        feature = np.asarray([f"group-{index % 4}" for index in sample_indices])
    elif column_name == "categorical_high":
        feature = np.asarray([f"category-{index:03d}" for index in sample_indices])
    elif column_name == "text":
        feature = np.asarray(
            [
                "falcon document sample "
                f"{index} contains several useful words about "
                f"{'alpha' if index % 2 else 'beta'}"
                for index in sample_indices
            ]
        )
    elif column_name == "date":
        feature = (
            pd.Timestamp("2020-01-01") + pd.to_timedelta(sample_indices, unit="D")
        ).strftime("%Y-%m-%d")
    elif column_name == "datetime":
        feature = (
            pd.Timestamp("2020-01-01") + pd.to_timedelta(sample_indices, unit="h")
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        raise ValueError(f"Unknown column case: {column_name}")

    if task == "tabular_classification":
        target = np.where(sample_indices % 2, "positive", "negative")
    elif task == "tabular_regression":
        target = 0.4 * sample_indices + np.sin(sample_indices)
    else:
        raise ValueError(f"Unknown task: {task}")
    return pd.DataFrame({column_name: feature, "target": target})


def _build_artifact(
    task: str, column_name: str, artifact_directory: Path
) -> RoundTripArtifact:
    frame = _make_training_frame(task, column_name)
    predictor = Predictor(task, config=_conformance_config(task)).fit(
        frame,
        features=[column_name],
        target="target",
    )
    assert predictor._training_data is not None
    assert predictor._training_data.schema.column_types == (_COLUMN_TYPES[column_name],)

    inputs = frame[[column_name]].copy()
    if column_name == "categorical_high":
        inputs.iloc[0, 0] = "Z"
    native_predictions = np.asarray(predictor.predict(inputs)).reshape(-1)
    native_probabilities = (
        predictor.predict_proba(inputs) if task == "tabular_classification" else None
    )

    artifact_path = artifact_directory / f"{task}-{column_name}.fnnx"
    bundle = predictor.save(artifact_path)
    runtime = Runtime(str(artifact_path))
    runtime_predictions = np.asarray(runtime.predict(inputs)).reshape(-1)
    runtime_probabilities = (
        np.asarray(runtime.predict_proba(inputs))
        if task == "tabular_classification"
        else None
    )
    return RoundTripArtifact(
        graph=extract_fnnx_graph(bundle),
        native_predictions=native_predictions,
        runtime_predictions=runtime_predictions,
        native_probabilities=native_probabilities,
        runtime_probabilities=runtime_probabilities,
    )


@pytest.fixture(scope="module")
def artifact_factory(tmp_path_factory: pytest.TempPathFactory) -> ArtifactFactory:
    artifact_directory = tmp_path_factory.mktemp("export-conformance")
    cache: dict[tuple[str, str], RoundTripArtifact] = {}

    def get_artifact(task: str, column_name: str) -> RoundTripArtifact:
        key = (task, column_name)
        if key not in cache:
            cache[key] = _build_artifact(task, column_name, artifact_directory)
        return cache[key]

    return get_artifact


def _make_identity_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 1])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"], name="identity")],
        "identity_graph",
        [input_info],
        [output_info],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", ONNX_OPSET_VERSION)],
    )


def test_fnnx_declares_only_opset_domains_used_by_the_graph() -> None:
    component = SerializedModelRepr(
        _make_identity_model(),
        n_inputs=1,
        n_outputs=1,
        initial_types=["FLOAT32"],
        initial_shapes=[[None, 1]],
    )

    bundle = FNNXSerializer([component], task="tabular_regression").serialize()
    graph = extract_fnnx_graph(bundle)

    assert {
        "ai.onnx" if opset.domain in {"", "ai.onnx"} else opset.domain
        for opset in graph.model.opset_import
    } == {"ai.onnx"}
    assert graph.fnnx_opset_domains == ("ai.onnx",)


def test_conformance_scanner_reports_forbidden_node() -> None:
    model = _make_identity_model()
    model.graph.node[0].domain = "com.microsoft"
    model.graph.node[0].name = "forbidden_identity"
    del model.opset_import[:]
    model.opset_import.extend(
        [
            helper.make_opsetid("", ONNX_OPSET_VERSION),
            helper.make_opsetid("com.microsoft", 1),
        ]
    )

    graph = ExtractedFNNXGraph(
        model=model,
        fnnx_opset_domains=("ai.onnx", "com.microsoft"),
        declared_ir_version=ONNX_IR_VERSION,
    )
    with pytest.raises(AssertionError) as error:
        assert_fnnx_conforms(graph)

    message = str(error.value)
    assert "forbidden_identity" in message
    assert "com.microsoft" in message


def test_node_domain_scanner_reports_forbidden_node_in_subgraph() -> None:
    nested_output = helper.make_tensor_value_info(
        "nested_output", TensorProto.FLOAT, [None, 1]
    )
    nested_graph = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["input"],
                ["nested_output"],
                name="nested_forbidden_identity",
                domain="com.microsoft",
            )
        ],
        "nested_graph",
        [],
        [nested_output],
    )
    model = _make_identity_model()
    model.graph.node[0].attribute.extend(
        [helper.make_attribute("nested_graph", nested_graph)]
    )

    with pytest.raises(AssertionError) as error:
        assert_standard_node_domains(model)

    message = str(error.value)
    assert "nested_forbidden_identity" in message
    assert "com.microsoft" in message


def test_declared_opset_scanner_reports_model_and_bundle_sources() -> None:
    model = _make_identity_model()
    model.opset_import.extend([helper.make_opsetid("com.microsoft", 1)])
    graph = ExtractedFNNXGraph(
        model=model,
        fnnx_opset_domains=("ai.onnx", "com.microsoft"),
        declared_ir_version=ONNX_IR_VERSION,
    )

    with pytest.raises(AssertionError) as error:
        assert_standard_opset_declarations(graph)

    message = str(error.value)
    assert "model imports com.microsoft" in message
    assert "FNNX ops.json declares com.microsoft" in message


@pytest.mark.parametrize(("task", "column_name"), _ALL_CASES)
def test_exported_model_passes_onnx_checker(
    artifact_factory: ArtifactFactory, task: str, column_name: str
) -> None:
    artifact = artifact_factory(task, column_name)
    assert_onnx_is_valid(artifact.graph.model)


def _collect_bundle_tags(bundle: bytes) -> tuple[list[str], list[str]]:
    """Returns every producer identity and every tag the bundle declares."""
    with tarfile.open(fileobj=BytesIO(bundle), mode="r:") as archive:
        manifest = json.loads(read_bundle_member(archive, "manifest.json"))
        metadata = json.loads(read_bundle_member(archive, "meta.json"))

    producers = [manifest["producer_name"]]
    tags = list(manifest["producer_tags"])
    for io_ in [*manifest["inputs"], *manifest["outputs"]]:
        tags.extend(io_.get("tags", []))
    for container in metadata:
        producers.append(container["producer"])
        tags.extend(container["producer_tags"])
    return producers, tags


@pytest.mark.parametrize("task", _TASKS)
def test_every_bundle_tag_is_namespaced_under_the_producer(
    task: str, tmp_path: Path
) -> None:
    frame = _make_training_frame(task, "numeric")
    predictor = Predictor(task, config=_conformance_config(task)).fit(
        frame, features=["numeric"], target="target"
    )
    bundle = predictor.save(tmp_path / f"{task}.fnnx")

    producers, tags = _collect_bundle_tags(bundle)

    assert DEFAULT_PRODUCER_NAME == "falcon.fnnx.ai"
    assert producers and set(producers) == {DEFAULT_PRODUCER_NAME}
    assert tags
    assert [
        tag for tag in tags if not tag.startswith(f"{DEFAULT_PRODUCER_NAME}::")
    ] == []


@pytest.mark.parametrize(("task", "column_name"), _ALL_CASES)
def test_exported_model_pins_the_ir_version_runtimes_accept(
    artifact_factory: ArtifactFactory, task: str, column_name: str
) -> None:
    """Guards against inheriting `onnx`'s newest IR version.

    A runtime refuses to load any model whose IR is newer than it knows, and the two
    versions move independently: Python 3.10 resolves onnx 1.22 (IR 13) against
    onnxruntime 1.23 (max IR 11), which rejected every exported model.
    """
    artifact = artifact_factory(task, column_name)

    assert artifact.graph.model.ir_version == ONNX_IR_VERSION
    assert artifact.graph.declared_ir_version == ONNX_IR_VERSION


@pytest.mark.parametrize(("task", "column_name"), _NODE_DOMAIN_CASES)
def test_exported_nodes_use_standard_domains(
    artifact_factory: ArtifactFactory, task: str, column_name: str
) -> None:
    artifact = artifact_factory(task, column_name)
    assert_standard_node_domains(artifact.graph.model)


@pytest.mark.parametrize(("task", "column_name"), _ALL_CASES)
def test_exported_opset_declarations_are_standard(
    artifact_factory: ArtifactFactory, task: str, column_name: str
) -> None:
    artifact = artifact_factory(task, column_name)
    assert_standard_opset_declarations(artifact.graph)


@pytest.mark.parametrize(("task", "column_name"), _ROUND_TRIP_CASES)
def test_fnnx_round_trip_matches_native_predictions(
    artifact_factory: ArtifactFactory, task: str, column_name: str
) -> None:
    artifact = artifact_factory(task, column_name)
    if task == "tabular_classification":
        np.testing.assert_array_equal(
            artifact.runtime_predictions, artifact.native_predictions
        )
        assert artifact.native_probabilities is not None
        assert artifact.runtime_probabilities is not None
        np.testing.assert_allclose(
            artifact.runtime_probabilities,
            artifact.native_probabilities,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            artifact.runtime_probabilities.sum(axis=1), 1.0, atol=1e-6
        )
    else:
        np.testing.assert_allclose(
            artifact.runtime_predictions,
            artifact.native_predictions,
            rtol=1e-5,
            atol=1e-5,
        )


def _make_missing_training_frame(task: str) -> pd.DataFrame:
    sample_indices = np.arange(128)
    features: dict[str, npt.NDArray[Any]] = {
        "numeric": np.linspace(-4.0, 7.0, sample_indices.size),
        "categorical_low": np.asarray(
            [f"group-{index % 4}" for index in sample_indices]
        ),
        "categorical_high": np.asarray(
            [f"category-{index:03d}" for index in sample_indices]
        ),
        "date": np.asarray(
            (
                pd.Timestamp("2020-01-01") + pd.to_timedelta(sample_indices, unit="D")
            ).strftime("%Y-%m-%d")
        ),
        "datetime": np.asarray(
            (
                pd.Timestamp("2020-01-01") + pd.to_timedelta(sample_indices, unit="h")
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
        "text": np.where(
            sample_indices % 2,
            "falcons fly quickly above the quiet mountain valley",
            "hawks circle slowly over the broad forest canopy",
        ),
    }
    for name, values in features.items():
        features[name] = np.asarray(values, dtype=np.object_)
        features[name][[5, 37]] = np.nan

    if task == "tabular_classification":
        target: npt.NDArray[Any] = np.where(sample_indices % 2, "positive", "negative")
    else:
        target = 0.4 * sample_indices + np.sin(sample_indices)
    return pd.DataFrame({**features, "target": target})


@pytest.mark.parametrize("task", _TASKS)
def test_missing_feature_rows_round_trip_without_being_dropped(
    task: str, tmp_path: Path
) -> None:
    frame = _make_missing_training_frame(task)
    feature_names = list(_COLUMN_TYPES)
    predictor = Predictor(task, config=_conformance_config(task)).fit(
        frame,
        features=feature_names,
        target="target",
    )
    assert predictor._training_data is not None
    assert predictor._training_data.X.shape[0] == frame.shape[0]
    assert pd.isna(predictor._training_data.X).any()
    assert predictor._training_data.schema.column_types == tuple(_COLUMN_TYPES.values())

    missing_inputs = frame.loc[[5, 37], feature_names]
    native_predictions = np.asarray(predictor.predict(missing_inputs)).reshape(-1)
    artifact_path = tmp_path / f"missing-{task}.fnnx"
    bundle = predictor.save(artifact_path)
    runtime = Runtime(str(artifact_path))
    runtime_predictions = np.asarray(runtime.predict(missing_inputs)).reshape(-1)

    assert_fnnx_conforms(extract_fnnx_graph(bundle))
    if task == "tabular_classification":
        np.testing.assert_array_equal(runtime_predictions, native_predictions)
        np.testing.assert_allclose(
            runtime.predict_proba(missing_inputs),
            predictor.predict_proba(missing_inputs),
            rtol=1e-5,
            atol=1e-6,
        )
    else:
        np.testing.assert_allclose(
            runtime_predictions, native_predictions, rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("task", _TASKS)
def test_export_without_imputation_is_free_of_data_dependent_branching(
    task: str, tmp_path: Path
) -> None:
    sample_indices = np.arange(128)
    frame = pd.DataFrame(
        {
            "numeric": np.linspace(-4.0, 7.0, sample_indices.size),
            "categorical_low": [f"group-{index % 4}" for index in sample_indices],
            "categorical_high": [f"category-{index:03d}" for index in sample_indices],
            "target": (
                np.where(sample_indices % 2, "positive", "negative")
                if task == "tabular_classification"
                else 0.4 * sample_indices + np.sin(sample_indices)
            ),
        }
    )
    feature_names = ["numeric", "categorical_low", "categorical_high"]
    config = _conformance_config(task).replaced(impute_missing=False)
    predictor = Predictor(task, config=config).fit(
        frame, features=feature_names, target="target"
    )

    artifact_path = tmp_path / f"branch-free-{task}.fnnx"
    bundle = predictor.save(artifact_path)
    graph = extract_fnnx_graph(bundle)
    assert_fnnx_conforms(graph)

    emitted_ops = {node.op_type for node in graph.model.graph.node}
    assert not emitted_ops & _BRANCHING_OPS

    inputs = frame.loc[[0, 63], feature_names]
    runtime = Runtime(str(artifact_path))
    native_predictions = np.asarray(predictor.predict(inputs)).reshape(-1)
    runtime_predictions = np.asarray(runtime.predict(inputs)).reshape(-1)
    if task == "tabular_classification":
        np.testing.assert_array_equal(runtime_predictions, native_predictions)
    else:
        np.testing.assert_allclose(
            runtime_predictions, native_predictions, rtol=1e-5, atol=1e-5
        )
