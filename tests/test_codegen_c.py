from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt
from onnx import TensorProto

from falcon import Predictor
from falcon.codegen import CArtifact, CodegenError, compile_to_c
from falcon.codegen.bundle import read_bundle
from falcon.config import PortfolioSource, RunConfig
from falcon.tabular.candidates import EstimatorSpec
from falcon.tabular.models.gbdt import get_gbdt_model_classes
from falcon.types import ColumnTypes

pytest.importorskip(
    "fnnx.extras.compilers.c",
    reason="generating C requires the FNNX ahead-of-time compiler",
)

_ROWS = 240
_BATCH = 64
_STRICT_FLAGS = ("-std=c99", "-Wall", "-Wextra", "-Werror")
_MISSING_ROWS = 24


@dataclass(frozen=True)
class Fixture:
    artifact: CArtifact
    predictor: Predictor
    frame: pd.DataFrame
    bundle_path: Path


FixtureFactory = Callable[[str, bool], Fixture]


def _training_frame(task: str, *, impute: bool) -> pd.DataFrame:
    generator = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "num_a": generator.normal(size=_ROWS),
            "num_b": generator.uniform(-3.0, 3.0, size=_ROWS),
            "cat_color": generator.choice(["red", "green", "blue"], size=_ROWS),
            "cat_size": generator.choice(["s", "m", "l", "xl"], size=_ROWS),
        }
    )
    signal = (
        1.5 * frame["num_a"]
        - 0.8 * frame["num_b"]
        + np.where(frame["cat_color"] == "red", 1.2, -0.4)
        + np.where(frame["cat_size"] == "xl", 0.9, 0.0)
        + generator.normal(scale=0.4, size=_ROWS)
    )
    if impute:
        frame.loc[generator.choice(_ROWS, _MISSING_ROWS, replace=False), "num_a"] = (
            np.nan
        )
        frame.loc[
            generator.choice(_ROWS, _MISSING_ROWS, replace=False), "cat_color"
        ] = None
    frame["target"] = (
        np.where(signal > float(np.median(signal)), "yes", "no")
        if task == "tabular_classification"
        else signal
    )
    return frame


def _config(impute: bool) -> RunConfig:
    return RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(
                    EstimatorSpec(
                        "trees", "random_forest", {"n_estimators": 8, "max_depth": 4}
                    ),
                )
            ),
        ),
        ensemble_enabled=False,
        eval_strategy=None,
        impute_missing=impute,
        random_state=0,
    )


def _build(task: str, impute: bool, directory: Path) -> Fixture:
    frame = _training_frame(task, impute=impute)
    features = ["num_a", "num_b", "cat_color", "cat_size"]
    predictor = Predictor(task, config=_config(impute)).fit(
        frame, features=features, target="target"
    )
    bundle_path = directory / "model.fnnx"
    predictor.save(bundle_path)
    artifact = compile_to_c(
        bundle_path, directory / "c", prefix="demo", batch_size=_BATCH
    )
    return Fixture(
        artifact=artifact,
        predictor=predictor,
        frame=frame[features],
        bundle_path=bundle_path,
    )


@pytest.fixture(scope="module")
def fixture_factory(tmp_path_factory: pytest.TempPathFactory) -> FixtureFactory:
    root = tmp_path_factory.mktemp("codegen-c")
    cache: dict[tuple[str, bool], Fixture] = {}

    def get(task: str, impute: bool) -> Fixture:
        key = (task, impute)
        if key not in cache:
            directory = root / f"{task}-{'imputed' if impute else 'plain'}"
            directory.mkdir()
            cache[key] = _build(task, impute, directory)
        return cache[key]

    return get


def _encode(mapping: Any, values: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    """Reference implementation of the encoder emitted into the helper header."""
    table = {category: index for index, category in enumerate(mapping.categories)}
    missing = set(mapping.missing_tokens)
    codes = []
    for value in values:
        text = None if pd.isna(value) else str(value)
        if text is None or text in missing:
            codes.append(mapping.missing_code if mapping.imputed else -1)
        else:
            codes.append(table.get(text, -1))
    return np.asarray(codes, dtype=np.int64).reshape(-1, 1)


def _feed(fixture: Fixture, rows: pd.DataFrame) -> dict[str, npt.NDArray[Any]]:
    mappings = {item.name: item for item in fixture.artifact.mapping.categoricals}
    return {
        name: (
            _encode(mappings[name], rows[name].to_numpy())
            if name in mappings
            else rows[name].to_numpy().astype(np.float32).reshape(-1, 1)
        )
        for name in rows.columns
    }


def _run(fixture: Fixture, rows: pd.DataFrame) -> dict[str, npt.NDArray[Any]]:
    from fnnx.extras.compilers.c import load_compiled

    if shutil.which("cc") is None:
        pytest.skip("no C compiler available")
    return load_compiled(fixture.artifact.header_path).run(_feed(fixture, rows))


@pytest.mark.parametrize("impute", [True, False])
def test_regression_artifact_matches_predictor(
    fixture_factory: FixtureFactory, impute: bool
) -> None:
    fixture = fixture_factory("tabular_regression", impute)
    rows = fixture.frame.iloc[:_BATCH]
    predicted = _run(fixture, rows)["y_pred"].reshape(-1)
    expected = np.asarray(fixture.predictor.predict(rows), dtype=np.float32)
    assert np.abs(predicted - expected).max() < 1e-4


@pytest.mark.parametrize("impute", [True, False])
def test_classification_artifact_matches_predictor(
    fixture_factory: FixtureFactory, impute: bool
) -> None:
    fixture = fixture_factory("tabular_classification", impute)
    rows = fixture.frame.iloc[:_BATCH]
    outputs = _run(fixture, rows)
    labels = np.asarray(fixture.artifact.mapping.class_labels, dtype=object)
    predicted = labels[outputs["y_pred"].reshape(-1)]
    assert np.array_equal(predicted, np.asarray(fixture.predictor.predict(rows)))
    expected = fixture.predictor.predict_proba(rows)
    assert np.abs(outputs["probabilities"] - expected).max() < 1e-5


def _postprocessed_config(task: str, **overrides: Any) -> RunConfig:
    settings: dict[str, Any] = {
        "candidate_sources": (
            PortfolioSource(
                specs=(
                    EstimatorSpec(
                        "trees", "random_forest", {"n_estimators": 8, "max_depth": 4}
                    ),
                )
            ),
        ),
        "ensemble_enabled": False,
        "eval_strategy": None,
        "impute_missing": False,
        "random_state": 0,
        "oof_folds": 3,
    }
    settings.update(overrides)
    return RunConfig(**settings)


def _build_with(task: str, config: RunConfig, directory: Path) -> Fixture:
    frame = _training_frame(task, impute=False)
    features = ["num_a", "num_b", "cat_color", "cat_size"]
    predictor = Predictor(task, config=config).fit(
        frame, features=features, target="target"
    )
    bundle_path = directory / "model.fnnx"
    predictor.save(bundle_path)
    artifact = compile_to_c(
        bundle_path, directory / "c", prefix="demo", batch_size=_BATCH
    )
    return Fixture(artifact, predictor, frame[features], bundle_path)


def test_calibrated_and_tuned_classification_artifact_matches_predictor(
    tmp_path: Path,
) -> None:
    fixture = _build_with(
        "tabular_classification",
        _postprocessed_config("tabular_classification", calibrate=True),
        tmp_path,
    )
    bundle = read_bundle(fixture.bundle_path)
    node_names = {node.name for node in bundle.model.graph.node}
    assert any("falcon_temperature" in name for name in node_names)
    assert any("falcon_decision" in name for name in node_names)

    rows = fixture.frame.iloc[:_BATCH]
    outputs = _run(fixture, rows)
    labels = np.asarray(fixture.artifact.mapping.class_labels, dtype=object)
    predicted = labels[outputs["y_pred"].reshape(-1)]

    assert np.array_equal(predicted, np.asarray(fixture.predictor.predict(rows)))
    expected = fixture.predictor.predict_proba(rows)
    assert np.abs(outputs["probabilities"] - expected).max() < 1e-5


def test_conformal_regression_artifact_matches_predictor(tmp_path: Path) -> None:
    fixture = _build_with(
        "tabular_regression",
        _postprocessed_config("tabular_regression", conformal_alpha=0.2),
        tmp_path,
    )
    bundle = read_bundle(fixture.bundle_path)
    assert any("falcon_conformal" in node.name for node in bundle.model.graph.node)

    rows = fixture.frame.iloc[:_BATCH]
    outputs = _run(fixture, rows)
    predicted = outputs["y_pred"].reshape(-1)
    expected = np.asarray(fixture.predictor.predict(rows), dtype=np.float32)
    lower = outputs["y_lower"].reshape(-1)
    upper = outputs["y_upper"].reshape(-1)

    assert np.abs(predicted - expected).max() < 1e-4
    assert np.all(lower <= upper)


def test_artifact_serves_a_partial_batch(fixture_factory: FixtureFactory) -> None:
    fixture = fixture_factory("tabular_regression", False)
    rows = fixture.frame.iloc[: _BATCH // 4]
    predicted = _run(fixture, rows)["y_pred"].reshape(-1)
    expected = np.asarray(fixture.predictor.predict(rows), dtype=np.float32)
    assert predicted.shape == (len(rows),)
    assert np.abs(predicted - expected).max() < 1e-4


def test_generated_graph_holds_no_strings(fixture_factory: FixtureFactory) -> None:
    fixture = fixture_factory("tabular_classification", True)
    for name in ("num_a", "num_b", "cat_color", "cat_size"):
        tensor = next(
            item
            for item in fixture.artifact.report["entrypoint"]["inputs"]
            if item["name"] == name
        )
        assert tensor["dtype"] in {"float32", "int64"}
    outputs = {
        item["name"]: item for item in fixture.artifact.report["entrypoint"]["outputs"]
    }
    assert outputs["y_pred"]["dtype"] == "int64"
    assert outputs["probabilities"]["dtype"] == "float32"


def test_exported_bundle_is_left_alone(fixture_factory: FixtureFactory) -> None:
    fixture = fixture_factory("tabular_classification", False)
    bundle = read_bundle(fixture.bundle_path)
    string_inputs = {
        value_info.name
        for value_info in bundle.model.graph.input
        if value_info.type.tensor_type.elem_type == TensorProto.STRING
    }
    assert string_inputs == {"cat_color", "cat_size"}
    manifest_dtypes = {
        item["name"]: item["dtype"] for item in bundle.manifest["inputs"]
    }
    assert manifest_dtypes["cat_color"] == "Array[string]"


def test_categorical_mapping_reports_the_vocabulary(
    fixture_factory: FixtureFactory,
) -> None:
    fixture = fixture_factory("tabular_regression", False)
    mappings = {item.name: item for item in fixture.artifact.mapping.categoricals}
    assert set(mappings) == {"cat_color", "cat_size"}
    assert mappings["cat_color"].categories == ("blue", "green", "red")
    assert mappings["cat_size"].categories == ("l", "m", "s", "xl")
    assert mappings["cat_color"].column_type == "CAT_LOW_CARD"
    assert not mappings["cat_color"].imputed
    assert mappings["cat_color"].missing_code == -1


def test_imputed_categorical_maps_missing_to_the_sentinel(
    fixture_factory: FixtureFactory,
) -> None:
    fixture = fixture_factory("tabular_regression", True)
    mapping = next(
        item
        for item in fixture.artifact.mapping.categoricals
        if item.name == "cat_color"
    )
    assert mapping.imputed
    assert "nan" in mapping.missing_tokens
    assert mapping.categories[mapping.missing_code] == "__falcon_missing__"


def test_missing_category_predicts_like_the_predictor(
    fixture_factory: FixtureFactory,
) -> None:
    """A missing value reaches the artifact as the sentinel code the pipeline fills with."""
    fixture = fixture_factory("tabular_regression", True)
    rows = fixture.frame.iloc[:_BATCH].copy()
    rows.loc[rows.index[0], "cat_color"] = None
    predicted = _run(fixture, rows)["y_pred"].reshape(-1)
    expected = np.asarray(fixture.predictor.predict(rows), dtype=np.float32)
    assert np.abs(predicted - expected).max() < 1e-4


def test_unknown_category_predicts_like_the_predictor(
    fixture_factory: FixtureFactory,
) -> None:
    fixture = fixture_factory("tabular_regression", False)
    rows = fixture.frame.iloc[:_BATCH].copy()
    rows.loc[rows.index[0], "cat_color"] = "chartreuse"
    assert _feed(fixture, rows)["cat_color"][0, 0] == -1
    predicted = _run(fixture, rows)["y_pred"].reshape(-1)
    expected = np.asarray(fixture.predictor.predict(rows), dtype=np.float32)
    assert np.abs(predicted - expected).max() < 1e-4


def test_helper_header_declares_the_tables(fixture_factory: FixtureFactory) -> None:
    fixture = fixture_factory("tabular_classification", False)
    helper = fixture.artifact.helper_path.read_text()
    assert "int64_t demo_encode_cat_color(const char* value);" in helper
    assert "const char* demo_class_label(int64_t index);" in helper
    assert '"chartreuse"' not in helper
    for category in ("blue", "green", "red", "l", "m", "s", "xl"):
        assert f'"{category}"' in helper
    for label in fixture.artifact.mapping.class_labels:
        assert f'"{label}"' in helper


def test_helper_header_documents_the_signature(fixture_factory: FixtureFactory) -> None:
    fixture = fixture_factory("tabular_classification", True)
    helper = fixture.artifact.helper_path.read_text()
    assert "demo_run() takes its inputs in this order:" in helper
    assert "cat_color (int64_t) -- category code from demo_encode_cat_color()" in helper
    assert "num_a (float) -- numeric feature, as-is" in helper
    assert "__falcon_missing__" in helper


@pytest.mark.parametrize("task", ["tabular_classification", "tabular_regression"])
def test_generated_headers_compile_and_agree_with_the_predictor(
    fixture_factory: FixtureFactory, task: str, tmp_path: Path
) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("no C compiler available")
    fixture = fixture_factory(task, True)
    rows = fixture.frame.iloc[:8]
    source = tmp_path / "main.c"
    source.write_text(_driver_source(task, rows))
    for header in (fixture.artifact.header_path, fixture.artifact.helper_path):
        shutil.copy(header, tmp_path / header.name)

    binary = tmp_path / "driver"
    subprocess.run(
        [compiler, *_STRICT_FLAGS, "-O1", str(source), "-lm", "-o", str(binary)],
        check=True,
        capture_output=True,
        cwd=tmp_path,
    )
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    produced = [line for line in completed.stdout.splitlines() if line]

    if task == "tabular_classification":
        assert produced == list(np.asarray(fixture.predictor.predict(rows)))
    else:
        expected = np.asarray(fixture.predictor.predict(rows), dtype=np.float32)
        assert np.abs(np.asarray(produced, dtype=np.float32) - expected).max() < 1e-4


def _driver_source(task: str, rows: pd.DataFrame) -> str:
    """A C program that feeds `rows` through the generated headers and prints the result."""
    literals = ",\n".join(
        "    {{{num_a}, {num_b}f, {color}, {size}}}".format(
            num_a="NAN" if pd.isna(row.num_a) else f"{row.num_a}f",
            num_b=row.num_b,
            color="NULL" if pd.isna(row.cat_color) else f'"{row.cat_color}"',
            size=f'"{row.cat_size}"',
        )
        for row in rows.itertuples()
    )
    classification = task == "tabular_classification"
    report = (
        'printf("%s\\n", demo_class_label(y_pred[i]));'
        if classification
        else 'printf("%.7g\\n", (double)y_pred[i]);'
    )
    output_type = "int64_t" if classification else "float"
    declarations = (
        "    float probabilities[DEMO_OUTPUT_PROBABILITIES_COUNT];\n"
        if classification
        else ""
    )
    arguments = (
        "cat_size, probabilities, y_pred" if classification else "cat_size, y_pred"
    )
    return f"""#include <math.h>
#include <stdio.h>

#define DEMO_IMPLEMENTATION
#include "demo.h"

#define DEMO_FALCON_IMPLEMENTATION
#include "demo_falcon.h"

struct row {{ float num_a; float num_b; const char* color; const char* size; }};

int main(void)
{{
    static const struct row rows[] = {{
{literals}
    }};
    const int32_t batch = (int32_t)(sizeof rows / sizeof rows[0]);
    float num_a[DEMO_INPUT_NUM_A_COUNT];
    float num_b[DEMO_INPUT_NUM_B_COUNT];
    int64_t cat_color[DEMO_INPUT_CAT_COLOR_COUNT];
    int64_t cat_size[DEMO_INPUT_CAT_SIZE_COUNT];
    {output_type} y_pred[DEMO_OUTPUT_Y_PRED_COUNT];
{declarations}
    for (int32_t i = 0; i < batch; i++) {{
        num_a[i] = rows[i].num_a;
        num_b[i] = rows[i].num_b;
        cat_color[i] = demo_encode_cat_color(rows[i].color);
        cat_size[i] = demo_encode_cat_size(rows[i].size);
    }}
    if (demo_run(batch, num_a, num_b, cat_color, {arguments}) != DEMO_OK) {{
        return 1;
    }}
    for (int32_t i = 0; i < batch; i++) {{
        {report}
    }}
    return 0;
}}
"""


@pytest.mark.parametrize("column", ["text", "date"])
def test_text_and_date_features_are_rejected(column: str, tmp_path: Path) -> None:
    indices = np.arange(_ROWS)
    if column == "text":
        feature = np.asarray(
            [
                f"falcon document {index} contains several useful words about "
                f"{'alpha' if index % 2 else 'beta'} and its many properties"
                for index in indices
            ]
        )
        expected_type = ColumnTypes.TEXT_UTF8
    else:
        feature = (
            pd.Timestamp("2020-01-01") + pd.to_timedelta(indices, unit="D")
        ).strftime("%Y-%m-%d")
        expected_type = ColumnTypes.DATE_YMD_ISO8601
    frame = pd.DataFrame(
        {"num": indices * 0.5, column: feature, "target": 0.3 * indices}
    )
    predictor = Predictor("tabular_regression", config=_config(True)).fit(
        frame, features=["num", column], target="target"
    )
    assert predictor._training_data is not None
    assert predictor._training_data.schema.column_types[1] == expected_type
    bundle_path = tmp_path / "model.fnnx"
    predictor.save(bundle_path)

    with pytest.raises(CodegenError) as error:
        compile_to_c(bundle_path, tmp_path / "c")
    assert column in str(error.value)
    assert "Text and date features" in str(error.value)


def test_missing_model_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CodegenError, match="Model not found"):
        compile_to_c(tmp_path / "absent.fnnx", tmp_path / "c")


def test_batch_size_must_be_positive(
    fixture_factory: FixtureFactory, tmp_path: Path
) -> None:
    fixture = fixture_factory("tabular_regression", False)
    with pytest.raises(CodegenError, match="batch_size"):
        compile_to_c(fixture.bundle_path, tmp_path / "c", batch_size=0)


def test_prefix_defaults_to_the_task(
    fixture_factory: FixtureFactory, tmp_path: Path
) -> None:
    fixture = fixture_factory("tabular_regression", False)
    artifact = compile_to_c(fixture.bundle_path, tmp_path / "c", batch_size=8)
    assert artifact.prefix == "tabular_regression"
    assert artifact.entrypoint == "tabular_regression_run"
    assert artifact.batch_size == 8
    assert artifact.header_path.name == "tabular_regression.h"
    assert artifact.helper_path.name == "tabular_regression_falcon.h"


def test_gbdt_ensemble_compiles(tmp_path: Path) -> None:
    families = get_gbdt_model_classes("tabular_regression")
    missing = {"lightgbm", "xgboost"} - set(families)
    if missing:
        pytest.skip(f"{', '.join(sorted(missing))} is not installed")

    frame = _training_frame("tabular_regression", impute=False)
    config = RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(
                    EstimatorSpec(
                        "rf", "random_forest", {"n_estimators": 8, "max_depth": 4}
                    ),
                    EstimatorSpec(
                        "xgb", "xgboost", {"n_estimators": 8, "max_depth": 3}
                    ),
                    EstimatorSpec(
                        "lgbm", "lightgbm", {"n_estimators": 8, "num_leaves": 7}
                    ),
                )
            ),
        ),
        plateau_enabled=False,
        ensemble_max_iterations=6,
        impute_missing=False,
        oof_folds=3,
        eval_strategy=None,
        random_state=0,
    )
    features = ["num_a", "num_b", "cat_color", "cat_size"]
    predictor = Predictor("tabular_regression", config=config).fit(
        frame, features=features, target="target"
    )
    bundle_path = tmp_path / "model.fnnx"
    predictor.save(bundle_path)
    artifact = compile_to_c(
        bundle_path, tmp_path / "c", prefix="ens", batch_size=_BATCH
    )
    assert artifact.report["memory"]["static_bytes"] > 0

    rows = frame[features].iloc[:_BATCH]
    fixture = Fixture(artifact, predictor, rows, bundle_path)
    predicted = _run(fixture, rows)["y_pred"].reshape(-1)
    expected = np.asarray(predictor.predict(rows), dtype=np.float32)
    assert np.abs(predicted - expected).max() < 1e-4
