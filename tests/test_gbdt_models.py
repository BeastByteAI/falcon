from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from falcon import Predictor
from falcon.config import PortfolioSource, RunConfig
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.runtime import Runtime
from falcon.serialization import serialize_to_onnx
from falcon.tabular.candidates import EstimatorSpec
from falcon.tabular.models.gbdt import get_gbdt_model_classes
from tests.fnnx_conformance import assert_standard_node_domains

_FAMILIES = ("lightgbm", "xgboost", "catboost")


def _model_parameters(family: str) -> dict[str, Any]:
    if family == "catboost":
        return {"iterations": 8, "depth": 3}
    return {"n_estimators": 8, "max_depth": 3}


def _model_class(family: str, task: str) -> type[Any]:
    classes = get_gbdt_model_classes(task, n_classes=2)
    if family not in classes:
        pytest.skip(f"{family} is not installed")
    return classes[family]


@pytest.mark.parametrize("family", _FAMILIES)
def test_gbdt_binary_classifier_onnx_parity(family: str) -> None:
    X, y = make_classification(
        n_samples=96,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=12,
    )
    X = X.astype(np.float32)
    model = _model_class(family, TABULAR_CLASSIFICATION_TASK)(
        random_state=17,
        **_model_parameters(family),
    )
    model.fit(X, y)

    graph = serialize_to_onnx(
        [model.serialize()],
        task=TABULAR_CLASSIFICATION_TASK,
    )
    onnx.checker.check_model(graph)
    assert_standard_node_domains(graph)
    assert any(node.op_type == "TreeEnsembleClassifier" for node in graph.graph.node)
    assert not graph.graph.output[0].type.tensor_type.shape.dim[0].HasField("dim_value")

    session = ort.InferenceSession(
        graph.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    labels, probabilities = session.run(
        None,
        {session.get_inputs()[0].name: X},
    )

    assert np.array_equal(np.asarray(labels).reshape(-1), model.predict(X))
    np.testing.assert_allclose(
        probabilities,
        model.predict_proba(X),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("family", _FAMILIES)
def test_gbdt_regressor_onnx_parity_and_output_shape(family: str) -> None:
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        random_state=12,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    model = _model_class(family, TABULAR_REGRESSION_TASK)(
        random_state=17,
        **_model_parameters(family),
    )
    model.fit(X, y)

    graph = serialize_to_onnx([model.serialize()], task=TABULAR_REGRESSION_TASK)
    onnx.checker.check_model(graph)
    assert_standard_node_domains(graph)
    assert any(node.op_type == "TreeEnsembleRegressor" for node in graph.graph.node)

    session = ort.InferenceSession(
        graph.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (predictions,) = session.run(
        None,
        {session.get_inputs()[0].name: X},
    )

    assert np.asarray(predictions).shape == (len(X),)
    np.testing.assert_allclose(
        predictions,
        model.predict(X),
        rtol=1e-5,
        atol=1e-4,
    )


def test_xgboost_regressor_fnnx_parity_after_numeric_preprocessing(
    tmp_path: Path,
) -> None:
    X, y = make_regression(
        n_samples=64,
        n_features=4,
        n_informative=3,
        random_state=4,
    )
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["target"] = y
    _model_class("xgboost", TABULAR_REGRESSION_TASK)
    config = RunConfig(
        candidate_sources=(
            PortfolioSource(
                specs=(
                    EstimatorSpec(
                        "xgboost-test",
                        "xgboost",
                        {"n_estimators": 4, "max_depth": 2},
                    ),
                )
            ),
        ),
        ensemble_enabled=False,
        eval_strategy=None,
    )
    predictor = Predictor(TABULAR_REGRESSION_TASK, config=config).fit(frame)
    inputs = frame.drop(columns="target")
    expected = predictor.predict(inputs)
    artifact_path = tmp_path / "xgboost-regression.fnnx"

    predictor.save(artifact_path)
    actual = Runtime(str(artifact_path)).predict(inputs)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


def test_catboost_is_excluded_from_multiclass_with_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    if "catboost" not in get_gbdt_model_classes(
        TABULAR_CLASSIFICATION_TASK, n_classes=2
    ):
        pytest.skip("catboost is not installed")

    with caplog.at_level("INFO", logger="falcon"):
        classes = get_gbdt_model_classes(
            TABULAR_CLASSIFICATION_TASK,
            n_classes=3,
        )

    assert "catboost" not in classes
    assert "CatBoost" in caplog.text
    assert "multiclass" in caplog.text


def test_each_optional_family_registers_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from falcon.tabular.models import gbdt

    available_modules = {"catboost"}
    monkeypatch.setattr(
        gbdt,
        "_module_is_available",
        lambda module_name: module_name in available_modules,
    )

    registry = gbdt._discover_gbdt_families()

    assert set(registry) == {"catboost"}


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        get_gbdt_model_classes("forecasting")
