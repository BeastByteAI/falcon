from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy import typing as npt
from onnx import checker
from onnxruntime import InferenceSession

from falcon.abstract import Pipeline
from falcon.serialization import SerializedModelRepr
from falcon.tabular.processors.multi_modal_encoder import MultiModalEncoder
from falcon.tabular.splitting import cross_validation_indices
from falcon.types import ColumnTypes, DatasetSchema, Float32Array, TargetKind


class _TrainingInputRecorder:
    fitted_X: npt.NDArray[Any] | None

    def __init__(self) -> None:
        self.fitted_X = None

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None:
        self.fitted_X = np.asarray(X).copy()

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.asarray(X)

    def serialize(self) -> SerializedModelRepr:
        raise AssertionError("Serialization is not used by this test step")

    def get_input_type(self) -> object:
        return Float32Array

    def get_output_type(self) -> object:
        return Float32Array


def _schema(shape: tuple[int, int], target_kind: TargetKind) -> DatasetSchema:
    return DatasetSchema(
        column_names=("category",),
        column_types=(ColumnTypes.CAT_HIGH_CARD,),
        target_name="target",
        target_kind=target_kind,
        dimensions=shape,
    )


@pytest.mark.parametrize(
    ("target_kind", "target"),
    [
        pytest.param(
            "regression",
            np.asarray([0.2, 0.5, 0.8, 2.1, 2.4, 2.7, 5.0, 5.3, 5.6]),
            id="regression",
        ),
        pytest.param(
            "classification",
            np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0]),
            id="binary",
        ),
        pytest.param(
            "classification",
            np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2]),
            id="multiclass",
        ),
    ],
)
def test_target_encoder_unseen_category_has_standard_onnx_parity(
    target_kind: TargetKind,
    target: npt.NDArray[Any],
) -> None:
    training = np.repeat(
        np.asarray([["alpha"], ["beta"], ["gamma"]], dtype=np.object_), 3, axis=0
    )
    unseen = np.asarray([["unseen"]], dtype=np.object_)
    encoder = MultiModalEncoder()
    encoder.fit(training, target, _schema(training.shape, target_kind))

    expected = encoder.transform(unseen)
    fitted_pipeline = encoder.ct.transformers_[0][1]
    target_encoder = fitted_pipeline.named_steps["target_encoder"]
    model = encoder.serialize().get_model()
    checker.check_model(model)
    actual = InferenceSession(model.SerializeToString()).run(
        None,
        {model.graph.input[0].name: unseen.astype(str)},
    )[0]

    np.testing.assert_allclose(
        expected[0], np.asarray(target_encoder.target_mean_).reshape(-1), atol=1e-6
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert all(
        node.domain in {"", "ai.onnx", "ai.onnx.ml"} for node in model.graph.node
    )


@pytest.mark.parametrize("use_explicit_groups", [False, True])
def test_pipeline_training_values_are_cross_fitted_without_group_leakage(
    use_explicit_groups: bool,
) -> None:
    group_ids = np.repeat(np.arange(6), 2)
    groups = group_ids if use_explicit_groups else None
    training = np.asarray([[f"group-{group}"] for group in group_ids], dtype=np.object_)
    target = np.repeat(np.asarray([1.0, 7.0, 15.0, 31.0, 63.0, 127.0]), 2)
    schema = _schema(training.shape, "regression")
    encoder = MultiModalEncoder()
    recorder = _TrainingInputRecorder()
    pipeline = Pipeline(
        task="tabular_regression", dataset_size=training.shape, schema=schema
    )
    pipeline.add_step(encoder)
    pipeline.add_step(recorder)

    pipeline.fit(training, target, schema, groups=groups)

    assert recorder.fitted_X is not None
    expected = np.empty(training.shape[0], dtype=np.float32)
    for train_indices, validation_indices in cross_validation_indices(
        training,
        target,
        task="tabular_regression",
        groups=groups,
    ):
        expected[validation_indices] = np.mean(target[train_indices])

    np.testing.assert_allclose(recorder.fitted_X[:, 0], expected, atol=1e-6)
    assert not np.allclose(recorder.fitted_X, encoder.transform(training))
