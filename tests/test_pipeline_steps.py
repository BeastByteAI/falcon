from typing import Any, cast

import numpy as np
import pytest
from numpy import typing as npt

import falcon.abstract as abstract
from falcon.abstract import Pipeline, PipelineStep
from falcon.config import RunConfig
from falcon.serialization import SerializedModelRepr
from falcon.tabular.processors.label_decoder import LabelDecoder
from falcon.tabular.processors.scaler_and_encoder import ScalerAndEncoder
from falcon.tabular.training import CandidateLearner
from falcon.types import ColumnTypes, DatasetSchema


class _RecordingStep:
    def __init__(self, input_type: object, output_type: object, offset: float) -> None:
        self.input_type = input_type
        self.output_type = output_type
        self.offset = offset
        self.fit_input: npt.NDArray[Any] | None = None
        self.fit_schema: DatasetSchema | None = None
        self.fit_groups: npt.NDArray[Any] | None = None

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None:
        self.fit_input = X.copy()
        self.fit_schema = schema
        self.fit_groups = None if groups is None else np.asarray(groups).copy()

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return X + self.offset

    def serialize(self) -> SerializedModelRepr:
        raise NotImplementedError

    def get_input_type(self) -> object:
        return self.input_type

    def get_output_type(self) -> object:
        return self.output_type


class _IncompleteStep:
    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return X


def _schema() -> DatasetSchema:
    return DatasetSchema(
        column_names=("value",),
        column_types=(ColumnTypes.NUMERIC_REGULAR,),
        target_name="target",
        target_kind="regression",
        dimensions=(3, 1),
    )


def test_pipeline_fits_steps_with_schema_and_composes_transforms() -> None:
    schema = _schema()
    X = np.asarray([[1.0], [2.0], [3.0]])
    y = np.asarray([1.0, 2.0, 3.0])
    groups = np.asarray([0, 0, 1])
    first = _RecordingStep("raw", "encoded", 2.0)
    second = _RecordingStep("encoded", "prediction", 3.0)
    pipeline = Pipeline("tabular_regression", X.shape, schema)
    pipeline.add_step(first)
    pipeline.add_step(second)

    pipeline.fit(X, y, schema, groups=groups)

    assert first.fit_schema is schema
    assert second.fit_schema is schema
    assert first.fit_input is not None
    assert second.fit_input is not None
    assert first.fit_groups is not None
    assert second.fit_groups is not None
    assert np.array_equal(first.fit_input, X)
    assert np.array_equal(second.fit_input, X + 2.0)
    assert np.array_equal(first.fit_groups, groups)
    assert np.array_equal(second.fit_groups, groups)
    assert np.array_equal(pipeline.predict(X), X + 5.0)


def test_pipeline_rejects_objects_that_do_not_implement_step_protocol() -> None:
    pipeline = Pipeline("tabular_regression", (3, 1), _schema())

    with pytest.raises(TypeError, match="Pipeline steps must implement"):
        pipeline.add_step(cast(Any, _IncompleteStep()))


@pytest.mark.parametrize(
    "step",
    [
        ScalerAndEncoder(),
        CandidateLearner("tabular_regression", (3, 1), RunConfig()),
        LabelDecoder(),
    ],
)
def test_existing_steps_use_the_single_protocol_without_legacy_aliases(
    step: PipelineStep,
) -> None:
    assert isinstance(step, PipelineStep)
    assert not hasattr(step, "fit_pipe")
    assert not hasattr(step, "forward")
    assert not hasattr(step, "to_onnx")
    assert not hasattr(abstract, "Model")
    assert not hasattr(abstract, "Learner")
    assert not hasattr(abstract, "Processor")
