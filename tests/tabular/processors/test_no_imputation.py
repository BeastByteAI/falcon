from typing import Any

import numpy as np
import pytest
from numpy import typing as npt
from onnx import ModelProto, checker
from onnxruntime import InferenceSession

from falcon.tabular.pipelines.simple_tabular_pipeline import SimpleTabularPipeline
from falcon.tabular.processors.multi_modal_encoder import MultiModalEncoder
from falcon.types import ColumnTypes, DatasetSchema

BRANCHING_OPS = frozenset({"Where", "IsNaN", "Equal", "Or", "If", "Loop", "Scan"})


def _schema(
    column_names: tuple[str, ...],
    column_types: tuple[ColumnTypes, ...],
    dimensions: tuple[int, int],
) -> DatasetSchema:
    return DatasetSchema(
        column_names=column_names,
        column_types=column_types,
        target_name="target",
        target_kind="regression",
        dimensions=dimensions,
    )


def _runtime_inputs(
    model: ModelProto,
    values: npt.NDArray[np.object_],
    column_types: tuple[ColumnTypes, ...],
) -> dict[str, npt.NDArray[Any]]:
    inputs: dict[str, npt.NDArray[Any]] = {}
    for index, model_input in enumerate(model.graph.input):
        column = values[:, index].reshape(-1, 1)
        if column_types[index] == ColumnTypes.NUMERIC_REGULAR:
            inputs[model_input.name] = column.astype(np.float32)
        else:
            inputs[model_input.name] = column.astype(str)
    return inputs


def _emitted_ops(model: ModelProto) -> set[str]:
    return {node.op_type for node in model.graph.node}


def test_encoder_without_imputation_exports_a_branch_free_graph() -> None:
    column_types = (
        ColumnTypes.NUMERIC_REGULAR,
        ColumnTypes.CAT_LOW_CARD,
        ColumnTypes.CAT_HIGH_CARD,
    )
    training = np.asarray(
        [[float(index), f"g{index % 3}", f"c{index}"] for index in range(12)],
        dtype=np.object_,
    )
    inference = np.asarray([[2.0, "g1", "c3"], [7.5, "unseen", "unseen"]], np.object_)
    schema = _schema(("numeric", "low_card", "high_card"), column_types, training.shape)
    encoder = MultiModalEncoder(impute_missing=False)
    encoder.fit(training, np.arange(training.shape[0], dtype=np.float64), schema)

    expected = encoder.transform(inference)
    model = encoder.serialize().get_model()
    checker.check_model(model)
    actual = InferenceSession(model.SerializeToString()).run(
        None, _runtime_inputs(model, inference, column_types)
    )[0]

    assert not BRANCHING_OPS & _emitted_ops(model)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_numeric_features_drop_the_missing_indicator_without_imputation() -> None:
    training = np.asarray([[1.0], [3.0], [9.0]], dtype=np.object_)
    schema = _schema(("numeric",), (ColumnTypes.NUMERIC_REGULAR,), training.shape)

    imputing = MultiModalEncoder()
    imputing.fit(training, np.arange(3, dtype=np.float64), schema)
    plain = MultiModalEncoder(impute_missing=False)
    plain.fit(training, np.arange(3, dtype=np.float64), schema)

    assert imputing.transform(training).shape == (3, 2)
    assert plain.transform(training).shape == (3, 1)


def test_missing_numeric_values_are_rejected_without_imputation() -> None:
    training = np.asarray([[1.0], [np.nan], [9.0]], dtype=np.object_)
    schema = _schema(("numeric",), (ColumnTypes.NUMERIC_REGULAR,), training.shape)
    encoder = MultiModalEncoder(impute_missing=False)

    with pytest.raises(ValueError, match="imputation is disabled"):
        encoder.fit(training, np.arange(3, dtype=np.float64), schema)


def test_missing_categories_become_regular_categories_without_imputation() -> None:
    column_types = (ColumnTypes.CAT_LOW_CARD,)
    training = np.asarray([["red"], [np.nan], ["blue"], ["red"]], dtype=np.object_)
    inference = np.asarray([[np.nan], ["red"]], dtype=np.object_)
    schema = _schema(("category",), column_types, training.shape)
    encoder = MultiModalEncoder(impute_missing=False)
    encoder.fit(training, np.arange(4, dtype=np.float64), schema)

    expected = encoder.transform(inference)
    model = encoder.serialize().get_model()
    actual = InferenceSession(model.SerializeToString()).run(
        None, _runtime_inputs(model, inference, column_types)
    )[0]

    assert list(encoder.ct.transformers_[0][1].named_steps["ohe"].categories_[0]) == [
        "blue",
        "nan",
        "red",
    ]
    assert not BRANCHING_OPS & _emitted_ops(model)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_date_features_are_rejected_without_imputation() -> None:
    training = np.asarray([["2022-02-02"], ["2022-02-25"]], dtype=np.object_)
    schema = _schema(("date",), (ColumnTypes.DATE_YMD_ISO8601,), training.shape)
    encoder = MultiModalEncoder(impute_missing=False)

    with pytest.raises(ValueError, match="not supported while imputation is disabled"):
        encoder.fit(training, np.zeros(2), schema)


@pytest.mark.parametrize("impute_missing", [True, False])
def test_pipeline_forwards_the_imputation_setting(impute_missing: bool) -> None:
    training = np.asarray([[float(index)] for index in range(12)], dtype=np.object_)
    schema = _schema(("numeric",), (ColumnTypes.NUMERIC_REGULAR,), training.shape)
    pipeline = SimpleTabularPipeline(
        task="tabular_regression",
        dataset_size=training.shape,
        learner=_RecordingLearner,
        schema=schema,
        impute_missing=impute_missing,
    )
    pipeline.fit(training, np.arange(12, dtype=np.float64), schema)

    encoder = pipeline.steps[0]
    assert isinstance(encoder, MultiModalEncoder)
    assert encoder.impute_missing is impute_missing


class _RecordingLearner:
    def __init__(self, task: str, dataset_size: tuple[int, ...]) -> None:
        self.task = task
        self.dataset_size = dataset_size

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None:
        self.n_features = X.shape[1]

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return X

    def serialize(self) -> Any:
        raise NotImplementedError

    def get_input_type(self) -> object:
        return MultiModalEncoder().get_output_type()

    def get_output_type(self) -> object:
        return MultiModalEncoder().get_output_type()
