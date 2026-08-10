from typing import Any

import numpy as np
from numpy import typing as npt
from onnx import checker
from onnxruntime import InferenceSession

from falcon.tabular.processors.multi_modal_encoder import MultiModalEncoder
from falcon.types import ColumnTypes, DatasetSchema


def _runtime_inputs(
    model_inputs: list[Any], values: npt.NDArray[np.object_]
) -> dict[str, npt.NDArray[Any]]:
    inputs: dict[str, npt.NDArray[Any]] = {}
    for index, model_input in enumerate(model_inputs):
        column = values[:, index].reshape(-1, 1)
        if index == 0:
            inputs[model_input.name] = column.astype(np.float32)
        else:
            inputs[model_input.name] = column.astype(str)
    return inputs


def test_multimodal_missing_value_preprocessing_has_onnx_parity() -> None:
    training = np.asarray(
        [
            [1.0, "red", "2024-01-01", "falcons fly over mountains today"],
            [np.nan, np.nan, np.nan, np.nan],
            [3.0, "blue", "2024-03-15", "hawks circle over valleys today"],
            [7.0, "red", "2024-04-20", "eagles glide above forests today"],
        ],
        dtype=np.object_,
    )
    inference = np.asarray(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [5.0, "unseen", "2024-06-30", "unseen words remain harmless here"],
        ],
        dtype=np.object_,
    )
    schema = DatasetSchema(
        column_names=("numeric", "category", "date", "text"),
        column_types=(
            ColumnTypes.NUMERIC_REGULAR,
            ColumnTypes.CAT_LOW_CARD,
            ColumnTypes.DATE_YMD_ISO8601,
            ColumnTypes.TEXT_UTF8,
        ),
        target_name="target",
        target_kind="regression",
        dimensions=training.shape,
    )
    encoder = MultiModalEncoder()
    encoder.fit(training, np.arange(training.shape[0]), schema)

    expected = encoder.transform(inference)
    categorical_imputer = encoder.ct.transformers_[1][1].named_steps["imputer"]
    text_imputer = encoder.ct.transformers_[3][1].named_steps["imputer"]
    model = encoder.serialize().get_model()
    checker.check_model(model)
    actual = InferenceSession(model.SerializeToString()).run(
        None, _runtime_inputs(list(model.graph.input), inference)
    )[0]

    assert expected.shape[0] == inference.shape[0]
    assert np.isfinite(expected).all()
    assert categorical_imputer.transform(np.asarray([[np.nan]])).item() == (
        "__falcon_missing__"
    )
    assert text_imputer.transform(np.asarray([np.nan])).item() == ""
    assert all(
        node.domain in {"", "ai.onnx", "ai.onnx.ml"} for node in model.graph.node
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_numeric_preprocessing_always_emits_a_missing_indicator() -> None:
    training = np.asarray([[1.0], [3.0], [9.0]], dtype=np.object_)
    schema = DatasetSchema(
        column_names=("numeric",),
        column_types=(ColumnTypes.NUMERIC_REGULAR,),
        target_name="target",
        target_kind="regression",
        dimensions=training.shape,
    )
    encoder = MultiModalEncoder()
    encoder.fit(training, np.arange(training.shape[0]), schema)

    transformed = encoder.transform(np.asarray([[np.nan], [3.0]], dtype=np.object_))
    numeric_pipeline = encoder.ct.transformers_[0][1]

    np.testing.assert_array_equal(
        numeric_pipeline.named_steps["imputer"].statistics_, np.asarray([3.0])
    )
    assert transformed.shape == (2, 2)
    assert transformed[0, 0] == transformed[1, 0]
    assert transformed[0, 1] != transformed[1, 1]
