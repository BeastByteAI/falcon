import numpy as np
import pytest
from numpy import typing as npt
from onnx import TensorProto, checker
from onnxruntime import InferenceSession
from skl2onnx import to_onnx

from falcon.addons.sklearn.preprocessing.date_tokenizer import DateTimeTokenizer

DATE_FORMAT = r"%Y-%m-%d"
DATETIME_FORMAT = r"%Y-%m-%dT%H:%M:%SZ"

DATE_VALUES = np.asarray([["2022-02-02"], ["2022-02-25"], ["2022-05-02"]], dtype=object)
DATE_COMPONENTS = np.asarray(
    [[2022, 2, 2], [2022, 2, 25], [2022, 5, 2]], dtype=np.float64
)
DATETIME_CASES = [
    pytest.param(
        np.asarray(
            [
                ["2022-02-02T12:13:14Z"],
                ["2022-02-25T15:16:17Z"],
                ["2022-05-02T18:19:20Z"],
            ],
            dtype=object,
        ),
        id="t-delimiter-with-z",
    ),
    pytest.param(
        np.asarray(
            [
                ["2022-02-02T12:13:14"],
                ["2022-02-25T15:16:17"],
                ["2022-05-02T18:19:20"],
            ],
            dtype=object,
        ),
        id="t-delimiter-without-z",
    ),
    pytest.param(
        np.asarray(
            [
                ["2022-02-02 12:13:14Z"],
                ["2022-02-25 15:16:17Z"],
                ["2022-05-02 18:19:20Z"],
            ],
            dtype=object,
        ),
        id="space-delimiter-with-z",
    ),
    pytest.param(
        np.asarray(
            [
                ["2022-02-02 12:13:14"],
                ["2022-02-25 15:16:17"],
                ["2022-05-02 18:19:20"],
            ],
            dtype=object,
        ),
        id="space-delimiter-without-z",
    ),
]
DATETIME_COMPONENTS = np.asarray(
    [
        [2022, 2, 2, 12, 13, 14],
        [2022, 2, 25, 15, 16, 17],
        [2022, 5, 2, 18, 19, 20],
    ],
    dtype=np.float64,
)


def _expected_features(
    components: npt.NDArray[np.float64],
    missing: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    if components.shape[1] == 3:
        cyclic_components = components[:, [1, 2]]
        periods = np.asarray([12, 31], dtype=np.float64)
    else:
        cyclic_components = components[:, [1, 2, 3, 4, 5]]
        periods = np.asarray([12, 31, 24, 60, 60], dtype=np.float64)
    angles = cyclic_components * ((2 * np.pi) / periods)
    if missing is None:
        missing = np.zeros((components.shape[0], 1), dtype=np.float64)
    return np.concatenate((components, np.sin(angles), np.cos(angles), missing), axis=1)


def _assert_onnx_parity(
    tokenizer: DateTimeTokenizer, values: npt.NDArray[np.object_]
) -> None:
    expected = tokenizer.transform(values)
    model = to_onnx(tokenizer, values)

    checker.check_model(model)
    assert all(
        node.domain in {"", "ai.onnx", "ai.onnx.ml"} for node in model.graph.node
    )
    assert {"Cast", "Cos", "Sin", "StringSplit"}.issubset(
        {node.op_type for node in model.graph.node}
    )
    cast_targets = {
        attribute.i
        for node in model.graph.node
        if node.op_type == "Cast"
        for attribute in node.attribute
        if attribute.name == "to"
    }
    assert TensorProto.INT64 in cast_targets

    session = InferenceSession(model.SerializeToString())
    actual = session.run(None, {"X": values})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_date_tokenizer_preserves_raw_and_adds_cyclical_features() -> None:
    tokenizer = DateTimeTokenizer(format=DATE_FORMAT).fit(DATE_VALUES)

    actual = tokenizer.transform(DATE_VALUES)

    np.testing.assert_allclose(
        actual, _expected_features(DATE_COMPONENTS), rtol=1e-6, atol=1e-5
    )
    assert actual.shape == (3, 8)
    assert actual.dtype == np.float64


def test_date_tokenizer_onnx_parity() -> None:
    tokenizer = DateTimeTokenizer(format=DATE_FORMAT).fit(DATE_VALUES)

    _assert_onnx_parity(tokenizer, DATE_VALUES)


@pytest.mark.parametrize("values", DATETIME_CASES)
def test_datetime_tokenizer_detects_variant_and_adds_cyclical_features(
    values: npt.NDArray[np.object_],
) -> None:
    tokenizer = DateTimeTokenizer(format=DATETIME_FORMAT).fit(values)

    actual = tokenizer.transform(values)

    np.testing.assert_allclose(
        actual, _expected_features(DATETIME_COMPONENTS), rtol=1e-6, atol=1e-5
    )
    assert actual.shape == (3, 17)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("values", DATETIME_CASES)
def test_datetime_tokenizer_onnx_parity(
    values: npt.NDArray[np.object_],
) -> None:
    tokenizer = DateTimeTokenizer(format=DATETIME_FORMAT).fit(values)

    _assert_onnx_parity(tokenizer, values)


def test_datetime_tokenizer_rejects_mixed_variants_at_fit() -> None:
    values = np.asarray(
        [["2022-02-02T12:13:14Z"], ["2022-02-25 15:16:17"]], dtype=object
    )

    with pytest.raises(ValueError, match="single datetime format variant"):
        DateTimeTokenizer(format=DATETIME_FORMAT).fit(values)


def test_datetime_tokenizer_rejects_variant_change_at_inference() -> None:
    tokenizer = DateTimeTokenizer(format=DATETIME_FORMAT).fit(
        np.asarray([["2022-02-02T12:13:14Z"]], dtype=object)
    )

    with pytest.raises(ValueError, match="fitted datetime format variant"):
        tokenizer.transform(np.asarray([["2022-02-02 12:13:14"]], dtype=object))


def test_date_tokenizer_imputes_missing_values_with_reference_and_indicator() -> None:
    training_values = np.asarray(
        [["2022-02-02"], [np.nan], ["2022-05-02"]], dtype=object
    )
    inference_values = np.asarray([[np.nan], ["2022-02-25"]], dtype=object)
    tokenizer = DateTimeTokenizer(format=DATE_FORMAT).fit(training_values)

    actual = tokenizer.transform(inference_values)
    expected = _expected_features(
        DATE_COMPONENTS[:2], np.asarray([[1.0], [0.0]], dtype=np.float64)
    )

    assert tokenizer.reference_value_ == "2022-02-02"
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
    _assert_onnx_parity(tokenizer, inference_values.astype(str))
