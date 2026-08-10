import numpy as np
import pandas as pd
from onnxruntime import InferenceSession
from sklearn.utils.validation import check_is_fitted

from falcon.tabular.processors.multi_modal_encoder import MultiModalEncoder
from falcon.types import ColumnTypes, DatasetSchema


def test_date_encoder_pipeline_is_recognized_as_fitted() -> None:
    data = np.asarray([["2022-02-02"], ["2022-02-25"], ["2022-05-02"]])
    schema = DatasetSchema(
        column_names=("date",),
        column_types=(ColumnTypes.DATE_YMD_ISO8601,),
        target_name="target",
        target_kind="regression",
        dimensions=data.shape,
    )
    encoder = MultiModalEncoder()

    encoder.fit(data, np.zeros(data.shape[0]), schema)

    date_pipeline = encoder.ct.transformers_[0][1]
    check_is_fitted(date_pipeline)


def test_mm_news100() -> None:
    data = pd.read_csv("tests/extra_files/news100.csv").fillna("")
    data.pop("label")
    data = data.to_numpy()[:5, :]
    print("data shape ", data.shape)
    ct = (
        ColumnTypes.NUMERIC_REGULAR,
        ColumnTypes.TEXT_UTF8,
        ColumnTypes.CAT_LOW_CARD,
        ColumnTypes.TEXT_UTF8,
    )
    schema = DatasetSchema(
        column_names=("timedelta", "title", "topic", "content"),
        column_types=ct,
        target_name="target",
        target_kind="regression",
        dimensions=data.shape,
    )

    enc = MultiModalEncoder()

    enc.fit(data, np.zeros(data.shape[0]), schema)

    y = enc.transform(data)
    print("y shape ", y.shape)

    assert y.dtype == np.float32, "Incorrect return type"

    assert y.shape[0] == data.shape[0], "Incorrect number of samples in the output"

    onx = enc.serialize().get_model()
    inps = {}
    for i, inp in enumerate(onx.graph.input):
        arr = data[:, i]
        if len(arr.shape) < 2:
            arr = np.expand_dims(arr, 1)
        print(i, arr.shape)
        if ct[i] != ColumnTypes.NUMERIC_REGULAR:
            arr = arr.astype(object)
        else:
            arr = arr.astype(np.float32)
        inps[inp.name] = arr

    sess = InferenceSession(onx.SerializeToString())
    got = sess.run(None, inps)[0]
    assert np.allclose(y, got, atol=1e-3), "Incorrect onnx output"
