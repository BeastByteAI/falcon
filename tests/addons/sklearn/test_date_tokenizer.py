from falcon.addons.sklearn.preprocessing.date_tokenizer import DateTimeTokenizer
import numpy as np
from onnxruntime import InferenceSession
from skl2onnx import to_onnx


def test_date_tokenizer_ymd():
    a = np.asarray([["2022-02-02"], ["2022-02-25"], ["2022-05-02"]])

    dt = DateTimeTokenizer(format=r"%Y-%m-%d")
    dt.fit(None)
    got = dt.transform(a).astype(object)
    exp = np.asarray(
        [["2022", "02", "02"], ["2022", "02", "25"], ["2022", "05", "02"]]
    ).astype(object)

    assert np.equal(got, exp).all()


def test_date_tokenizer_ymd_onnx():
    a = np.asarray([["2022-02-02"], ["2022-02-25"], ["2022-05-02"]])

    dt = DateTimeTokenizer(format=r"%Y-%m-%d")
    dt.fit(None)
    exp = dt.transform(a).astype(object)

    onx = to_onnx(dt, a.astype(object))
    sess = InferenceSession(onx.SerializeToString())

    got = sess.run(None, {"X": a.astype(object)})[0]

    assert np.equal(got, exp).all()


def test_datetime_tokenizer_ymd_spaced():
    a = np.asarray(
        [["2022-02-02 12:13:14"], ["2022-02-25 15:16:17"], ["2022-05-02 18:19:20"]]
    )

    dt = DateTimeTokenizer(format=r"%Y-%m-%d %H:%M:%S")
    dt.fit(None)
    got = dt.transform(a).astype(object)
    exp = np.asarray(
        [
            ["2022", "02", "02", "12", "13", "14"],
            ["2022", "02", "25", "15", "16", "17"],
            ["2022", "05", "02", "18", "19", "20"],
        ]
    ).astype(object)

    assert np.equal(got, exp).all()


def test_datetime_tokenizer_ymd_spaced_onnx():
    a = np.asarray(
        [["2022-02-02 12:13:14"], ["2022-02-25 15:16:17"], ["2022-05-02 18:19:20"]]
    )

    dt = DateTimeTokenizer(format=r"%Y-%m-%d %H:%M:%S")
    dt.fit(None)
    exp = dt.transform(a).astype(object)

    onx = to_onnx(dt, a.astype(object))
    sess = InferenceSession(onx.SerializeToString())

    got = sess.run(None, {"X": a.astype(object)})[0]

    assert np.equal(got, exp).all()


def test_datetime_tokenizer_ymd():
    a = np.asarray(
        [["2022-02-02T12:13:14Z"], ["2022-02-25T15:16:17Z"], ["2022-05-02T18:19:20Z"]]
    )

    dt = DateTimeTokenizer(format=r"%Y-%m-%dT%H:%M:%SZ")
    dt.fit(None)
    got = dt.transform(a).astype(object)
    exp = np.asarray(
        [
            ["2022", "02", "02", "12", "13", "14"],
            ["2022", "02", "25", "15", "16", "17"],
            ["2022", "05", "02", "18", "19", "20"],
        ]
    ).astype(object)

    assert np.equal(got, exp).all()


def test_datetime_tokenizer_ymd_onnx():
    a = np.asarray(
        [["2022-02-02T12:13:14Z"], ["2022-02-25T15:16:17Z"], ["2022-05-02T18:19:20Z"]]
    )

    dt = DateTimeTokenizer(format=r"%Y-%m-%dT%H:%M:%SZ")
    dt.fit(None)
    exp = dt.transform(a).astype(object)

    onx = to_onnx(dt, a.astype(object))
    sess = InferenceSession(onx.SerializeToString())

    got = sess.run(None, {"X": a.astype(object)})[0]

    assert np.equal(got, exp).all()
