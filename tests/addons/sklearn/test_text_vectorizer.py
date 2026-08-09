import numpy as np
from onnx import helper
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

from falcon.addons.sklearn.preprocessing.text_vectorizer import (
    FalconTfidfVectorizer,
)


def test_text_vectorizer_onnx_uses_standard_ops_with_native_parity() -> None:
    training_documents = np.asarray(
        [
            "THE Falcon, flies quickly",
            "falcon, rests and waits",
            "alpha beta beta",
            "gamma alpha",
        ],
        dtype=object,
    )
    inference_documents = np.asarray(
        ["FALCON,   flies\tquickly", "the alpha\nbeta", "unknown token"],
        dtype=object,
    )
    vectorizer = FalconTfidfVectorizer().fit(training_documents)

    expected = vectorizer.transform(inference_documents).toarray()
    model = to_onnx(
        vectorizer,
        initial_types=[("X", StringTensorType([None]))],
        target_opset={"": 21, "ai.onnx.ml": 4},
    )
    session = InferenceSession(model.SerializeToString())
    actual = session.run(None, {"X": inference_documents})[0]

    node_types = {node.op_type for node in model.graph.node}
    assert {"StringNormalizer", "StringSplit", "TfIdfVectorizer"} <= node_types
    assert all(
        node.domain in {"", "ai.onnx", "ai.onnx.ml"} for node in model.graph.node
    )
    assert "the" not in vectorizer.vocabulary_
    tfidf_node = next(
        node for node in model.graph.node if node.op_type == "TfIdfVectorizer"
    )
    pool = helper.get_attribute_value(
        next(
            attribute
            for attribute in tfidf_node.attribute
            if attribute.name == "pool_strings"
        )
    )
    assert b"the" not in pool
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_text_vectorizer_onnx_matches_native_for_non_ascii_case() -> None:
    training_documents = np.asarray(
        ["CAFÉ déjà", "café DÉJÀ", "İSTANBUL şehir", "istanbul ŞEHİR"],
        dtype=object,
    )
    vectorizer = FalconTfidfVectorizer(stop_words=None).fit(training_documents)
    model = to_onnx(
        vectorizer,
        initial_types=[("X", StringTensorType([None]))],
        target_opset={"": 21, "ai.onnx.ml": 4},
    )

    expected = vectorizer.transform(training_documents).toarray()
    actual = InferenceSession(model.SerializeToString()).run(
        None, {"X": training_documents}
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
