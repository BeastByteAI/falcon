from __future__ import annotations

from typing import Any

import numpy as np
from skl2onnx import update_registered_converter
from skl2onnx.common._apply_operation import (
    apply_identity,
    apply_normalizer,
    apply_reshape,
)
from skl2onnx.common.data_types import FloatTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

_ASCII_LOWERCASE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"
)


def _lower_ascii(text: str) -> str:
    return text.translate(_ASCII_LOWERCASE)


class FalconTfidfVectorizer(TfidfVectorizer):
    def __init__(
        self,
        *,
        max_features: int | None = 1024,
        stop_words: str | list[str] | None = "english",
    ) -> None:
        super().__init__(
            analyzer="word",
            input="content",
            lowercase=False,
            max_features=max_features,
            preprocessor=_lower_ascii,
            stop_words=stop_words,
            token_pattern=r"(?u)[^ ]+",
        )


def _text_shape_calculator(operator: Any) -> None:
    vectorizer: FalconTfidfVectorizer = operator.raw_operator
    check_is_fitted(vectorizer, "vocabulary_")
    batch_size = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = FloatTensorType(
        [batch_size, len(vectorizer.vocabulary_)]
    )


def _ordered_vocabulary(vectorizer: FalconTfidfVectorizer) -> list[str]:
    vocabulary = vectorizer.vocabulary_
    expected_indices = list(range(len(vocabulary)))
    if sorted(vocabulary.values()) != expected_indices:
        raise RuntimeError("Text vocabulary indices must be contiguous")

    terms = ["" for _ in vocabulary]
    for term, index in vocabulary.items():
        terms[index] = term
    return terms


def _text_converter(scope: Any, operator: Any, container: Any) -> None:
    vectorizer: FalconTfidfVectorizer = operator.raw_operator
    check_is_fitted(vectorizer, ("vocabulary_", "idf_"))
    vocabulary = _ordered_vocabulary(vectorizer)

    flattened = scope.get_unique_variable_name("flattened_documents")
    apply_reshape(
        scope,
        operator.inputs[0].full_name,
        flattened,
        container,
        desired_shape=(-1,),
    )

    normalized = scope.get_unique_variable_name("normalized_documents")
    container.add_node(
        "StringNormalizer",
        [flattened],
        [normalized],
        name=scope.get_unique_operator_name("StringNormalizer"),
        op_domain="",
        op_version=10,
        case_change_action="LOWER",
        is_case_sensitive=0,
        locale="C",
    )

    tokens = scope.get_unique_variable_name("document_tokens")
    token_counts = scope.get_unique_variable_name("document_token_counts")
    container.add_node(
        "StringSplit",
        [normalized],
        [tokens, token_counts],
        name=scope.get_unique_operator_name("StringSplit"),
        op_domain="",
        op_version=20,
        delimiter=" ",
    )

    tfidf = scope.get_unique_variable_name("tfidf_features")
    container.add_node(
        "TfIdfVectorizer",
        [tokens],
        [tfidf],
        name=scope.get_unique_operator_name("TfIdfVectorizer"),
        op_domain="",
        op_version=9,
        max_gram_length=1,
        max_skip_count=0,
        min_gram_length=1,
        mode="TFIDF",
        ngram_counts=[0],
        ngram_indexes=list(range(len(vocabulary))),
        pool_strings=vocabulary,
        weights=list(np.asarray(vectorizer.idf_, dtype=np.float32)),
    )

    output_name = operator.outputs[0].full_name
    if vectorizer.norm is None:
        apply_identity(scope, tfidf, output_name, container)
    else:
        apply_normalizer(
            scope,
            tfidf,
            output_name,
            container,
            norm=vectorizer.norm.upper(),
            use_float=True,
        )


update_registered_converter(
    FalconTfidfVectorizer,
    "FalconTfidfVectorizer",
    _text_shape_calculator,
    _text_converter,
)
