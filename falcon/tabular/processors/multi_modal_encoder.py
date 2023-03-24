import numpy as np
from numpy import typing as npt
from falcon.types import Float32Array, ColumnTypes
from typing import List, Optional, Type, Any, Tuple, Union
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.preprocessing import MaxAbsScaler
from skl2onnx.sklapi import CastTransformer
from falcon.tabular.processors.scaler_and_encoder import ScalerAndEncoder
from falcon.addons.sklearn.preprocessing.date_tokenizer import DateTimeTokenizer
from falcon.addons.sklearn.decomposition.svd import ConditionalSVD


class MultiModalEncoder(ScalerAndEncoder):
    """
    Applies different types of encodings on numerical, categorical, text and date/datetime features.
    """

    def _get_date_tokenizer(self, ct: ColumnTypes) -> SKLPipeline:
        if ct == ColumnTypes.DATE_YMD_ISO8601:
            f = r"%Y-%m-%d"
        elif ct == ColumnTypes.DATETIME_YMDHMS_ISO8601:
            f = r"%Y-%m-%dT%H:%M:%SZ"
        else:
            raise ValueError("Unknown column type encountered")
        return SKLPipeline(
            steps=[
                ("cast_str", CastTransformer(dtype=np.str_)),
                ("date_tokenizer", DateTimeTokenizer(format=f)),
                ("cast32", CastTransformer()),
                ("sc", MaxAbsScaler()),
            ]
        )

    def _get_text_tfidf(self) -> SKLPipeline:
        return SKLPipeline(
            steps=[
                ("cast_str", CastTransformer(dtype=np.str_)),
                (
                    "tfidf_vectorizer",
                    TfidfVectorizer(
                        stop_words="english",
                        input="content",
                        analyzer="word",
                        max_features=1024,
                        token_pattern="[a-zA-Z0-9_]+",
                    ),
                ),
                ("cast32", CastTransformer()),
                ("svd", ConditionalSVD(n_components=32)),
            ]
        )

    def fit(self, X: npt.NDArray, y: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Fits the encoder.

        Parameters
        ----------
        X : npt.NDArray
            data to encode
        _ : Any, optional
            dummy argument to keep compatibility with pipeline training
        """
        transformers = []
        for i, v in enumerate(self.mask):
            if v == ColumnTypes.CAT_LOW_CARD:
                method = self._get_ohe()
            elif v == ColumnTypes.NUMERIC_REGULAR:
                method = self._get_numeric_scaler()
            elif v in [
                ColumnTypes.DATE_YMD_ISO8601,
                ColumnTypes.DATETIME_YMDHMS_ISO8601,
            ]:
                method = self._get_date_tokenizer(v)
            elif v == ColumnTypes.TEXT_UTF8:
                method = self._get_text_tfidf()
            else:
                method = self._get_ordinal_encoder()
            t: Tuple[str, Any, Union[int, List[int]]]
            if v != ColumnTypes.TEXT_UTF8:
                t = (f"input {i}", method, [i])
            else:
                t = (f"input {i}", method, i)
            transformers.append(t)
        self.ct = ColumnTransformer(transformers)
        self.ct.fit(X)
