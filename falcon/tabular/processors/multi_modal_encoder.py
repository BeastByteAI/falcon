from typing import Any

from numpy import typing as npt
from skl2onnx.sklapi import CastTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.preprocessing import MaxAbsScaler

from falcon.addons.sklearn.decomposition.svd import ConditionalSVD
from falcon.addons.sklearn.preprocessing.date_tokenizer import DateTimeTokenizer
from falcon.addons.sklearn.preprocessing.text_vectorizer import (
    FalconTfidfVectorizer,
)
from falcon.tabular.processors.scaler_and_encoder import ScalerAndEncoder
from falcon.types import ColumnTypes, DatasetSchema

DATE_COLUMN_TYPES = (
    ColumnTypes.DATE_YMD_ISO8601,
    ColumnTypes.DATETIME_YMDHMS_ISO8601,
)


class MultiModalEncoder(ScalerAndEncoder):
    """
    Extends `ScalerAndEncoder` with tokenization of date/datetime features and
    tf-idf vectorization of text features.
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
                ("date_tokenizer", DateTimeTokenizer(format=f)),
                ("cast32", CastTransformer()),
                ("sc", MaxAbsScaler()),
            ]
        )

    def _reject_unsupported_date_columns(self, schema: DatasetSchema) -> None:
        if self.impute_missing:
            return
        unsupported = [
            name
            for name, column_type in zip(
                schema.column_names, schema.column_types, strict=True
            )
            if column_type in DATE_COLUMN_TYPES
        ]
        if unsupported:
            raise ValueError(
                "Date and datetime features are not supported while imputation is "
                f"disabled: {', '.join(unsupported)}"
            )

    def _get_text_tfidf(self) -> SKLPipeline:
        return SKLPipeline(
            steps=[
                ("imputer", self._get_string_imputer("")),
                (
                    "tfidf_vectorizer",
                    FalconTfidfVectorizer(),
                ),
                ("cast32", CastTransformer()),
                ("svd", ConditionalSVD(n_components=32)),
            ]
        )

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None:
        if X.ndim != 2 or X.shape[1] != schema.n_features:
            raise ValueError("Feature data does not match the dataset schema")
        self._reject_unsupported_date_columns(schema)
        self.column_types = schema.column_types
        transformers = []
        for i, v in enumerate(self.column_types):
            if v == ColumnTypes.CAT_LOW_CARD:
                method = self._get_ohe()
            elif v == ColumnTypes.CAT_HIGH_CARD:
                method = self._get_target_encoder(schema.target_kind)
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
            t: tuple[str, Any, int | list[int]]
            if v != ColumnTypes.TEXT_UTF8:
                t = (f"input {i}", method, [i])
            else:
                t = (f"input {i}", method, i)
            transformers.append(t)
        self.ct = ColumnTransformer(transformers)
        self.ct.fit(X, y)
        self._align_numeric_scalers_with_onnx()
