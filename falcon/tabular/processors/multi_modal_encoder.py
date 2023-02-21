import numpy as np
from numpy import typing as npt
from falcon.types import Float32Array, ColumnTypes
from sklearn.compose import ColumnTransformer
from typing import List, Optional, Type, Any
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.preprocessing import MaxAbsScaler, OrdinalEncoder
from skl2onnx.sklapi import CastTransformer
from falcon.tabular.processors.scaler_and_encoder import ScalerAndEncoder
from falcon.addons.sklearn.preprocessing.date_tokenizer import DateTimeTokenizer

class MultiModalEncoder(ScalerAndEncoder):
    def _get_date_tokenizer(self, ct: ColumnTypes) -> BaseEstimator:
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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
                ColumnTypes.DATETIME_YMDHMS_ISO8601
            ]:
                method = self._get_date_tokenizer(v)
            else:
                method = self._get_ordinal_encoder()
            t = (f"input {i}", method, [i])
            transformers.append(t)

        self.ct = ColumnTransformer(transformers)
        self.ct.fit(X)