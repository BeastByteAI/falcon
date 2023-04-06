from numpy import typing as npt
import numpy as np
import pandas as pd
from typing import List, Optional, Any
from falcon.types import ColumnTypes
import re

NUM_CAT_THRESHOLD: int = 10
HIGH_CARD_THRESHOLD: int = 100
NP_NUMERIC_TYPES = (int, float, np.int32, np.int64, np.float32, np.float64)
REGEX_MAYBE_DATE = r"[0-9]+[/-][0-9]+[/-][0-9]+"
# not strictly valid, but should be sufficient
REGEX_UTC_LIKE = r"[0-9]+[-][0-9]{2}[-][0-9]{2}[ T]{1}[0-9]{2}[:][0-9]{2}[:][0-9]{2}Z?"
REGEX_UTF_TOKEN = r"(?u)\b\w\w+\b"


def _fullmatch(expr: str, x: Any) -> bool:
    fm = re.fullmatch(expr, x) is not None
    return fm


def _determine_date_type(X: pd.DataFrame, column: int) -> Optional[ColumnTypes]:
    if (
        pd.to_datetime(X.iloc[:, column], format=r"%Y-%m-%d", errors="coerce")
        .notnull()
        .all()
    ):
        return ColumnTypes.DATE_YMD_ISO8601
    return None


def determine_column_types(data: npt.NDArray) -> List[ColumnTypes]:
    mask: List[ColumnTypes] = []
    tmp_df: pd.DataFrame = pd.DataFrame(data).infer_objects()
    # print(tmp_df.dtypes.apply(lambda x: x.name).to_dict())
    for col in range(tmp_df.shape[-1]):
        determined_type = None
        if tmp_df.iloc[:, col].map(lambda x: isinstance(x, NP_NUMERIC_TYPES)).all():
            if len(tmp_df.iloc[:, col].unique().tolist()) > NUM_CAT_THRESHOLD:
                determined_type = ColumnTypes.NUMERIC_REGULAR
            else:
                determined_type = ColumnTypes.CAT_LOW_CARD
        if determined_type is None:
            tmp_df[tmp_df.columns[col]] = tmp_df.iloc[:, col].astype(str)
            if tmp_df.iloc[:, col].map(lambda x: _fullmatch(REGEX_MAYBE_DATE, x)).all():
                determined_type = _determine_date_type(tmp_df, col)
            elif tmp_df.iloc[:, col].map(lambda x: _fullmatch(REGEX_UTC_LIKE, x)).all():
                determined_type = ColumnTypes.DATETIME_YMDHMS_ISO8601
            elif (
                tmp_df.iloc[:, col]
                .map(lambda x: len(re.findall(REGEX_UTF_TOKEN, x)))
                .median()
                > 5
            ):
                determined_type = ColumnTypes.TEXT_UTF8
        if determined_type is None:
            if len(tmp_df.iloc[:, col].unique().tolist()) > HIGH_CARD_THRESHOLD:
                determined_type = ColumnTypes.CAT_HIGH_CARD
            else:
                determined_type = ColumnTypes.CAT_LOW_CARD
        mask.append(determined_type)
    return mask
