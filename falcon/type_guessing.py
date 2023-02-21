from numpy import typing as npt
import numpy as np
import pandas as pd
from typing import List, Optional
from falcon.types import ColumnTypes
import re

NUM_CAT_THRESHOLD: int = 10
HIGH_CARD_THRESHOLD: int = 100
NP_NUMERIC_TYPES = (int, float, np.int32, np.int64, np.float32, np.float64)
REGEX_MAYBE_DATE = r"[0-9]+[/-][0-9]+[/-][0-9]+"
# not strictly valid, but should be sufficient
REGEX_UTC_LIKE = r"[0-9]+[-][0-9]{2}[-][0-9]{2}[ T]{1}[0-9]{2}[:][0-9]{2}[:][0-9]{2}Z?"


def _determine_date_type(X: np.ndarray, column: int) -> Optional[ColumnTypes]:
    df = pd.DataFrame(X)
    if (
        pd.to_datetime(df.iloc[:, column], format=r"%Y-%m-%d", errors="coerce")
        .notnull()
        .all()
    ):
        return ColumnTypes.DATE_YMD_ISO8601
    return None


def determine_column_types(data: npt.NDArray) -> List[ColumnTypes]:
    mask: List[ColumnTypes] = []
    tmp_df: pd.DataFrame = pd.DataFrame(data).infer_objects()
    for col in range(tmp_df.shape[-1]):
        determined_type = None
        if tmp_df.iloc[:, col].map(lambda x: isinstance(x, NP_NUMERIC_TYPES)).all():
            if len(tmp_df.iloc[:, col].unique().tolist()) > NUM_CAT_THRESHOLD:
                determined_type = ColumnTypes.NUMERIC_REGULAR
            else:
                determined_type = ColumnTypes.CAT_LOW_CARD
        elif (
            tmp_df.iloc[:, col]
            .map(lambda x: re.fullmatch(REGEX_MAYBE_DATE, x) is not None)
            .all()
        ):
            determined_type = _determine_date_type(tmp_df, col)
        elif (
            tmp_df.iloc[:, col]
            .map(lambda x: re.fullmatch(REGEX_UTC_LIKE, x) is not None)
            .all()
        ):
            determined_type = ColumnTypes.DATETIME_YMDHMS_ISO8601

        if determined_type is None:
            if len(tmp_df.iloc[:, col].unique().tolist()) > HIGH_CARD_THRESHOLD:
                determined_type = ColumnTypes.CAT_HIGH_CARD
            else:
                determined_type = ColumnTypes.CAT_LOW_CARD
        mask.append(determined_type)
    return mask
