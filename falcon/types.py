from typing import Union, List, Tuple, Optional
import numpy as np
from numpy import typing as npt
from enum import Enum

ColumnsList = Union[List[str], List[int]]
Float32Array = npt.NDArray[np.float32]
Int64Array = npt.NDArray[np.int64]


class ColumnTypes(Enum):
    NUMERIC_REGULAR = 0
    CAT_LOW_CARD = 1
    CAT_HIGH_CARD = 2
    TEXT_UTF8 = 3
    DATE_YMD_ISO8601 = 100  # %Y-%m-%d i.e. '2023-02-21'
    DATETIME_YMDHMS_ISO8601 = 101  # %Y-%m-%dT%H:%M:%SZ i.e. '2023-02-21T17:24:22Z' OR %Y-%m-%d %H:%M:%S i.e. '2023-02-21 17:24:22'
    