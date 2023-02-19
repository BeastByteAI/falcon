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
    DATE_YMD_ISO8601 = 100 # %Y-%m-%d