from typing import Union, List, Tuple, Optional
import numpy as np
from numpy import typing as npt


ColumnsList = Union[List[str], List[int]]
Float32Array = npt.NDArray[np.float32]
Int64Array = npt.NDArray[np.int64]
SerializedModelTuple = Tuple[bytes, int, int, List[str], List[List[Optional[int]]]]
ModelsList = List[Tuple[SerializedModelTuple, str]]
