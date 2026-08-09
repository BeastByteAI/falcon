from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy import typing as npt

ColumnsList = list[str] | list[int]
Float32Array = npt.NDArray[np.float32]
Int64Array = npt.NDArray[np.int64]


class ColumnTypes(Enum):
    NUMERIC_REGULAR = 0
    CAT_LOW_CARD = 1
    CAT_HIGH_CARD = 2
    TEXT_UTF8 = 3
    DATE_YMD_ISO8601 = 100  # %Y-%m-%d i.e. '2023-02-21'
    DATETIME_YMDHMS_ISO8601 = 101  # %Y-%m-%dT%H:%M:%SZ i.e. '2023-02-21T17:24:22Z' OR %Y-%m-%d %H:%M:%S i.e. '2023-02-21 17:24:22'


TargetKind = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetSchema:
    column_names: tuple[str, ...]
    column_types: tuple[ColumnTypes, ...]
    target_name: str
    target_kind: TargetKind
    dimensions: tuple[int, int]

    @property
    def n_rows(self) -> int:
        return self.dimensions[0]

    @property
    def n_features(self) -> int:
        return self.dimensions[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "columns": [
                {"name": name, "type": column_type.name}
                for name, column_type in zip(
                    self.column_names, self.column_types, strict=True
                )
            ],
            "target": {"name": self.target_name, "kind": self.target_kind},
            "dimensions": {
                "rows": self.n_rows,
                "features": self.n_features,
            },
        }
