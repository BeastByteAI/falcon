import numpy as np

from falcon.tabular.processors.scaler_and_encoder import ScalerAndEncoder
from falcon.types import ColumnTypes, DatasetSchema


def test_scaler_encoder() -> None:
    X = np.array([[1, 1, 1], [1, 2, 1]])
    expected_transformed = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    schema = DatasetSchema(
        column_names=("numeric_a", "category", "numeric_b"),
        column_types=(
            ColumnTypes.NUMERIC_REGULAR,
            ColumnTypes.CAT_LOW_CARD,
            ColumnTypes.NUMERIC_REGULAR,
        ),
        target_name="target",
        target_kind="regression",
        dimensions=X.shape,
    )
    encoder = ScalerAndEncoder()
    encoder.fit(X, np.zeros(X.shape[0]), schema)

    transformed = encoder.transform(X)

    assert np.allclose(expected_transformed, transformed)

    assert transformed.dtype == np.float32
