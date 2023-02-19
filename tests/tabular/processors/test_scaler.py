import numpy as np
from falcon.tabular.processors.scaler_and_encoder import ScalerAndEncoder
from falcon.types import ColumnTypes

def test_scaler_encoder():
    X = np.array([[1, 1, 1], [1, 2, 1]])
    expected_transformed = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    encoder = ScalerAndEncoder(mask=[ColumnTypes.NUMERIC_REGULAR, ColumnTypes.CAT_LOW_CARD, ColumnTypes.NUMERIC_REGULAR])
    encoder.fit(X)

    transformed = encoder.transform(X)

    assert np.allclose(expected_transformed, transformed)

    assert transformed.dtype == np.float32
