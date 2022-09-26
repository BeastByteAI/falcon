import numpy as np
from falcon.tabular.processors.scaler_and_encoder import ScalerAndEncoder


def test_scaler_encoder():
    X = np.array([[1, 1, 1], [1, 2, 1]])
    expected_transformed = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    encoder = ScalerAndEncoder(mask=[False, True, False])
    encoder.fit(X)

    transformed = encoder.transform(X)

    assert np.allclose(expected_transformed, transformed)

    assert transformed.dtype == np.float32
