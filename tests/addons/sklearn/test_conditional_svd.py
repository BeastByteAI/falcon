import numpy as np 
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from falcon.addons.sklearn.decomposition.svd import ConditionalSVD

def test_svd_id():
    X = np.random.uniform(size = (100, 24))
    svd = ConditionalSVD(n_components = 32)
    y = svd.fit_transform(X)
    assert np.equal(X, y).all()

    X = np.random.uniform(size = (100, 24))
    svd = ConditionalSVD(n_components = 32)
    y = svd.fit(X).transform(X)
    assert np.equal(X, y).all()

def test_svd():
    X = np.random.uniform(size = (100, 64))
    svd = ConditionalSVD(n_components = 32)
    y = svd.fit_transform(X)
    assert y.shape[-1] == 32

    X = np.random.uniform(size = (100, 64))
    svd = ConditionalSVD(n_components = 32)
    y = svd.fit(X).transform(X)
    assert y.shape[-1] == 32

def test_svd_onnx():
    X = np.random.uniform(size = (100, 64))
    svd = ConditionalSVD(n_components = 32)
    expected = svd.fit(X).transform(X)
    onx = to_onnx(svd, X)
    sess = InferenceSession(onx.SerializeToString())
    got = sess.run(None, {"X": X})[0]
    assert np.allclose(expected, got)

def test_svd_id_onnx():
    X = np.random.uniform(size = (100, 24))
    svd = ConditionalSVD(n_components = 32)
    _ = svd.fit(X).transform(X)
    onx = to_onnx(svd, X)
    sess = InferenceSession(onx.SerializeToString())
    got = sess.run(None, {"X": X})[0]
    assert np.allclose(X, got)