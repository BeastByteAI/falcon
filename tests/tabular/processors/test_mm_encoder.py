import pandas as pd
import numpy as np
from onnxruntime import InferenceSession
from falcon.types import ColumnTypes
from falcon.tabular.processors.multi_modal_encoder import MultiModalEncoder


def test_mm_news100():
    data = pd.read_csv("tests/extra_files/news100.csv").fillna("")
    data.pop('label')
    data = data.to_numpy()[:5, :]
    print('data shape ', data.shape)
    ct = [
        ColumnTypes.NUMERIC_REGULAR,
        ColumnTypes.TEXT_UTF8,
        ColumnTypes.CAT_LOW_CARD,
        ColumnTypes.TEXT_UTF8,
    ]

    enc = MultiModalEncoder(ct)

    enc.fit(data, None)

    y = enc.transform(data)
    print('y shape ', y.shape)

    assert y.dtype == np.float32, 'Incorrect return type'

    assert y.shape[0] == data.shape[0], 'Incorrect number of samples in the output'

    onx = enc.to_onnx().get_model()
    with open('tmp.onnx', 'wb+') as f:
        f.write(onx.SerializeToString())
    inps = {}
    for i, inp in enumerate(onx.graph.input):
        arr = data[:, i] 
        if len(arr.shape) < 2: 
            arr = np.expand_dims(arr, 1)
        print(i, arr.shape)
        if ct[i] != ColumnTypes.NUMERIC_REGULAR:
            arr = arr.astype(object)
        else: 
            arr = arr.astype(np.float32)
        inps[inp.name] = arr 

    sess = InferenceSession(onx.SerializeToString())
    got = sess.run(None, inps)[0]
    assert np.allclose(y, got, atol=1e-3), 'Incorrect onnx output'
