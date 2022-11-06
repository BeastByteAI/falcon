import pandas as pd
import random
import numpy as np

data = { 
    'cat_lc' : [random.randint(0, 9) for i in range(128)],
    'cat_hc' : [f"f{i}" for i in range(128)],
    'static': [0 for _ in range(128)], 
    'num': [i for i in range(128)],
    'labels': [i/2 for i in range(128)]
}

df = pd.DataFrame(data)

from falcon import initialize
from falcon.utils import run_onnx

sample = np.asarray([[12, 'unk', 1, 256], [10, 'f10', 0, 10]])

manager = initialize(task = 'tabular_regression', data = df, features=['cat_lc', 'cat_hc', 'static', 'num'], target='labels')
manager.train()
manager.save_model(filename='new_sc_enc.onnx')

pred_onnx = run_onnx('new_sc_enc.onnx', sample)
pred = manager.predict(sample)

print(pred, pred_onnx)