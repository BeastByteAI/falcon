import pandas as pd
import random
import numpy as np

data = { 
    'cat_lc' : [random.randint(0, 9) for i in range(128)],
    'cat_hc' : [f"f{i}" for i in range(128)],
    'num': [i for i in range(128)],
    'labels': [i/2 for i in range(128)]
}

df = pd.DataFrame(data)

from falcon import initialize

manager = initialize(task = 'tabular_regression', data = df, features=['cat_lc', 'cat_hc', 'num'], target='labels')
manager.train()
manager.save_model(filename='new_sc_enc.onnx')


