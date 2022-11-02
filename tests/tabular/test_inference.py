from falcon import initialize
from falcon.utils import run_falcon, run_onnx
import numpy as np
from sklearn.metrics import r2_score 
import random

def eval_saved_model(manager, is_regr=False, format="onnx", prefix = ''):
    X = manager._data[0]
    y = manager._data[1]
    pred = manager.predict(X)
    manager.save_model(format=format, filename=f"{prefix}test_model")
    if format == "onnx":
        pred_ = run_onnx(f"{prefix}test_model.{format}", X)
        if len(pred_) > 1:
            print(
                "Onnx model returned multiple predictions. Only the first one will be used for testing."
            )
        pred_ = pred_[0].squeeze()
        print(pred_.shape)
    else:
        with open(f"{prefix}test_model.{format}", "rb") as f:
            model_r = f.read()
        pred_ = run_falcon(model_r, X)
    if not is_regr:
        eq_ = np.equal(pred, pred_)
        print(eq_)
        return len(eq_[eq_ == False]) / len(eq_) < 0.1
    else:
        ac = np.isclose(pred, pred_)
        ac = len(ac[ac == False]) 
        print(ac, len(pred))
        ac = ac / len(pred) < 0.1
        me1 = r2_score(pred, y)
        me2 = r2_score(pred_, y)
        mset = 0.001
        msec = np.abs(me1 - me2) < mset
        print(me1, me2, np.abs(me1 - me2), msec)
        print(ac)
        return ac, msec, (pred, pred_)


def test_inference_classification():
    for i in range(3):
        random.seed(42+i)
        np.random.seed(42+i)
        print(i)
        manager = initialize(
            task="tabular_classification", data="tests/extra_files/iris.csv"
        )
        manager.train(pre_eval=False)
        assert eval_saved_model(manager=manager, is_regr=False, format="onnx")
        assert eval_saved_model(manager=manager, is_regr=False, format="falcon")
    

def test_inference_regression():
    for i in range(3):
        random.seed(42+i)
        np.random.seed(42+i)
        manager = initialize(
        task="tabular_regression",
        data="tests/extra_files/prices.csv",
        features="SqFt,Bedrooms,Bathrooms,Offers,Brick,Neighborhood".split(","),
        target="Price",
        )
        manager.train(pre_eval=False)
        ac, msec, data =  eval_saved_model(manager=manager, is_regr=True, format="onnx")
        print(i, data)
        assert ac
        assert msec 
        ac, msec, data = eval_saved_model(manager=manager, is_regr=True, format="falcon")
        print(i, data)
        assert ac
        assert msec 
