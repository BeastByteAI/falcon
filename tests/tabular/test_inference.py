from falcon import initialize
from falcon.utils import run_falcon, run_onnx
import numpy as np


def eval_saved_model(manager, is_regr=False, format="onnx"):
    X = manager._data[0]
    pred = manager.predict(X)
    model = manager.save_model(format=format, filename="test_model")
    if format == "onnx":
        pred_ = run_onnx(f"test_model.{format}", X)
        if len(pred_) > 1:
            print(
                "Onnx model returned multiple predictions. Only the first one will be used for testing."
            )
        pred_ = pred_[0].squeeze()
        print(pred_.shape)
    else:
        with open(f"test_model.{format}", "rb") as f:
            model_r = f.read()
        pred_ = run_falcon(model_r, X)
    if not is_regr:
        eq_ = np.equal(pred, pred_)
        return not False in eq_
    else:
        ac = np.allclose(pred, pred_, atol = 0.01)
        return ac, (pred, pred_)


def test_inference_classification():
    manager = initialize(
        task="tabular_classification", data="tests/extra_files/iris.csv"
    )
    manager.train(pre_eval=False)
    assert eval_saved_model(manager=manager, is_regr=False, format="onnx")
    assert eval_saved_model(manager=manager, is_regr=False, format="falcon")
    

def test_inference_regression():
    manager = initialize(
        task="tabular_regression",
        data="tests/extra_files/prices.csv",
        features="SqFt,Bedrooms,Bathrooms,Offers,Brick,Neighborhood".split(","),
        target="Price",
    )
    manager.train(pre_eval=False)
    ac, data =  eval_saved_model(manager=manager, is_regr=True, format="onnx")
    print(data)
    assert ac is True
    ac, data = eval_saved_model(manager=manager, is_regr=True, format="falcon")
    print(data)
    assert ac is True
