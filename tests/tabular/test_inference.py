from falcon import initialize
import numpy as np
from sklearn.metrics import r2_score
import random
from falcon.task_configurations import get_task_configuration
from falcon.runtime import Runtime
from dataclasses import dataclass
from typing import Callable, List, Dict
import os


def eval_saved_model(manager, is_regr=False, prefix=""):
    X = manager._data[0]
    y = manager._data[1]
    pred = manager.predict(X)

    filename = f"{prefix}test_model.fnnx"
    manager.save_model(filename=filename)

    pred_ = Runtime(filename).predict(X)

    os.remove(filename)
    
    if not is_regr:
        eq_ = np.equal(pred, pred_)
        print(eq_)
        return False not in eq_
    else:
        if len(pred_.shape) > len(pred.shape):
            pred_ = pred_.squeeze()
        assert pred.shape == pred_.shape, "Shapes are not equal"
        ac = np.isclose(pred, pred_)
        assert False not in ac
        ac = len(ac[ac == False])
        ac = ac / len(pred) < 0.1
        me1 = r2_score(pred, y)
        me2 = r2_score(pred_, y)
        mset = 0.001
        msec = np.abs(me1 - me2) < mset
        return ac, msec, (pred, pred_)


def inference_classification(config, config_name):
    random.seed(42)
    np.random.seed(42)
    manager = initialize(
        task="tabular_classification", data="tests/extra_files/iris.csv", **config
    )
    manager.train(pre_eval=False)
    print("model ", manager._pipeline._pipeline[1].model)
    print("task ", manager._pipeline._pipeline[1].task)
    assert eval_saved_model(
        manager=manager, is_regr=False, prefix=f"clf_{config_name}_"
    )


def inference_regression(config, config_name):
    manager = initialize(
        task="tabular_regression",
        data="tests/extra_files/prices.csv",
        features="SqFt,Bedrooms,Bathrooms,Offers,Brick,Neighborhood".split(","),
        target="Price",
        **config,
    )
    manager.train(pre_eval=False)
    ac, msec, data = eval_saved_model(
        manager=manager, is_regr=True, prefix=f"regr_{config_name}_"
    )
    assert ac
    assert msec


@dataclass
class _TestCase:
    config_name: str
    config_fn: Callable


def _cv2(config):
    config["extra_pipeline_options"]["learner_kwargs"]["cv"] = 2


def _tr2(config):
    config["extra_pipeline_options"]["learner_kwargs"]["n_trials"] = 2


_TEST_CASES = [
    _TestCase("SuperLearner", lambda c: None),
    _TestCase("SuperLearner.mini", _cv2),
    _TestCase("SuperLearner.mid", _cv2),
    _TestCase("SuperLearner.large", _cv2),
    _TestCase("SuperLearner.xlarge", _cv2),
    _TestCase("OptunaLearner.hgbt", _tr2),
    _TestCase("PlainLearner", lambda c: None),
    _TestCase("PlainLearner.hgbt", lambda c: None),
]


_TASK_TEST_SCENARIOS: Dict[str, List[_TestCase]] = {
    "tabular_classification": _TEST_CASES,
    "tabular_regression": _TEST_CASES,
}


def test_inference():
    for task, test_cases in _TASK_TEST_SCENARIOS.items():
        for test_case in test_cases:
            config = get_task_configuration(
                task=task, configuration_name=test_case.config_name
            )
            test_case.config_fn(config)
            if task == "tabular_classification":
                inference_classification(
                    config=config, config_name=test_case.config_name
                )
            else:
                inference_regression(config=config, config_name=test_case.config_name)
