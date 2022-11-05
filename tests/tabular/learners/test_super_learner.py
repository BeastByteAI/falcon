import numpy as np

from falcon.tabular.learners.super_learner import _default_estimators, SuperLearner


def fit_superlearner(config):
    X = np.random.normal(size=(500, 16))
    task = config["task"]
    if task == "tabular_regression":
        y = np.random.uniform(size=(500, 1))
    else:
        y = np.random.randint(0, 3, size=(500,))

    model = SuperLearner(**config)
    model.fit(X, y)
    pred = model.predict(X)
    return pred


def gen_config(regr=True, size="mini"):
    task = "tabular_regression" if regr else "tabular_classification"
    base_estimators = _default_estimators[task][size]
    config = {
        "task": task,
        "base_estimators": base_estimators,
        "cv": 2,
        "filter_estimators": False,
    }
    return config


def assert_preds(pred):
    assert pred is not None
    assert len(pred) > 0
    assert None not in pred


def test_super_learner_regr_mini():
    config = gen_config(True, "mini")
    pred = fit_superlearner(config=config)
    assert_preds(pred)


def test_super_learner_regr_mid():
    config = gen_config(True, "mid")
    pred = fit_superlearner(config=config)
    assert_preds(pred)


def test_super_learner_regr_large():
    config = gen_config(True, "large")
    pred = fit_superlearner(config=config)
    assert_preds(pred)


def test_super_learner_clf_mini():
    config = gen_config(False, "mini")
    pred = fit_superlearner(config=config)
    assert_preds(pred)


def test_super_learner_clf_mid():
    config = gen_config(False, "mid")
    pred = fit_superlearner(config=config)
    assert_preds(pred)


def test_super_learner_clf_large():
    config = gen_config(False, "large")
    pred = fit_superlearner(config=config)
    assert_preds(pred)
