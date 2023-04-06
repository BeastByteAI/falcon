from sklearn.utils.estimator_checks import check_estimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from falcon.sklapi import (
    FalconRegressor,
    FalconTabularRegressor,
    FalconClassifier,
    FalconTabularClassifier,
)
from falcon.abstract import Model, ONNXConvertible, Pipeline
from falcon.tabular.pipelines import SimpleTabularPipeline
from falcon.tabular.learners import PlainLearner
from unittest.case import SkipTest


class DummyTestPipeline(Pipeline):
    def __init__(self, task, learner, learner_kwargs, dataset_size, **kwargs):
        mask = kwargs.get("mask", [])
        super().__init__(task=task, dataset_size=dataset_size, mask = mask)
        self.add_element(learner(task=task, **learner_kwargs))

    def fit(self, X, y) -> None:
        if self.task == "tabular_classification":
            y = y.astype(np.str_)
        for p in self._pipeline:
            p.fit_pipe(X, y) 
            X = p.forward(X)

    def predict(self, X, *args, **kwargs):
        for p in self._pipeline:
            X = p.forward(X)
        return X


class FalconSklModelWrapper(Model, ONNXConvertible):
    def __init__(self, model, **kwargs):
        self._model = model
        self._kwargs = kwargs
        # kwargs['random_state'] = 42

    def fit(self, X, y):
        self.model_ = self._model(**self._kwargs)
        self.model_.fit(X, y)

    def predict(self, X):
        return self.model_.predict(X)

    def to_onnx(self):
        pass


class FalconR(FalconSklModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(model=LinearRegression, **kwargs)

    def fit(self, X, y):
        self.model_ = self._model(**self._kwargs)
        self.model_.fit(X, y)

    def predict(self, X):
        return self.model_.predict(X)

    def to_onnx(self):
        pass


class FalconC(FalconSklModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(model=LogisticRegression)


def _test_regr(est):
    tests = check_estimator(est, generate_only=True)
    for t in tests:
        if t[1].func.__name__ in [
            "check_no_attributes_set_in_init",
            "check_fit_score_takes_y",
            "check_estimators_fit_returns_self",
            "check_estimator_get_tags_default_keys",
            "check_regressors_train",
            "check_estimators_unfitted",
            "check_set_params",
            "check_dont_overwrite_parameters",
            "check_n_features_in"
        ]:
            print(t)
            try:
                t[1](t[0])
            except SkipTest:
                pass


def _test_clf(est):
    tests = check_estimator(est, generate_only=True)
    for t in tests:
        if t[1].func.__name__ in [
            "check_no_attributes_set_in_init",
            "check_fit_score_takes_y",
            "check_estimators_fit_returns_self",
            "check_estimator_get_tags_default_keys",
            "check_classification_train",
            "check_estimators_unfitted",
            "check_set_params",
            "check_dont_overwrite_parameters",
            "check_n_features_in"
        ]:
            print(t)
            try:
                t[1](t[0])
            except SkipTest:
                pass


def test_skl_regr():
    config = {
        "pipeline": DummyTestPipeline,
        "extra_pipeline_options": {
            "learner": PlainLearner,
            "learner_kwargs": {"model_class": FalconR},
        },
    }

    _test_regr(FalconRegressor(config=config, eval_strategy='auto'))
    _test_regr(FalconTabularRegressor(config=config, eval_strategy='auto'))


def test_skl_clf():
    config = {
        "pipeline": DummyTestPipeline,
        "extra_pipeline_options": {
            "learner": PlainLearner,
            "learner_kwargs": {"model_class": FalconC},
        },
    }
    _test_clf(FalconClassifier(config=config, eval_strategy='auto'))
    _test_clf(FalconTabularClassifier(config=config, eval_strategy='auto'))
