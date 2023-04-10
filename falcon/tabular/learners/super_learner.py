from typing import Callable
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.svm import NuSVR, SVR
from falcon.abstract import Learner
from falcon.abstract.onnx_convertible import ONNXConvertible
from falcon.types import Float32Array, Int64Array
from typing import Dict, List, Tuple, Callable, Optional, List, Type, Any, Union

from imblearn.over_sampling import RandomOverSampler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    # QuadraticDiscriminantAnalysis, # DO NOT USE
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR, NuSVC, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, balanced_accuracy_score

from falcon.tabular.models import StackingRegressor, StackingClassifier
from falcon.utils import print_
import pandas as pd

import numpy as np
from numpy import typing as npt
from falcon.serialization import SerializedModelRepr

_SKLEARN_VERBOSE = 0  # for debugging only

_default_estimators: Dict = {
    "tabular_regression": {
        "mini": [
            ("LinearRegression", LinearRegression, {}),
            ("ElasticNet", ElasticNet, {}),
            ("SVR", SVR, {}),
            ("NuSVR", NuSVR, {}),
            ("DecisionTreeRegressor", DecisionTreeRegressor, {}),
            (
                "HistGradientBoostingRegressor",
                HistGradientBoostingRegressor,
                {"min_samples_leaf": 2},
            ),
            ("GradientBoostingRegressor_100", GradientBoostingRegressor, {}),
            ("AdaBoostRegressor_50", AdaBoostRegressor, {}),
            (
                "BaggingRegressor_10",
                BaggingRegressor,
                {
                    "base_estimator": DecisionTreeRegressor(min_samples_split=2),
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_100",
                RandomForestRegressor,
                {"min_samples_split": 2, "n_jobs": -1, "verbose": _SKLEARN_VERBOSE},
            ),
            (
                "ExtraTreesRegressor_100",
                ExtraTreesRegressor,
                {"min_samples_split": 2, "n_jobs": -1, "verbose": _SKLEARN_VERBOSE},
            ),
            ("SVR_linear", SVR, {"C": 0.025, "kernel": "linear"}),
            (
                "GradientBoostingRegressor_10",
                GradientBoostingRegressor,
                {"n_estimators": 10},
            ),
            (
                "GradientBoostingRegressor_25",
                GradientBoostingRegressor,
                {"n_estimators": 25},
            ),
            (
                "GradientBoostingRegressor_50",
                GradientBoostingRegressor,
                {"n_estimators": 50},
            ),
            (
                "GradientBoostingRegressor_200",
                GradientBoostingRegressor,
                {"n_estimators": 200},
            ),
            ("AdaBoostRegressor_10", AdaBoostRegressor, {"n_estimators": 10}),
            ("AdaBoostRegressor_25", AdaBoostRegressor, {"n_estimators": 25}),
            ("AdaBoostRegressor_100", AdaBoostRegressor, {"n_estimators": 100}),
            (
                "BaggingRegressor_25",
                BaggingRegressor,
                {
                    "n_estimators": 25,
                    "base_estimator": DecisionTreeRegressor(min_samples_split=2),
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingRegressor_50",
                BaggingRegressor,
                {
                    "n_estimators": 50,
                    "base_estimator": DecisionTreeRegressor(min_samples_split=2),
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingRegressor_100",
                BaggingRegressor,
                {
                    "n_estimators": 100,
                    "base_estimator": DecisionTreeRegressor(min_samples_split=2),
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_10",
                RandomForestRegressor,
                {
                    "n_estimators": 10,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_25",
                RandomForestRegressor,
                {
                    "n_estimators": 25,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_50",
                RandomForestRegressor,
                {
                    "n_estimators": 50,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_200",
                RandomForestRegressor,
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_10",
                ExtraTreesRegressor,
                {
                    "n_estimators": 10,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_25",
                ExtraTreesRegressor,
                {
                    "n_estimators": 25,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_50",
                ExtraTreesRegressor,
                {
                    "n_estimators": 50,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_200",
                ExtraTreesRegressor,
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
        ],
        "mid": [
            ("ElasticNet", ElasticNet, {}),
            (
                "DecisionTreeRegressor",
                DecisionTreeRegressor,
                {"min_samples_split": 0.001},
            ),
            (
                "HistGradientBoostingRegressor_150",
                HistGradientBoostingRegressor,
                {"max_iter": 150},
            ),
            ("HistGradientBoostingRegressor", HistGradientBoostingRegressor, {}),
            (
                "HistGradientBoostingRegressor_50",
                HistGradientBoostingRegressor,
                {"max_iter": 50},
            ),
            (
                "GradientBoostingRegressor_100",
                GradientBoostingRegressor,
                {"min_samples_split": 0.003},
            ),
            ("AdaBoostRegressor_50", AdaBoostRegressor, {}),
            (
                "BaggingRegressor_10",
                BaggingRegressor,
                {
                    "base_estimator": DecisionTreeRegressor(min_samples_split=0.001),
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_100",
                RandomForestRegressor,
                {"min_samples_split": 0.003, "n_jobs": 4, "verbose": _SKLEARN_VERBOSE},
            ),
            (
                "ExtraTreesRegressor_100",
                ExtraTreesRegressor,
                {"min_samples_split": 0.003, "n_jobs": 4, "verbose": _SKLEARN_VERBOSE},
            ),
            (
                "GradientBoostingRegressor_10",
                GradientBoostingRegressor,
                {"n_estimators": 10, "min_samples_split": 0.003},
            ),
            (
                "GradientBoostingRegressor_25",
                GradientBoostingRegressor,
                {"n_estimators": 25, "min_samples_split": 0.003},
            ),
            (
                "GradientBoostingRegressor_50",
                GradientBoostingRegressor,
                {"n_estimators": 50, "min_samples_split": 0.003},
            ),
            ("AdaBoostRegressor_25", AdaBoostRegressor, {"n_estimators": 25}),
            ("AdaBoostRegressor_100", AdaBoostRegressor, {"n_estimators": 100}),
            (
                "BaggingRegressor_25",
                BaggingRegressor,
                {
                    "n_estimators": 25,
                    "base_estimator": DecisionTreeRegressor(min_samples_split=0.003),
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingRegressor_50",
                BaggingRegressor,
                {
                    "n_estimators": 50,
                    "base_estimator": DecisionTreeRegressor(min_samples_split=0.003),
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingRegressor_100",
                BaggingRegressor,
                {
                    "n_estimators": 100,
                    "base_estimator": DecisionTreeRegressor(min_samples_split=0.003),
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_10",
                RandomForestRegressor,
                {
                    "n_estimators": 10,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_25",
                RandomForestRegressor,
                {
                    "n_estimators": 25,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_50",
                RandomForestRegressor,
                {
                    "n_estimators": 50,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_10",
                ExtraTreesRegressor,
                {
                    "n_estimators": 10,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_25",
                ExtraTreesRegressor,
                {
                    "n_estimators": 25,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_50",
                ExtraTreesRegressor,
                {
                    "n_estimators": 50,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
        ],
        "large": [
            ("ElasticNet", ElasticNet, {}),
            ("HistGradientBoostingRegressor", HistGradientBoostingRegressor, {}),
            (
                "HistGradientBoostingRegressor_200",
                HistGradientBoostingRegressor,
                {"max_iter": 200},
            ),
            (
                "BaggingRegressor_10",
                BaggingRegressor,
                {
                    "base_estimator": DecisionTreeRegressor(min_samples_split=0.001),
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_100",
                RandomForestRegressor,
                {"min_samples_split": 0.003, "n_jobs": 1, "verbose": _SKLEARN_VERBOSE},
            ),
            (
                "ExtraTreesRegressor_100",
                ExtraTreesRegressor,
                {"min_samples_split": 0.003, "n_jobs": 2, "verbose": _SKLEARN_VERBOSE},
            ),
            (
                "GradientBoostingRegressor_25",
                GradientBoostingRegressor,
                {"n_estimators": 25},
            ),
            ("AdaBoostRegressor_10", AdaBoostRegressor, {"n_estimators": 10}),
            ("AdaBoostRegressor_25", AdaBoostRegressor, {"n_estimators": 25}),
            (
                "BaggingRegressor_25",
                BaggingRegressor,
                {
                    "n_estimators": 25,
                    "base_estimator": DecisionTreeRegressor(min_samples_split=0.001),
                    "n_jobs": 2,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_10",
                RandomForestRegressor,
                {
                    "n_estimators": 10,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_25",
                RandomForestRegressor,
                {
                    "n_estimators": 25,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestRegressor_50",
                RandomForestRegressor,
                {
                    "n_estimators": 50,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_10",
                ExtraTreesRegressor,
                {
                    "n_estimators": 10,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_25",
                ExtraTreesRegressor,
                {
                    "n_estimators": 25,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesRegressor_50",
                ExtraTreesRegressor,
                {
                    "n_estimators": 50,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
        ],
        "x-large": [
            ("HistGradientBoostingRegressor_100", HistGradientBoostingRegressor, {}),
            (
                "HistGradientBoostingRegressor_200",
                HistGradientBoostingRegressor,
                {"max_iter": 200},
            ),
            (
                "HistGradientBoostingRegressor_200",
                HistGradientBoostingRegressor,
                {"max_iter": 50},
            ),
            (
                "RandomForestRegressor_100",
                RandomForestRegressor,
                {"min_samples_split": 0.003, "n_jobs": 1, "verbose": _SKLEARN_VERBOSE},
            ),
        ],
    },
    "tabular_classification": {
        "mini": [
            ("LogisticRegression", LogisticRegression, {}),
            ("DecisionTreeClassifier", DecisionTreeClassifier, {}),
            ("SVC", SVC, {}),
            ("NuSVC", NuSVC, {}),
            ("GaussianNB", GaussianNB, {}),
            ("AdaBoostClassifier_50", AdaBoostClassifier, {}),
            ("GradientBoostingClassifier_100", GradientBoostingClassifier, {}),
            (
                "HistGradientBoostingClassifier",
                HistGradientBoostingClassifier,
                {"min_samples_leaf": 2},
            ),
            (
                "RandomForestClassifier_100",
                RandomForestClassifier,
                {"min_samples_split": 0.003, "n_jobs": -1, "verbose": _SKLEARN_VERBOSE},
            ),
            (
                "BaggingClassifier_10",
                BaggingClassifier,
                {
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_100",
                ExtraTreesClassifier,
                {"min_samples_split": 0.003, "n_jobs": -1, "verbose": _SKLEARN_VERBOSE},
            ),
            ("SVC_linear", SVC, {"C": 0.025, "kernel": "linear"}),
            ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis, {}),
            # ("QuadraticDiscriminantAnalysis", QuadraticDiscriminantAnalysis, {}), # this breaks onnx
            ("AdaBoostClassifier_10", AdaBoostClassifier, {"n_estimators": 10}),
            ("AdaBoostClassifier_25", AdaBoostClassifier, {"n_estimators": 25}),
            ("AdaBoostClassifier_100", AdaBoostClassifier, {"n_estimators": 100}),
            (
                "GradientBoostingClassifier_10",
                GradientBoostingClassifier,
                {"n_estimators": 10},
            ),
            (
                "GradientBoostingClassifier_25",
                GradientBoostingClassifier,
                {"n_estimators": 25},
            ),
            (
                "GradientBoostingClassifier_50",
                GradientBoostingClassifier,
                {"n_estimators": 50},
            ),
            (
                "GradientBoostingClassifier_200",
                GradientBoostingClassifier,
                {"n_estimators": 200},
            ),
            (
                "RandomForestClassifier_10",
                RandomForestClassifier,
                {
                    "n_estimators": 10,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestClassifier_25",
                RandomForestClassifier,
                {
                    "n_estimators": 25,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestClassifier_50",
                RandomForestClassifier,
                {
                    "n_estimators": 50,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestClassifier_200",
                RandomForestClassifier,
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingClassifier_25",
                BaggingClassifier,
                {
                    "n_estimators": 25,
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingClassifier_50",
                BaggingClassifier,
                {
                    "n_estimators": 50,
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingClassifier_100",
                BaggingClassifier,
                {
                    "n_estimators": 100,
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_10",
                ExtraTreesClassifier,
                {
                    "n_estimators": 10,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_25",
                ExtraTreesClassifier,
                {
                    "n_estimators": 25,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_50",
                ExtraTreesClassifier,
                {
                    "n_estimators": 50,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_200",
                ExtraTreesClassifier,
                {
                    "n_estimators": 200,
                    "min_samples_split": 2,
                    "n_jobs": -1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
        ],
        "mid": [
            ("LogisticRegression", LogisticRegression, {"max_iter": 150}),
            (
                "DecisionTreeClassifier",
                DecisionTreeClassifier,
                {"min_samples_split": 0.001},
            ),
            ("GaussianNB", GaussianNB, {}),
            ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis, {}),
            ("AdaBoostClassifier_50", AdaBoostClassifier, {}),
            (
                "GradientBoostingClassifier_100",
                GradientBoostingClassifier,
                {"min_samples_split": 0.003},
            ),
            (
                "HistGradientBoostingClassifier_50",
                HistGradientBoostingClassifier,
                {"max_iter": 50},
            ),
            ("HistGradientBoostingClassifier", HistGradientBoostingClassifier, {}),
            (
                "HistGradientBoostingClassifier_150",
                HistGradientBoostingClassifier,
                {"max_iter": 150},
            ),
            (
                "RandomForestClassifier_100",
                RandomForestClassifier,
                {"min_samples_split": 0.003, "n_jobs": 4, "verbose": _SKLEARN_VERBOSE},
            ),
            (
                "BaggingClassifier_10",
                BaggingClassifier,
                {
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_100",
                ExtraTreesClassifier,
                {"min_samples_split": 0.003, "n_jobs": 4, "verbose": _SKLEARN_VERBOSE},
            ),
            ("AdaBoostClassifier_25", AdaBoostClassifier, {"n_estimators": 25}),
            ("AdaBoostClassifier_100", AdaBoostClassifier, {"n_estimators": 100}),
            (
                "GradientBoostingClassifier_10",
                GradientBoostingClassifier,
                {"n_estimators": 10, "min_samples_split": 0.003},
            ),
            (
                "GradientBoostingClassifier_25",
                GradientBoostingClassifier,
                {"n_estimators": 25, "min_samples_split": 0.003},
            ),
            (
                "GradientBoostingClassifier_50",
                GradientBoostingClassifier,
                {"n_estimators": 50, "min_samples_split": 0.003},
            ),
            (
                "RandomForestClassifier_10",
                RandomForestClassifier,
                {
                    "n_estimators": 10,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestClassifier_25",
                RandomForestClassifier,
                {
                    "n_estimators": 25,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestClassifier_50",
                RandomForestClassifier,
                {
                    "n_estimators": 50,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingClassifier_25",
                BaggingClassifier,
                {
                    "n_estimators": 25,
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingClassifier_50",
                BaggingClassifier,
                {
                    "n_estimators": 50,
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingClassifier_100",
                BaggingClassifier,
                {
                    "n_estimators": 100,
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_10",
                ExtraTreesClassifier,
                {
                    "n_estimators": 10,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_25",
                ExtraTreesClassifier,
                {
                    "n_estimators": 25,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_50",
                ExtraTreesClassifier,
                {
                    "n_estimators": 50,
                    "min_samples_split": 0.003,
                    "n_jobs": 4,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
        ],
        "large": [
            ("LogisticRegression", LogisticRegression, {"max_iter": 250}),
            ("GaussianNB", GaussianNB, {}),
            ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis, {}),
            ("HistGradientBoostingClassifier", HistGradientBoostingClassifier, {}),
            (
                "HistGradientBoostingClassifier_200",
                HistGradientBoostingClassifier,
                {"max_iter": 200},
            ),
            (
                "HistGradientBoostingClassifier_50",
                HistGradientBoostingClassifier,
                {"max_iter": 50},
            ),
            (
                "RandomForestClassifier_100",
                RandomForestClassifier,
                {"min_samples_split": 0.003, "n_jobs": 1, "verbose": _SKLEARN_VERBOSE},
            ),
            (
                "BaggingClassifier_10",
                BaggingClassifier,
                {
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_100",
                ExtraTreesClassifier,
                {"min_samples_split": 0.003, "n_jobs": 1, "verbose": _SKLEARN_VERBOSE},
            ),
            ("AdaBoostClassifier_200", AdaBoostClassifier, {"n_estimators": 200}),
            (
                "RandomForestClassifier_10",
                RandomForestClassifier,
                {
                    "n_estimators": 10,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestClassifier_25",
                RandomForestClassifier,
                {
                    "n_estimators": 25,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "RandomForestClassifier_50",
                RandomForestClassifier,
                {
                    "n_estimators": 50,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "BaggingClassifier_25",
                BaggingClassifier,
                {
                    "n_estimators": 25,
                    "base_estimator": DecisionTreeClassifier(min_samples_split=0.001),
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_10",
                ExtraTreesClassifier,
                {
                    "n_estimators": 10,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_25",
                ExtraTreesClassifier,
                {
                    "n_estimators": 25,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
            (
                "ExtraTreesClassifier_50",
                ExtraTreesClassifier,
                {
                    "n_estimators": 50,
                    "min_samples_split": 0.003,
                    "n_jobs": 1,
                    "verbose": _SKLEARN_VERBOSE,
                },
            ),
        ],
        "x-large": [
            ("HistGradientBoostingClassifier_100", HistGradientBoostingClassifier, {}),
            (
                "HistGradientBoostingClassifier_200",
                HistGradientBoostingClassifier,
                {"max_iter": 200},
            ),
            (
                "HistGradientBoostingClassifier_50",
                HistGradientBoostingClassifier,
                {"max_iter": 50},
            ),
            (
                "RandomForestClassifier_100",
                RandomForestClassifier,
                {"min_samples_split": 0.003, "n_jobs": 1, "verbose": _SKLEARN_VERBOSE},
            ),
        ],
    },
}


class SuperLearner(Learner, ONNXConvertible):
    """
    Tabular learner which employs StackingModel for construction of meta estimator.
    """

    def __init__(
        self,
        task: str,
        base_estimators: Optional[List[Tuple[str, Callable, Dict]]] = None,
        base_score_threshold: Optional[float] = None,
        cv: Any = None,
        filter_estimators: Optional[bool] = None,
        dataset_size: Optional[Tuple[int, ...]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Constructs a meta model which is trained on cross-validated predictions of base estimators.

        Parameters
        ----------
        task : str
            `tabular_classification` or `tabular_regression`
        base_estimators : Optional[List[Tuple[str, Callable, Dict]]], optional
            list of base estimators, by default None
        base_score_threshold : Optional[float], optional
            threshold for filtering of the estimators, by default None
        cv : Any, optional
            number of CV folds or CV custom object, by default None
        filter_estimators : Optional[bool], optional
            when True, the perfomance of the estimators pre-estimated on the subset of training, estimators with the performance below the threshold will not be used for meta model construction, by default None
        dataset_size : Optional[Tuple[int]], optional
            size of the dataset, by default None
        """

        if task not in ["tabular_classification", "tabular_regression"]:
            raise ValueError(
                f"Invalid task type. Expected `tabular_classification` or `tabular_regression`, found `{task}`."
            )

        self.base_estimators = base_estimators
        self.dataset_size: Optional[Tuple[int, ...]] = dataset_size
        self.task = task
        self.base_score_threshold = base_score_threshold
        self.cv = cv
        self.filter_estimators = filter_estimators

    def _split(
        self, X: npt.NDArray, y: npt.NDArray
    ) -> Tuple[Float32Array, Float32Array, Float32Array, Float32Array]:
        X_train: Float32Array
        y_train: Float32Array
        X_val: Float32Array
        y_val: Float32Array

        if self.task == "tabular_classification":
            X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y)
            X_train_upsampled: Float32Array
            y_train_upsampled: Float32Array
            X_train_upsampled, y_train_upsampled = RandomOverSampler().fit_resample(
                X_train, y_train
            )
            return X_train_upsampled, X_val, y_train_upsampled, y_val
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=None)
        return X_train, X_val, y_train, y_val

    def _calculate_base_score(self, y_hat: Float32Array, y: Float32Array) -> float:
        if self.task == "tabular_classification":
            return balanced_accuracy_score(y, y_hat)
        else:
            return (r2_score(y, y_hat) + 1) / 2

    def _set_size_optimized_config(self, X: Float32Array) -> None:
        if self.dataset_size is None:
            self.dataset_size = X.shape
        volume = self.dataset_size[0] * self.dataset_size[1]

        min_threshold = 80_000  # 5_000 samples with 16 features
        mid_threshold = 4_000_000  # 125_000 samples with 32 featrues / 250_000 samples with 16 features
        large_threshold = 16_000_000  # 1_000_000 samples with 16 features

        if volume < min_threshold:
            print_("Setting up learner config [small dataset]")
            cv = 10
            base_estimators = _default_estimators[self.task]["mini"]
            filter_estimators = True
        elif volume < mid_threshold:
            print_("Setting up learner config [mid dataset]")
            cv = 5
            base_estimators = _default_estimators[self.task]["mid"]
            filter_estimators = True
        elif volume < large_threshold:
            print_("Setting up learner config [large dataset]")
            base_estimators = _default_estimators[self.task]["large"]
            cv = 3
            filter_estimators = False
        else:
            print_("Setting up learner config [x-large dataset]")
            base_estimators = _default_estimators[self.task]["x-large"]
            cv = 3
            filter_estimators = False

        if self.cv is None:
            self.cv = cv
        if self.base_estimators is None:
            self.base_estimators = base_estimators
        if self.filter_estimators is None:
            self.filter_estimators = filter_estimators

    def _preselect(
        self, X: Float32Array, y: Float32Array
    ) -> List[
        Tuple[str, Callable]
    ]:  # select estimators to be used in the main training loop
        if self.base_estimators is None:
            raise ValueError("expected base_estimators to be a list, found None")
        selected_estimators: List[Tuple[str, Callable]] = []
        if not self.filter_estimators:
            print(
                "\t -> Skipping filtering of base classifiers => all estimators will be used for final model"
            )
            selected_estimators = [
                (estimator[0], estimator[1](**estimator[2]))
                for estimator in self.base_estimators
            ]
            return selected_estimators
        print_(f"\t -> Filtering base classifiers:")
        if self.base_score_threshold is None:
            if self.task == "tabular_classification":
                n_classes = len(np.unique(y, return_counts=False))
                baseline = 1 / n_classes
                self.base_score_threshold = 1.1 * baseline
                print_(
                    f"\t Using {self.base_score_threshold} as baseline score for {n_classes} classes classification task"
                )
            else:
                self.base_score_threshold = 0.55
        X_train: Float32Array
        y_train: Float32Array
        X_val: Float32Array
        y_val: Float32Array
        X_train, X_val, y_train, y_val = self._split(X, y)
        for estimator in self.base_estimators:

            est: SklearnBaseEstimator = estimator[1](**estimator[2])
            est.fit(X_train, y_train)
            y_hat: Float32Array = est.predict(X_val)
            base_score: float = self._calculate_base_score(y_hat, y_val)
            print_(f"\t\t-> {estimator[0]} base score: {base_score}")
            if base_score >= self.base_score_threshold:
                selected_estimators.append((estimator[0], estimator[1](**estimator[2])))
        if len(selected_estimators) < 3:  # using all estimators
            for estimator in self.base_estimators:
                selected_estimators = []
                selected_estimators.append((estimator[0], estimator[1](**estimator[2])))
        return selected_estimators

    def fit(self, X: Float32Array, y: Float32Array, *args: Any, **kwargs: Any) -> None:
        """
        Fits the model. The hyperparameters that were not passed to the `__init__` will be automatically determined based on the size of the training set.
        For classification tasks, the dataset will be balanced by upsampling the minority class(es).

        Parameters
        ----------
        X : Float32Array
            features
        y : Float32Array
            targets
        """
        print_("Fitting stacked model... ")
        self._set_size_optimized_config(X)
        estimators: List[Tuple[str, Callable]] = self._preselect(X, y)
        print_(f"\t -> Fitting the final estimator")
        stacked_estimator: Union[StackingClassifier, StackingRegressor]
        if self.task == "tabular_classification":
            stacked_estimator = StackingClassifier(
                estimators=estimators, final_estimator=LogisticRegression(), cv=self.cv
            )
        else:
            stacked_estimator = StackingRegressor(
                estimators=estimators, final_estimator=LinearRegression(), cv=self.cv
            )

        stacked_estimator.fit(X, y)
        self.model = stacked_estimator

    def predict(
        self, X: Float32Array, *args: Any, **kwargs: Any
    ) -> Union[Float32Array, Int64Array]:
        """
        Makes a prediction for given X.

        Parameters
        ----------
        X : Float32Array
            features

        Returns
        -------
        Union[Float32Array, Int64Array]
            predictions
        """
        return self.model.predict(X)

    def get_input_type(self) -> Type:
        """
        Returns
        -------
        Type
            Float32Array
        """
        return Float32Array

    def get_output_type(self) -> Type:
        """
        Returns
        -------
        Type
            Float32Array for regression, Int64Array for classification
        """
        return Float32Array if self.task == "tabular_regression" else Int64Array

    def forward(
        self, X: Float32Array, *args: Any, **kwargs: Any
    ) -> Union[Float32Array, Int64Array]:
        """
        Equivalen to `.predict(X)`

        Parameters
        ----------
        X : Float32Array
            features

        Returns
        -------
        Union[Float32Array, Int64Array]
            predictions
        """
        return self.model.predict(X)

    def fit_pipe(
        self, X: Float32Array, y: Float32Array, *args: Any, **kwargs: Any
    ) -> None:
        """
        Equivalent to `.fit(X, y)`

        Parameters
        ----------
        X : Float32Array
            features
        y : Float32Array
            targets
        """
        self.fit(X, y)

    def to_onnx(self) -> SerializedModelRepr:
        """
        Serializes the underlying model to onnx by calling its `.to_onnx()` method.

        Returns
        -------
        SerializedModelRepr
        """
        return self.model.to_onnx()
