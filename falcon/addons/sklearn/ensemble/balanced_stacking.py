# type: ignore
from imblearn.over_sampling import RandomOverSampler
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np
from joblib import Parallel
import scipy.sparse as sparse

from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_scalar
from sklearn.utils.fixes import delayed
from sklearn.ensemble import StackingClassifier
from types import MethodType
from falcon.addons.sklearn.model_selection.balanced_strat_kfold import (
    BalancedStratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn import __version__ as sklearn_version
from packaging import version

# Slightly modified version of StackingClassifier from sklearn that upsamples the minority class during training
# The .fit() method was adopted from https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/ensemble/_stacking.py


def _fit(self, X, y, sample_weight=None):
    check_classification_targets(y)
    self._le = LabelEncoder().fit(y)
    self.classes_ = self._le.classes_
    check_scalar(
        self.passthrough,
        name="passthrough",
        target_type=(np.bool_, bool),
        include_boundaries="neither",
    )
    # all_estimators contains all estimators, the one to be fitted and the
    # 'drop' string.
    names, all_estimators = self._validate_estimators()
    self._validate_final_estimator()

    stack_method = [self.stack_method] * len(all_estimators)

    if self.cv == "prefit":
        self.estimators_ = []
        for estimator in all_estimators:
            if estimator != "drop":
                check_is_fitted(estimator)
                self.estimators_.append(estimator)
    else:
        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        X_resampled, y_resampled = RandomOverSampler().fit_resample(X, y)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone(est), X_resampled, y_resampled, sample_weight
            )
            for est in all_estimators
            if est != "drop"
        )

    self.named_estimators_ = Bunch()
    est_fitted_idx = 0
    for name_est, org_est in zip(names, all_estimators):
        if org_est != "drop":
            current_estimator = self.estimators_[est_fitted_idx]
            self.named_estimators_[name_est] = current_estimator
            est_fitted_idx += 1
            if hasattr(current_estimator, "feature_names_in_"):
                self.feature_names_in_ = current_estimator.feature_names_in_
        else:
            self.named_estimators_[name_est] = "drop"

    self.stack_method_ = [
        self._method_name(name, est, meth)
        for name, est, meth in zip(names, all_estimators, stack_method)
    ]

    if self.cv == "prefit":
        # Generate predictions from prefit models
        predictions = [
            getattr(estimator, predict_method)(X)
            for estimator, predict_method in zip(all_estimators, self.stack_method_)
            if estimator != "drop"
        ]
    else:
        # To train the meta-classifier using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.
        # To ensure that the data provided to each estimator are the same,
        # we need to set the random state of the cv if there is one and we
        # need to take a copy.

        if isinstance(self.cv, int):
            self.cv = BalancedStratifiedKFold(self.cv)

        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, "random_state") and cv.random_state is None:
            cv.random_state = np.random.RandomState()

        fit_params = (
            {"sample_weight": sample_weight} if sample_weight is not None else None
        )

        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(
                clone(est),
                X,
                y,
                cv=deepcopy(cv),
                method=meth,
                n_jobs=self.n_jobs,
                fit_params=fit_params,
                verbose=self.verbose,
            )
            for est, meth in zip(all_estimators, self.stack_method_)
            if est != "drop"
        )

    # Only not None or not 'drop' estimators will be used in transform.
    # Remove the None from the method as wl.
    self.stack_method_ = [
        meth for (meth, est) in zip(self.stack_method_, all_estimators) if est != "drop"
    ]

    X_meta = self._concatenate_predictions(X, predictions)
    X_meta_resampled, y_meta_resampled = RandomOverSampler().fit_resample(X_meta, y)
    _fit_single_estimator(
        self.final_estimator_,
        X_meta_resampled,
        y_meta_resampled,
        sample_weight=sample_weight,
    )

    return self

class _EncoderPlaceholder(LabelEncoder):
    
    def fit(self, y, **args):
        return self
    
    def transform(self, y, **args):
        return y
    
    def fit_transform(self, y, **args):
        return y
    
    def inverse_transform(self, y, **args):
        return y



# the object is being patched with a new method instead of subclassing
# so the estimator can be converted to ONNX using the default converter
def BalancedStackingClassifier(estimators, final_estimator, **kwargs):
    clf = StackingClassifier(estimators, final_estimator, **kwargs)
    clf.fit = MethodType(_fit, clf)
    if version.parse(sklearn_version) >= version.parse("1.2.0"):
        clf._label_encoder = _EncoderPlaceholder()
    return clf
