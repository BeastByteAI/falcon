from falcon import initialize
import numpy as np
from sklearn.model_selection import KFold
import pytest

class _BrokenKFold(KFold):
    def split(self, *args, **kwargs): 
        raise ValueError("pytest :: Broken KFold")

def _broken_split(*args, **kwargs):
    raise ValueError("pytest :: Broken split")

def test_auto_eval_strategy():

    m = initialize(
        task="tabular_classification",
        eval_strategy="auto",
        data=(np.random.rand(250, 2), np.random.randint(0, 2, 250).reshape(-1, 1)),
        config = 'PlainLearner'
    )

    m.train()

    s = m.performance_summary(None)

    assert 'eval_cv' in s.keys()
    assert 'eval' not in s.keys()


    m = initialize(
        task="tabular_classification",
        eval_strategy="auto",
        data=(np.random.rand(2500, 2), np.random.randint(0, 2, 2500).reshape(-1, 1)),
        config = 'PlainLearner'
    )

    m.train()

    s = m.performance_summary(None)

    assert 'eval' in s.keys()
    assert 'eval_cv' not in s.keys()

def test_cv_eval_strategy():

    m = initialize(
        task="tabular_classification",
        eval_strategy="cv",
        data=(np.random.rand(100, 2), np.random.randint(0, 2, 100).reshape(-1, 1)),
        config = 'PlainLearner'
    )

    m.train()

    s = m.performance_summary(None)

    assert 'eval_cv' in s.keys()
    assert 'eval' not in s.keys()

def test_holdout_eval_strategy():

    m = initialize(
        task="tabular_classification",
        eval_strategy="holdout",
        data=(np.random.rand(100, 2), np.random.randint(0, 2, 100).reshape(-1, 1)),
        config = 'PlainLearner'
    )

    m.train()

    s = m.performance_summary(None)

    assert 'eval_cv' not in s.keys()
    assert 'eval' in s.keys()

def test_custom_cv_eval_strategy():

    cv = _BrokenKFold(n_splits=5, shuffle=True, random_state=42)

    m = initialize(
        task="tabular_classification",
        eval_strategy=cv,
        data=(np.random.rand(100, 2), np.random.randint(0, 2, 100).reshape(-1, 1)),
        config = 'PlainLearner'
    )


    with pytest.raises(ValueError, match="pytest :: Broken KFold"):
        m.train()

def test_custom_holdout_eval_strategy():
    m = initialize(
        task="tabular_classification",
        eval_strategy=_broken_split,
        data=(np.random.rand(100, 2), np.random.randint(0, 2, 100).reshape(-1, 1)),
        config = 'PlainLearner'
    )

    with pytest.raises(ValueError, match="pytest :: Broken split"):
        m.train()

def test_no_eval_strategy():
    m = initialize(
        task="tabular_classification",
        eval_strategy=None,
        data=(np.random.rand(100, 2), np.random.randint(0, 2, 100).reshape(-1, 1)),
        config = 'PlainLearner'
    )

    m.train()

    s = m.performance_summary(None)

    assert 'eval_cv' not in s.keys()
    assert 'eval' not in s.keys()
    assert 'train' in s.keys()
    assert len(s.keys()) == 1