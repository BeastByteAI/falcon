from falcon import initialize
from falcon.tabular.tabular_manager import TabularTaskManager
import pytest


def test_initialize():
    manager1 = initialize(
        data="tests/extra_files/iris.csv", task="tabular_classification"
    )
    manager2 = initialize(
        data="tests/extra_files/prices.csv", task="tabular_regression"
    )
    assert isinstance(manager1, TabularTaskManager)
    assert isinstance(manager2, TabularTaskManager)
    with pytest.raises(ValueError, match="Unknown task"):
        manager1 = initialize(data="tests/extra_files/iris.csv", task="unknown_task")
