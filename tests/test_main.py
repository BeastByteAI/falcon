import subprocess
import sys
from importlib import import_module

import pytest

import falcon
import falcon.tabular


def test_initialize_is_not_public_api() -> None:
    assert not hasattr(falcon, "initialize")


def test_public_import_does_not_load_the_legacy_training_path() -> None:
    code = (
        "import sys; import falcon; "
        "blocked = ('super_learner', 'optuna_learner', 'tabular_manager', "
        "'task_configurations', 'tabular.configurations', 'models.stacking', "
        "'abstract.optuna'); "
        "assert not any(any(part in name for part in blocked) for name in sys.modules)"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_tabular_task_manager_is_not_public_api() -> None:
    assert not hasattr(falcon.tabular, "TabularTaskManager")


@pytest.mark.parametrize(
    "module_name",
    [
        "falcon.abstract.optuna",
        "falcon.abstract.task_manager",
        "falcon.addons.sklearn.ensemble.balanced_stacking",
        "falcon.addons.sklearn.model_selection.balanced_strat_kfold",
        "falcon.tabular.configurations",
        "falcon.tabular.learners.optuna_learner",
        "falcon.tabular.learners.plain_learner",
        "falcon.tabular.learners.super_learner",
        "falcon.tabular.models.hist_gbt",
        "falcon.tabular.models.stacking",
        "falcon.tabular.reporting",
        "falcon.tabular.tabular_manager",
        "falcon.tabular.utils",
        "falcon.tabular.wrappers",
        "falcon.task_configurations",
    ],
)
def test_legacy_modules_are_removed(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        import_module(module_name)
