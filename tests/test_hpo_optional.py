from typing import Any

import numpy as np
import pytest

from falcon.config import HPOSource
from falcon.tabular import hpo


def test_hpo_source_without_optuna_has_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(module_name: str) -> Any:
        if module_name == "optuna":
            raise ModuleNotFoundError("No module named 'optuna'", name="optuna")
        raise AssertionError(f"Unexpected import: {module_name}")

    monkeypatch.setattr(hpo, "import_module", unavailable)

    with pytest.raises(ImportError, match=r"pip install falcon-ml\[hpo\]"):
        HPOSource(family="linear", n_trials=1).get_candidates(
            "tabular_regression",
            X=np.arange(16, dtype=np.float32).reshape(8, 2),
            y=np.arange(8, dtype=np.float32),
            groups=np.arange(8),
        )
