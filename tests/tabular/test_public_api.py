from importlib import import_module

import pytest

from falcon import tabular


def test_time_series_adapter_is_removed() -> None:
    assert not hasattr(tabular, "TSAdapter")

    with pytest.raises(ModuleNotFoundError):
        import_module("falcon.tabular.adapters.ts.adapter")
