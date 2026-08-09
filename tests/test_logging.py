import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from falcon.utils import logger, set_verbosity_level

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_verbosity_controls_falcon_logger_without_environment_state(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("FALCON_VERBOSITY_LEVEL", raising=False)
    original_level = logger.level

    try:
        set_verbosity_level(1)
        with caplog.at_level(logging.INFO, logger="falcon"):
            logger.info("falcon-visible-info")

        assert logger.level == logging.INFO
        assert "falcon-visible-info" in caplog.text
        assert "FALCON_VERBOSITY_LEVEL" not in os.environ

        set_verbosity_level(0)
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(original_level)


def test_import_does_not_suppress_python_warnings() -> None:
    env = os.environ.copy()
    env.pop("PYTHONWARNINGS", None)
    code = (
        "import os, warnings; import falcon; "
        "warnings.warn('falcon-visible-warning'); "
        "print(os.environ.get('PYTHONWARNINGS', ''))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "falcon-visible-warning" in result.stderr
    assert result.stdout.strip() == ""
