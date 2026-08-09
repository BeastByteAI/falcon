import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import falcon

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _matrix_entries() -> list[dict[str, object]]:
    payload = json.loads(
        (REPOSITORY_ROOT / "ci/matrix.json").read_text(encoding="utf-8")
    )
    return payload["include"]


def test_ci_runs_supported_python_versions_with_runtime() -> None:
    entries = _matrix_entries()
    locked = {
        entry["python"]
        for entry in entries
        if entry["resolution"] == "locked" and entry.get("extras") == ["runtime"]
    }

    assert locked == {"3.10", "3.11", "3.12", "3.13"}
    assert all(
        entry["resolution"] in {"locked", "lowest-direct", "highest"}
        for entry in entries
    )


def test_ci_matrix_covers_the_dependency_floor_and_ceiling() -> None:
    """Both bounds have to be exercised, not just the pinned middle.

    Every job used to run `uv sync --locked`, so the whole matrix tested one dependency
    set. That is how a protobuf release broke installs while CI stayed green.
    """
    resolutions = {entry["name"]: entry["resolution"] for entry in _matrix_entries()}

    assert "lowest-direct" in resolutions.values()
    assert "highest" in resolutions.values()

    ceiling = next(
        entry for entry in _matrix_entries() if entry["resolution"] == "highest"
    )
    assert ceiling.get("allow_failure") is True


def test_ci_workflow_runs_the_matrix_through_the_local_entry_point() -> None:
    """CI and `python scripts/run_matrix.py` must not be able to drift apart."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/tests.yml").read_text(
        encoding="utf-8"
    )

    assert "jq -c '.include' ci/matrix.json" in workflow
    assert "python scripts/run_matrix.py ${{ matrix.name }}" in workflow
    assert (REPOSITORY_ROOT / "scripts/run_matrix.py").is_file()


def test_runtime_extra_pins_string_op_compatible_onnxruntime() -> None:
    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    requirements = configuration["project"]["optional-dependencies"]["runtime"]
    assert any(
        requirement.startswith("onnxruntime>=1.18.1") for requirement in requirements
    )


def test_gbdt_extra_and_ci_parity_job_are_configured() -> None:
    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = configuration["project"]["optional-dependencies"]["gbdt"]
    dependency_names = {
        requirement.split(">", maxsplit=1)[0].split("=", maxsplit=1)[0]
        for requirement in requirements
    }

    assert dependency_names == {"lightgbm", "xgboost", "catboost", "onnxmltools"}
    assert any(
        entry.get("extras") == ["runtime", "gbdt"] for entry in _matrix_entries()
    )


def test_hpo_extra_contains_optuna_and_tqdm() -> None:
    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = configuration["project"]["optional-dependencies"]["hpo"]
    dependency_names = {
        requirement.split(">", maxsplit=1)[0].split("=", maxsplit=1)[0]
        for requirement in requirements
    }

    assert dependency_names == {"optuna", "tqdm"}
    assert any(entry.get("extras") == ["runtime", "hpo"] for entry in _matrix_entries())


def test_release_metadata_and_slimmed_core_dependencies() -> None:
    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependency_names = {
        requirement.split(">", maxsplit=1)[0]
        .split("<", maxsplit=1)[0]
        .split("=", maxsplit=1)[0]
        for requirement in configuration["project"]["dependencies"]
    }

    assert configuration["project"]["version"] == "1.0.0"
    assert falcon.__version__ == "1.0.0"
    assert dependency_names == {
        "numpy",
        "onnx",
        "pandas",
        "protobuf",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "skl2onnx",
    }
    assert set(configuration["project"]["optional-dependencies"]) == {
        "gbdt",
        "hpo",
        "runtime",
    }


def test_the_lockfile_records_the_released_version() -> None:
    """A stale lock breaks every `uv sync --locked` CI leg, not just packaging.

    `uv.lock` pins the project's own version, so bumping `pyproject.toml` without
    re-running `uv lock` fails the install stage before a single test runs.
    """
    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"))

    locked = next(
        package for package in lock["package"] if package["name"] == "falcon-ml"
    )

    assert locked["version"] == configuration["project"]["version"]


def test_protobuf_is_capped_below_the_bool_attribute_rejection() -> None:
    """Guards the pin that keeps tree export working.

    skl2onnx emits Python bools into `nodes_missing_value_tracks_true`, an int64
    attribute. protobuf 7.34 turned that from a warning into a `TypeError`, which fails
    every tree model export, so the cap has to hold until skl2onnx stops doing it.
    """
    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirement = next(
        entry
        for entry in configuration["project"]["dependencies"]
        if entry.startswith("protobuf")
    )
    assert requirement == "protobuf>=4.25.1,<7.34"

    from google.protobuf import __version__ as protobuf_version

    major, minor = (int(part) for part in protobuf_version.split(".")[:2])
    assert (major, minor) < (7, 34)


def test_core_skl2onnx_version_imports_without_optuna_transitive_dependencies() -> None:
    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirement = next(
        requirement
        for requirement in configuration["project"]["dependencies"]
        if requirement.startswith("skl2onnx")
    )

    assert requirement.startswith("skl2onnx>=1.20.0")

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys


class PackagingImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "packaging" or fullname.startswith("packaging."):
            raise ModuleNotFoundError("packaging is unavailable")
        return None


sys.meta_path.insert(0, PackagingImportBlocker())
import skl2onnx
""",
        ],
        check=True,
    )


def test_benchmarks_are_excluded_from_distribution_and_ci() -> None:
    configuration = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_finder = configuration["tool"]["setuptools"]["packages"]["find"]

    assert package_finder["include"] == ["falcon*"]
    assert "benchmarks*" in package_finder["exclude"]

    workflow = (REPOSITORY_ROOT / ".github/workflows/tests.yml").read_text(
        encoding="utf-8"
    )
    assert "benchmarks/run.py" not in workflow


def test_local_artifacts_are_untracked_and_ignored() -> None:
    tracked_result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    tracked_paths = [
        PurePosixPath(raw_path) for raw_path in tracked_result.stdout.splitlines()
    ]
    tracked_artifacts = {
        path
        for path in tracked_paths
        if (
            path.suffix == ".fnnx"
            or path.name.startswith("tmp.")
            or any(part in {"build", "dist"} for part in path.parts[:-1])
        )
    }

    assert not tracked_artifacts

    artifacts = ("build/package", "dist/package", "tmp.cache", "model.fnnx")
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=REPOSITORY_ROOT,
        input="\n".join(artifacts),
        capture_output=True,
        check=True,
        text=True,
    )

    assert set(result.stdout.splitlines()) == set(artifacts)
