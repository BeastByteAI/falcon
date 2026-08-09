"""Run the test matrix defined in ci/matrix.json.

CI runs one entry per job through this same script, so a local run and a CI run cannot
drift apart. Each entry gets its own virtualenv outside the tree, leaving the project's
own `.venv` alone.

    python scripts/run_matrix.py                 # every entry
    python scripts/run_matrix.py latest-3.13     # one entry
    python scripts/run_matrix.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_FILE = REPOSITORY_ROOT / "ci" / "matrix.json"

# `locked` installs the lock file verbatim, the way a CI job and `uv sync` do. The other
# two re-resolve, and go through `uv pip install` rather than `uv sync` for one reason:
# `uv sync --resolution ...` rewrites uv.lock in place, so running the floor entry would
# silently replace the project's lock with a lowest-direct one.
RESOLUTIONS = frozenset({"locked", "lowest-direct", "highest"})


@dataclass(frozen=True)
class Entry:
    name: str
    python: str
    resolution: str
    extras: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    allow_failure: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.resolution not in RESOLUTIONS:
            raise SystemExit(
                f"Entry `{self.name}` has unknown resolution `{self.resolution}`; "
                f"expected one of {', '.join(sorted(RESOLUTIONS))}."
            )

    def install_commands(self, venv: Path) -> list[list[str]]:
        if self.resolution == "locked":
            command = ["uv", "sync", "--python", self.python, "--locked"]
            for extra in self.extras:
                command += ["--extra", extra]
            return [command]

        target = ["--python", str(venv / "bin" / "python")]
        specifier = f".[{','.join(self.extras)}]" if self.extras else "."
        return [
            ["uv", "venv", str(venv), "--python", self.python],
            [
                "uv",
                "pip",
                "install",
                *target,
                "--resolution",
                self.resolution,
                specifier,
            ],
            # The test tooling is installed separately and at its own newest version:
            # the resolution mode is there to exercise falcon's declared bounds, and
            # applying it to pytest just resolves an unusable twenty-year-old release.
            [
                "uv",
                "pip",
                "install",
                *target,
                "pytest",
                "tomli ; python_full_version < '3.11'",
            ],
        ]

    def pytest_command(self, venv: Path) -> list[str]:
        runner = str(venv / "bin" / "python")
        return [runner, "-m", "pytest", "-q", "-p", "no:cacheprovider", *self.tests]


@dataclass
class Result:
    entry: Entry
    stage: str = "ok"
    returncode: int = 0
    versions: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def load_entries() -> list[Entry]:
    payload = json.loads(MATRIX_FILE.read_text(encoding="utf-8"))
    return [
        Entry(
            name=item["name"],
            python=item["python"],
            resolution=item["resolution"],
            extras=tuple(item.get("extras", ())),
            tests=tuple(item.get("tests", ())),
            allow_failure=bool(item.get("allow_failure", False)),
            description=item.get("description", ""),
        )
        for item in payload["include"]
    ]


def installed_versions(venv: Path) -> str:
    probe = (
        "import sys, importlib.metadata as m\n"
        "names=('scikit-learn','skl2onnx','onnx','onnxruntime','numpy','pandas','protobuf')\n"
        "parts=['python ' + '.'.join(map(str, sys.version_info[:3]))]\n"
        "for n in names:\n"
        "    try:\n"
        "        parts.append(n + ' ' + m.version(n))\n"
        "    except Exception:\n"
        "        pass\n"
        "print(' | '.join(parts))\n"
    )
    completed = subprocess.run(
        [str(venv / "bin" / "python"), "-c", probe],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "(version probe failed)"


def run_entry(entry: Entry, *, workspace: Path, verbose: bool) -> Result:
    venv = workspace / entry.name
    environment = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}
    result = Result(entry)

    for command in entry.install_commands(venv):
        install = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=not verbose,
            text=True,
        )
        if install.returncode != 0:
            result.stage = "install"
            result.returncode = install.returncode
            if not verbose and install.stderr:
                result.failures = install.stderr.strip().splitlines()[-8:]
            return result

    result.versions = installed_versions(venv)
    print(f"    {result.versions}", flush=True)

    tests = subprocess.run(
        entry.pytest_command(venv),
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=not verbose,
        text=True,
    )
    if tests.returncode != 0:
        result.stage = "pytest"
        result.returncode = tests.returncode
        if not verbose:
            output = f"{tests.stdout}\n{tests.stderr}"
            result.failures = [
                line
                for line in output.splitlines()
                if line.startswith(("FAILED", "ERROR")) or " failed" in line
            ][-12:]
    return result


def report(results: list[Result]) -> int:
    print("\n" + "=" * 78)
    blocking = 0
    for result in results:
        if result.ok:
            status = "PASS"
        elif result.entry.allow_failure:
            status = "WARN"
        else:
            status = "FAIL"
            blocking += 1
        print(f"  {status:4}  {result.entry.name:16}  {result.versions}")
        if not result.ok:
            print(f"        stage={result.stage} exit={result.returncode}")
            for line in result.failures:
                print(f"        {line}")
    print("=" * 78)
    if blocking:
        print(f"{blocking} blocking failure(s).")
    else:
        print("All blocking entries passed.")
    return 1 if blocking else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="entries to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list entries and exit")
    parser.add_argument(
        "--verbose", action="store_true", help="stream uv and pytest output"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="where to build the virtualenvs (default: a temporary directory)",
    )
    arguments = parser.parse_args()

    entries = load_entries()
    if arguments.list:
        for entry in entries:
            marker = " (advisory)" if entry.allow_failure else ""
            print(f"{entry.name:16} python {entry.python:5} {entry.resolution}{marker}")
            if entry.description:
                print(f"                 {entry.description}")
        return 0

    if arguments.names:
        known = {entry.name: entry for entry in entries}
        unknown = [name for name in arguments.names if name not in known]
        if unknown:
            raise SystemExit(
                f"Unknown entries: {', '.join(unknown)}. Known: {', '.join(known)}."
            )
        entries = [known[name] for name in arguments.names]

    workspace = arguments.workspace
    temporary = None
    if workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="falcon_matrix_")
        workspace = Path(temporary.name)
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        results = []
        for index, entry in enumerate(entries, start=1):
            print(f"\n[{index}/{len(entries)}] {entry.name}", flush=True)
            results.append(
                run_entry(entry, workspace=workspace, verbose=arguments.verbose)
            )
        return report(results)
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    sys.exit(main())
