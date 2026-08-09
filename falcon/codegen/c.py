"""Generating a self-contained C99 artifact from an exported falcon model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from falcon import __version__
from falcon.codegen.bundle import Bundle, CodegenError, read_bundle
from falcon.codegen.graph import StringMapping, resolve_strings
from falcon.codegen.helper import render_helper
from falcon.utils import logger

DEFAULT_BATCH_SIZE = 128

_COLUMN_NOTES = {
    "NUMERIC_REGULAR": "numeric feature, as-is",
    "CAT_LOW_CARD": "category code from {prefix}_encode_{name}()",
    "CAT_HIGH_CARD": "category code from {prefix}_encode_{name}()",
}


@dataclass(frozen=True)
class CArtifact:
    """The files written for one model, and the string tables they were built with."""

    header_path: Path
    helper_path: Path
    report_path: Path
    mapping: StringMapping
    report: dict[str, Any]

    @property
    def prefix(self) -> str:
        return str(self.report["prefix"])

    @property
    def entrypoint(self) -> str:
        return f"{self.prefix}_run"

    @property
    def batch_size(self) -> int:
        dims = self.report.get("runtime_dims") or []
        return int(dims[0]["max"]) if dims else 1


def compile_to_c(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    prefix: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> CArtifact:
    """Compile an exported `.fnnx` model into a self-contained C99 artifact.

    Writes three files into `output_dir`: `<prefix>.h` with the model as straight-line C,
    `<prefix>_falcon.h` with the category and class-label tables, and the compiler's
    `<prefix>_report.json`. Neither header needs a runtime, an allocator, or ONNX.

    `batch_size` is the largest number of rows one call may pass; a call can pass fewer.
    The model at `model_path` is left unmodified.

    Raises `CodegenError` for a model C cannot represent, in practice one with a text or
    date feature.
    """
    if batch_size < 1:
        raise CodegenError(f"batch_size must be at least 1, got {batch_size}.")
    compile_onnx = _load_compiler()

    bundle = read_bundle(model_path)
    identifier = _sanitize(prefix or bundle.name)
    logger.info(f"Generating C for `{identifier}` from {model_path}...")

    mapping = resolve_strings(bundle.model, bundle.column_types(), bundle.output_names)
    destination = Path(output_dir)
    result = compile_onnx(
        bundle.model,
        destination,
        prefix=identifier,
        runtime_dims={"batch": batch_size},
    )

    helper_path = destination / f"{identifier}_falcon.h"
    helper_path.write_text(
        render_helper(
            identifier,
            mapping,
            version=__version__,
            inputs=_input_notes(identifier, result.report, bundle),
            outputs=[
                (tensor["c_name"], tensor["c_type"])
                for tensor in result.report["entrypoint"]["outputs"]
            ],
        ),
        encoding="utf-8",
    )
    logger.info(
        f"Wrote {result.header_path.name} and {helper_path.name} to {destination} "
        f"({result.report['memory']['static_bytes'] / 1024:.1f} KiB static)."
    )
    return CArtifact(
        header_path=result.header_path,
        helper_path=helper_path,
        report_path=result.report_path,
        mapping=mapping,
        report=result.report,
    )


def _input_notes(
    prefix: str, report: dict[str, Any], bundle: Bundle
) -> list[tuple[str, str, str]]:
    column_types = bundle.column_types()
    notes = []
    for tensor in report["entrypoint"]["inputs"]:
        column_type = column_types.get(tensor["name"], "unknown")
        template = _COLUMN_NOTES.get(column_type, f"{column_type} feature")
        notes.append(
            (
                tensor["c_name"],
                tensor["c_type"],
                template.format(prefix=prefix, name=tensor["name"]),
            )
        )
    return notes


def _load_compiler() -> Any:
    try:
        from fnnx.extras.compilers.c import compile_onnx
    except ImportError as error:
        raise CodegenError(
            "Generating C requires the FNNX ahead-of-time compiler. Install it with "
            '`pip install "fnnx[compiler]"`.'
        ) from error
    return compile_onnx


def _sanitize(name: str) -> str:
    identifier = re.sub(r"[^0-9a-zA-Z_]", "_", name).strip("_").lower()
    if not identifier or identifier[0].isdigit():
        identifier = f"model_{identifier}" if identifier else "model"
    return identifier
