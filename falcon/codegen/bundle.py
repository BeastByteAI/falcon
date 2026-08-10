from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx

from falcon.constants import DEFAULT_PRODUCER_NAME

MANIFEST_FILE = "manifest.json"
VARIANT_FILE = "variant_config.json"
ONNX_MODEL_FILE = "ops_artifacts/onnx_main/model.onnx"


class CodegenError(Exception):
    """Raised when a model cannot be turned into C."""


@dataclass(frozen=True)
class Bundle:
    """The parts of an exported `.fnnx` file the C code generator reads."""

    model: onnx.ModelProto
    manifest: dict[str, Any]
    variant: dict[str, Any]

    @property
    def name(self) -> str:
        tags = self.manifest.get("producer_tags", [])
        for tag in tags:
            if tag.startswith(f"{DEFAULT_PRODUCER_NAME}::tabular_"):
                return tag.split("::", 1)[1].split(":", 1)[0]
        return "falcon_model"

    @property
    def output_names(self) -> list[str]:
        return [output["name"] for output in self.manifest.get("outputs", [])]

    def column_types(self) -> dict[str, str]:
        """Falcon's inferred type per feature column, keyed by column name."""
        schema = self.manifest.get("schema")
        if not isinstance(schema, dict):
            return {}
        return {
            column["name"]: column["type"]
            for column in schema.get("columns", [])
            if isinstance(column, dict)
        }


def read_bundle(path: str | Path) -> Bundle:
    """Read a `.fnnx` file, or an already unpacked bundle directory."""
    source = Path(path)
    if not source.exists():
        raise CodegenError(f"Model not found: `{source}`.")
    reader = _read_directory if source.is_dir() else _read_archive
    try:
        model_bytes, manifest_bytes, variant_bytes = reader(source)
    except (KeyError, OSError, tarfile.TarError) as error:
        raise CodegenError(
            f"Could not read the FNNX bundle `{source}`: {error}."
        ) from error

    manifest = json.loads(manifest_bytes)
    if manifest.get("variant") != "pipeline":
        raise CodegenError(
            f"`{source}` is an FNNX `{manifest.get('variant')}` bundle; the C code "
            "generator only handles the `pipeline` variant that falcon exports."
        )
    return Bundle(
        model=onnx.load_from_string(model_bytes),
        manifest=manifest,
        variant=json.loads(variant_bytes),
    )


def _read_directory(source: Path) -> tuple[bytes, bytes, bytes]:
    return (
        (source / ONNX_MODEL_FILE).read_bytes(),
        (source / MANIFEST_FILE).read_bytes(),
        (source / VARIANT_FILE).read_bytes(),
    )


def _read_archive(source: Path) -> tuple[bytes, bytes, bytes]:
    with tarfile.open(source) as archive:
        return (
            _member(archive, ONNX_MODEL_FILE),
            _member(archive, MANIFEST_FILE),
            _member(archive, VARIANT_FILE),
        )


def _member(archive: tarfile.TarFile, name: str) -> bytes:
    handle = archive.extractfile(name)
    if handle is None:
        raise KeyError(f"`{name}` is missing or is not a file")
    return handle.read()
