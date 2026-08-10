from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import onnx
from onnx import ModelProto

STANDARD_ONNX_DOMAINS = frozenset({"ai.onnx", "ai.onnx.ml"})
_MODEL_PATH = "ops_artifacts/onnx_main/model.onnx"
_OPS_PATH = "ops.json"


@dataclass(frozen=True)
class ExtractedFNNXGraph:
    model: ModelProto
    fnnx_opset_domains: tuple[str, ...]
    declared_ir_version: int


def read_bundle_member(archive: tarfile.TarFile, member_name: str) -> bytes:
    try:
        member = archive.extractfile(member_name)
    except KeyError as error:
        raise AssertionError(f"FNNX bundle is missing {member_name}") from error
    if member is None:
        raise AssertionError(f"FNNX bundle member {member_name} is not a file")
    return member.read()


def _parse_fnnx_opset_domains(raw_ops: object) -> tuple[str, ...]:
    if not isinstance(raw_ops, list):
        raise TypeError("FNNX ops.json must contain a list of operations")

    domains: list[str] = []
    for op_index, raw_op in enumerate(raw_ops):
        if not isinstance(raw_op, dict):
            raise TypeError(f"FNNX operation {op_index} must be an object")
        attributes = raw_op.get("attributes")
        if not isinstance(attributes, dict):
            raise TypeError(f"FNNX operation {op_index} has no attributes object")
        raw_opsets = attributes.get("opsets")
        if not isinstance(raw_opsets, list):
            raise TypeError(f"FNNX operation {op_index} has no opsets list")
        for opset_index, raw_opset in enumerate(raw_opsets):
            if not isinstance(raw_opset, dict):
                raise TypeError(
                    f"FNNX operation {op_index} opset {opset_index} must be an object"
                )
            domain = raw_opset.get("domain")
            if not isinstance(domain, str):
                raise TypeError(
                    f"FNNX operation {op_index} opset {opset_index} has no domain"
                )
            domains.append(domain)
    return tuple(domains)


def _parse_declared_ir_version(raw_ops: object) -> int:
    if not isinstance(raw_ops, list) or not raw_ops:
        raise TypeError("FNNX ops.json must contain a non-empty list of operations")

    attributes = raw_ops[0].get("attributes") if isinstance(raw_ops[0], dict) else None
    if not isinstance(attributes, dict):
        raise TypeError("FNNX operation 0 has no attributes object")

    declared = attributes.get("onnx_ir_version")
    if not isinstance(declared, int):
        raise TypeError("FNNX operation 0 declares no onnx_ir_version")
    return declared


def _extract_graph_from_archive(archive: tarfile.TarFile) -> ExtractedFNNXGraph:
    model = onnx.load_model_from_string(read_bundle_member(archive, _MODEL_PATH))
    raw_ops = json.loads(read_bundle_member(archive, _OPS_PATH))
    return ExtractedFNNXGraph(
        model=model,
        fnnx_opset_domains=_parse_fnnx_opset_domains(raw_ops),
        declared_ir_version=_parse_declared_ir_version(raw_ops),
    )


def extract_fnnx_graph(bundle: bytes | str | Path) -> ExtractedFNNXGraph:
    if isinstance(bundle, bytes):
        with tarfile.open(fileobj=BytesIO(bundle), mode="r:") as archive:
            return _extract_graph_from_archive(archive)

    with tarfile.open(name=Path(bundle), mode="r:") as archive:
        return _extract_graph_from_archive(archive)


def _normalized_domain(domain: str) -> str:
    return "ai.onnx" if domain in {"", "ai.onnx"} else domain


def assert_onnx_is_valid(model: ModelProto) -> None:
    onnx.checker.check_model(model)


def _iter_nested_nodes(node: onnx.NodeProto) -> Iterator[onnx.NodeProto]:
    for attribute in node.attribute:
        if attribute.HasField("g"):
            yield from _iter_graph_nodes(attribute.g)
        for graph in attribute.graphs:
            yield from _iter_graph_nodes(graph)


def _iter_graph_nodes(graph: onnx.GraphProto) -> Iterator[onnx.NodeProto]:
    for node in graph.node:
        yield node
        yield from _iter_nested_nodes(node)


def _iter_model_nodes(model: ModelProto) -> Iterator[onnx.NodeProto]:
    yield from _iter_graph_nodes(model.graph)
    for function in model.functions:
        for node in function.node:
            yield node
            yield from _iter_nested_nodes(node)


def assert_standard_node_domains(model: ModelProto) -> None:
    violations: list[str] = []
    for node in _iter_model_nodes(model):
        domain = _normalized_domain(node.domain)
        if domain not in STANDARD_ONNX_DOMAINS:
            node_name = node.name or f"<unnamed {node.op_type}>"
            violations.append(f"{node_name} ({node.op_type}) uses {domain}")

    if violations:
        raise AssertionError("Non-standard ONNX node domains: " + "; ".join(violations))


def assert_standard_opset_declarations(graph: ExtractedFNNXGraph) -> None:
    model_domains = {
        _normalized_domain(opset.domain) for opset in graph.model.opset_import
    }
    fnnx_domains = {_normalized_domain(domain) for domain in graph.fnnx_opset_domains}
    violations = [
        *(
            f"model imports {domain}"
            for domain in sorted(model_domains - STANDARD_ONNX_DOMAINS)
        ),
        *(
            f"FNNX ops.json declares {domain}"
            for domain in sorted(fnnx_domains - STANDARD_ONNX_DOMAINS)
        ),
    ]

    if violations:
        raise AssertionError(
            "Non-standard ONNX opset declarations: " + "; ".join(violations)
        )


def assert_fnnx_conforms(graph: ExtractedFNNXGraph) -> None:
    assert_onnx_is_valid(graph.model)
    assert_standard_node_domains(graph.model)
    assert_standard_opset_declarations(graph)
