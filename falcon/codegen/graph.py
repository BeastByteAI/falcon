"""Resolving falcon's string tensors into integer codes, at code generation time.

C has no string tensor, so before the graph reaches the FNNX compiler every string is
replaced by the integer indexing it: a category by its position in the fitted vocabulary,
a predicted label by its class index. The vocabularies leave as `StringMapping`, which the
generated helper header turns into lookup tables. The exported `.fnnx` file is untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import onnx
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, helper
from onnx.numpy_helper import to_array

from falcon.codegen.bundle import CodegenError

# Column types encoded by a lookup keyed on the whole input value, which is what makes an
# integer code a faithful stand-in for the string.
CODEABLE_COLUMN_TYPES = frozenset({"CAT_LOW_CARD", "CAT_HIGH_CARD"})

# Ops that carry a string through without inspecting it, so tracing walks past them.
VIEW_OPS = frozenset({"Identity", "Reshape", "Gather"})

LOOKUP_KEYS = {"OneHotEncoder": "cats_strings", "LabelEncoder": "keys_strings"}
LOOKUP_CODES = {"OneHotEncoder": "cats_int64s", "LabelEncoder": "keys_int64s"}

# Attributes that hold string tensor data, as opposed to string attributes that only
# configure a kernel (a locale, a mode).
STRING_VALUED_ATTRIBUTES = frozenset(
    {
        "cats_strings",
        "classlabels_strings",
        "keys_strings",
        "pool_strings",
        "values_strings",
    }
)


@dataclass(frozen=True)
class CategoricalMapping:
    """How one categorical input's strings map to the codes the artifact consumes."""

    name: str
    column_type: str
    categories: tuple[str, ...]
    missing_tokens: tuple[str, ...]
    missing_code: int

    @property
    def imputed(self) -> bool:
        return bool(self.missing_tokens)


@dataclass(frozen=True)
class StringMapping:
    categoricals: tuple[CategoricalMapping, ...]
    class_labels: tuple[str, ...]


def resolve_strings(
    model: ModelProto, column_types: dict[str, str], output_names: list[str]
) -> StringMapping:
    """Rewrite `model` in place so no tensor is a string, and return what was replaced.

    Raises `CodegenError` for a feature whose encoding reads inside the string rather than
    looking the whole value up (free text, dates), since no integer stands in for those.
    """
    graph = model.graph
    _reject_uncodeable_columns(graph, column_types)

    categoricals = tuple(
        _recode_categorical(graph, value_info.name, column_types)
        for value_info in graph.input
        if value_info.type.tensor_type.elem_type == TensorProto.STRING
    )
    class_labels = _drop_label_decoder(graph)

    _prune(graph)
    _name_batch_dimension(graph)
    _rename_outputs(graph, output_names)
    del graph.value_info[:]

    _reject_remaining_strings(graph)
    onnx.checker.check_model(model, full_check=False)
    return StringMapping(categoricals=categoricals, class_labels=class_labels)


def _reject_uncodeable_columns(graph: GraphProto, column_types: dict[str, str]) -> None:
    offenders = [
        (value_info.name, column_types.get(value_info.name, "unknown"))
        for value_info in graph.input
        if value_info.type.tensor_type.elem_type == TensorProto.STRING
        and column_types.get(value_info.name, "unknown") not in CODEABLE_COLUMN_TYPES
    ]
    if not offenders:
        return
    listed = ", ".join(f"`{name}` ({column_type})" for name, column_type in offenders)
    raise CodegenError(
        f"Cannot generate C for {listed}. Text and date features are encoded by "
        "splitting and parsing the string itself, so no integer code can stand in for "
        "it the way one can for a category. Drop these columns from `features`, or "
        "deploy the model through the FNNX runtime instead."
    )


def _recode_categorical(
    graph: GraphProto, name: str, column_types: dict[str, str]
) -> CategoricalMapping:
    path, terminal = _trace(graph, name)
    attribute_name = LOOKUP_KEYS[terminal.op_type]
    keys = _attribute(terminal, attribute_name)
    categories = tuple(value.decode() for value in keys.strings)
    if terminal.op_type == "OneHotEncoder" and _zeros(terminal) != 1:
        raise CodegenError(
            f"The OneHotEncoder for `{name}` reports an unseen category as an error "
            "rather than an all-zero row, which a code the artifact never saw would "
            "trigger at inference."
        )

    terminal.attribute.remove(keys)
    terminal.attribute.append(
        helper.make_attribute(
            LOOKUP_CODES[terminal.op_type], list(range(len(categories)))
        )
    )

    missing_tokens, fill_value = _elide_imputation(graph, path)
    _set_input_type(graph, name, TensorProto.INT64)
    return CategoricalMapping(
        name=name,
        column_type=column_types.get(name, "unknown"),
        categories=categories,
        missing_tokens=missing_tokens,
        missing_code=categories.index(fill_value) if fill_value in categories else -1,
    )


def _trace(graph: GraphProto, name: str) -> tuple[list[NodeProto], NodeProto]:
    """Walk from a string input to the node that looks its value up.

    The mask-building `Equal` nodes of the string imputer also read the input, so only the
    edge carrying the value itself is followed. The nodes on the way are returned so the
    imputer can be elided once the vocabulary is known.
    """
    path: list[NodeProto] = []
    current = name
    while True:
        candidates = [
            node for node in _consumers(graph, current) if _reads_value(node, current)
        ]
        if len(candidates) != 1:
            raise CodegenError(
                f"Cannot generate C for `{name}`: expected its value to flow into exactly "
                f"one operation, found {len(candidates)}."
            )
        node = candidates[0]
        if node.op_type in LOOKUP_KEYS:
            return path, node
        if node.op_type not in VIEW_OPS and node.op_type != "Where":
            raise CodegenError(
                f"Cannot generate C for `{name}`: it flows into `{node.op_type}`, which "
                "is not a category lookup."
            )
        path.append(node)
        current = node.output[0]


def _reads_value(node: NodeProto, name: str) -> bool:
    """Whether `name` reaches `node` as the data being encoded, not as a mask or index."""
    if node.op_type == "Where":
        return len(node.input) > 2 and node.input[2] == name
    if node.op_type == "Equal":
        return False
    return bool(node.input) and node.input[0] == name


def _elide_imputation(
    graph: GraphProto, path: list[NodeProto]
) -> tuple[tuple[str, ...], str | None]:
    """Turn the string imputer's `Where` into a pass-through and report what it filled.

    The fill is not lost: the caller maps a missing string to the fill value's code in the
    generated helper, so the lookup still sees the category the `Where` produced.
    """
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    tokens: tuple[str, ...] = ()
    fill_value: str | None = None
    for node in path:
        if node.op_type != "Where":
            continue
        fill_value = _string_constant(initializers, node.input[1])
        tokens = _missing_tokens(graph, initializers, node.input[0])
        data = node.input[2]
        node.op_type = "Identity"
        del node.input[:]
        node.input.append(data)
        del node.attribute[:]
    return tokens, fill_value


def _missing_tokens(
    graph: GraphProto, initializers: dict[str, TensorProto], mask: str
) -> tuple[str, ...]:
    """The strings the imputer treats as missing, read off the `Equal`/`Or` mask."""
    produced_by = {output: node for node in graph.node for output in node.output}
    tokens: list[str] = []
    pending = [mask]
    while pending:
        node = produced_by.get(pending.pop())
        if node is None:
            continue
        if node.op_type == "Or":
            pending.extend(node.input)
        elif node.op_type == "Equal":
            for operand in node.input:
                value = _string_constant(initializers, operand)
                if value is not None:
                    tokens.append(value)
    return tuple(dict.fromkeys(tokens))


def _string_constant(initializers: dict[str, TensorProto], name: str) -> str | None:
    tensor = initializers.get(name)
    if tensor is None or tensor.data_type != TensorProto.STRING:
        return None
    value = to_array(tensor).reshape(-1)
    return str(value[0].decode() if isinstance(value[0], bytes) else value[0])


def _drop_label_decoder(graph: GraphProto) -> tuple[str, ...]:
    """Remove the `LabelEncoder` decoding class indices, leaving the index as the output."""
    outputs = {output.name: output for output in graph.output}
    for node in list(graph.node):
        if node.op_type != "LabelEncoder" or node.output[0] not in outputs:
            continue
        values = next((a for a in node.attribute if a.name == "values_strings"), None)
        if values is None:
            continue
        value_info = outputs[node.output[0]]
        value_info.name = node.input[0]
        value_info.type.tensor_type.elem_type = TensorProto.INT64
        graph.node.remove(node)
        return tuple(label.decode() for label in values.strings)
    return ()


def _prune(graph: GraphProto) -> None:
    produced_by = {output: node for node in graph.node for output in node.output}
    seen: set[str] = {output.name for output in graph.output}
    live_nodes: set[int] = set()
    pending = list(seen)
    while pending:
        node = produced_by.get(pending.pop())
        if node is None or id(node) in live_nodes:
            continue
        live_nodes.add(id(node))
        for name in node.input:
            if name and name not in seen:
                seen.add(name)
                pending.append(name)

    kept = [node for node in graph.node if id(node) in live_nodes]
    del graph.node[:]
    graph.node.extend(kept)

    reachable = {name for node in kept for name in node.input}
    retained = [tensor for tensor in graph.initializer if tensor.name in reachable]
    del graph.initializer[:]
    graph.initializer.extend(retained)


def _name_batch_dimension(graph: GraphProto) -> None:
    """Give the leading axis a name so the artifact can serve a range of batch sizes."""
    for value_info in graph.input:
        dimensions = value_info.type.tensor_type.shape.dim
        if dimensions and not dimensions[0].HasField("dim_value"):
            dimensions[0].dim_param = "batch"


def _rename_outputs(graph: GraphProto, names: list[str]) -> None:
    """Rename graph outputs to the names the manifest advertises (`y_pred`, ...).

    The compiler derives its C parameter names from these, so the artifact ends up naming
    its outputs the way the Python API does rather than after internal pipeline tensors.
    """
    if len(names) != len(graph.output):
        return
    taken = {name for node in graph.node for name in node.input} | {
        name for node in graph.node for name in node.output
    }
    mapping = {
        value_info.name: name
        for value_info, name in zip(graph.output, names, strict=True)
        if value_info.name != name and name not in taken
    }
    for node in graph.node:
        for names in (node.input, node.output):
            for index, name in enumerate(names):
                if name in mapping:
                    names[index] = mapping[name]
    for value_info in graph.output:
        value_info.name = mapping.get(value_info.name, value_info.name)


def _reject_remaining_strings(graph: GraphProto) -> None:
    offenders = sorted(
        {
            name
            for name, elem_type in _element_types(graph)
            if elem_type == TensorProto.STRING
        }
    )
    if offenders:
        raise CodegenError(
            "The graph still holds string tensors after recoding: "
            f"{', '.join(f'`{name}`' for name in offenders)}."
        )


def _element_types(graph: GraphProto) -> Iterator[tuple[str, int]]:
    for value_info in list(graph.input) + list(graph.output):
        yield value_info.name, value_info.type.tensor_type.elem_type
    for tensor in graph.initializer:
        yield tensor.name, tensor.data_type
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.name in STRING_VALUED_ATTRIBUTES and attribute.strings:
                yield f"{node.name}.{attribute.name}", TensorProto.STRING


def _set_input_type(graph: GraphProto, name: str, elem_type: int) -> None:
    for value_info in graph.input:
        if value_info.name == name:
            value_info.type.tensor_type.elem_type = elem_type


def _consumers(graph: GraphProto, name: str) -> list[NodeProto]:
    return [node for node in graph.node if name in node.input]


def _attribute(node: NodeProto, name: str) -> onnx.AttributeProto:
    for attribute in node.attribute:
        if attribute.name == name:
            return attribute
    raise CodegenError(
        f"Node `{node.name}` (`{node.op_type}`) has no `{name}` attribute."
    )


def _zeros(node: NodeProto) -> int:
    for attribute in node.attribute:
        if attribute.name == "zeros":
            return int(attribute.i)
    return 1
