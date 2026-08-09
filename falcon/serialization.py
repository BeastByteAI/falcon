import io
import json
import tarfile
from collections.abc import Iterator, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

import onnx
from onnx import ModelProto
from onnx import helper as h
from onnx.compose import add_prefix, merge_models
from onnx.helper import make_model

from falcon import __version__ as falcon_version
from falcon.config import (
    ML_ONNX_OPSET_VERSION,
    ONNX_IR_VERSION,
    ONNX_OPSET_VERSION,
)
from falcon.constants import DEFAULT_PRODUCER_NAME
from falcon.types import ColumnTypes, DatasetSchema
from falcon.utils import logger

onnx_type_map: dict[int, str] = {
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.UINT16: "uint16",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.STRING: "string",
    onnx.TensorProto.BOOL: "bool",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.DOUBLE: "float64",
    onnx.TensorProto.UINT32: "uint32",
    onnx.TensorProto.UINT64: "uint64",
    onnx.TensorProto.COMPLEX64: "complex64",
    onnx.TensorProto.COMPLEX128: "complex128",
    onnx.TensorProto.BFLOAT16: "bfloat16",
}


class SerializedModelRepr:
    def __init__(
        self,
        model: onnx.ModelProto,
        n_inputs: int,
        n_outputs: int,
        initial_types: list[str],
        initial_shapes: list[list[int | None]],
        type_: str = "onnx",
    ) -> None:
        self._model = model
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._initial_types = initial_types
        self._initial_shapes = initial_shapes
        self._type = type_

    def get_model(self) -> onnx.ModelProto:
        return self._model

    def get_n_inputs(self) -> int:
        return self._n_inputs

    def get_n_outputs(self) -> int:
        return self._n_outputs

    def get_initial_types(self) -> list[str]:
        return self._initial_types

    def get_initial_shapes(self) -> list[list[int | None]]:
        return self._initial_shapes

    def get_type(self) -> str:
        return self._type

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_inputs": self._n_inputs,
            "n_outputs": self._n_outputs,
            "initial_types": self._initial_types,
            "initial_shapes": self._initial_shapes,
            "model": self._model,
            "type": self._type,
        }


def _sanitize_feature_name(name: Any, index: int) -> str:
    sanitized = "".join(
        character if character.isalnum() else "_" for character in str(name)
    )
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized or f"feature_{index}"


def _sanitized_feature_names(feature_names: list[Any], count: int) -> list[str]:
    if len(feature_names) != count:
        feature_names = [f"feature_{index}" for index in range(count)]

    sanitized_names: list[str] = []
    used_names: set[str] = set()
    for index, feature_name in enumerate(feature_names):
        base_name = _sanitize_feature_name(feature_name, index)
        unique_name = base_name
        if unique_name in used_names:
            unique_name = f"{base_name}_{index}"
            suffix = 1
            while unique_name in used_names:
                unique_name = f"{base_name}_{index}_{suffix}"
                suffix += 1
        sanitized_names.append(unique_name)
        used_names.add(unique_name)
    return sanitized_names


def _rename_inputs(
    model: onnx.ModelProto,
    feature_names: list[Any],
) -> None:
    sanitized_names = _sanitized_feature_names(feature_names, len(model.graph.input))
    mapping: dict[str, str] = {}
    for i, inp in enumerate(model.graph.input):
        new_name = sanitized_names[i]
        mapping[inp.name] = new_name
        inp.name = new_name
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name in mapping:
                node.input[input_index] = mapping[input_name]


def _iter_graph_nodes(graph: onnx.GraphProto) -> Iterator[onnx.NodeProto]:
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from _iter_graph_nodes(attribute.g)
            for nested_graph in attribute.graphs:
                yield from _iter_graph_nodes(nested_graph)


def _normalized_onnx_domain(domain: str) -> str:
    return "" if domain in {"", "ai.onnx"} else domain


def _opset_imports_for_models(
    models: Sequence[onnx.ModelProto],
) -> list[onnx.OperatorSetIdProto]:
    used_domains = {
        _normalized_onnx_domain(node.domain)
        for model in models
        for node in _iter_graph_nodes(model.graph)
    }
    declared_versions: dict[str, list[int]] = {}
    for model in models:
        for opset in model.opset_import:
            domain = _normalized_onnx_domain(opset.domain)
            declared_versions.setdefault(domain, []).append(opset.version)

    imports: list[onnx.OperatorSetIdProto] = []
    for domain in sorted(
        used_domains,
        key=lambda item: (item != "", item != "ai.onnx.ml", item),
    ):
        if domain == "":
            version = ONNX_OPSET_VERSION
        elif domain == "ai.onnx.ml":
            version = ML_ONNX_OPSET_VERSION
        elif domain in declared_versions:
            version = max(declared_versions[domain])
        else:
            raise ValueError(f"No opset version is declared for domain {domain!r}")
        imports.append(h.make_operatorsetid(domain, version))
    return imports


def _normalize_regression_output(model: onnx.ModelProto) -> None:
    if not model.graph.output:
        raise RuntimeError("A regression graph must expose an output")

    output_count = len(model.graph.output)
    for index, source_output in enumerate(model.graph.output):
        if not source_output.type.HasField("tensor_type"):
            raise RuntimeError("A regression graph must expose tensor outputs")

        source_name = source_output.name
        shape_name = f"{source_name}_falcon_shape"
        normalized_name = f"{source_name}_falcon_normalized"
        node_suffix = "" if output_count == 1 else f"_{index}"
        model.graph.initializer.append(
            h.make_tensor(shape_name, onnx.TensorProto.INT64, [1], [-1])
        )
        model.graph.node.append(
            h.make_node(
                "Reshape",
                [source_name, shape_name],
                [normalized_name],
                name=f"falcon_normalize_regression_output{node_suffix}",
            )
        )
        normalized_output = h.make_tensor_value_info(
            normalized_name,
            source_output.type.tensor_type.elem_type,
            [None],
        )
        source_output.CopyFrom(normalized_output)


def _normalized_ensemble_weights(weights: Sequence[float]) -> list[float]:
    if not weights or any(weight < 0 for weight in weights):
        raise ValueError("Ensemble weights must be non-negative and not empty")
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one ensemble weight must be positive")
    return [weight / total for weight in weights]


def _replace_graph_input(
    graph: onnx.GraphProto,
    source_name: str,
    target_name: str,
) -> None:
    for node in _iter_graph_nodes(graph):
        for index, input_name in enumerate(node.input):
            if input_name == source_name:
                node.input[index] = target_name


def _append_average(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    inputs: Sequence[str],
    output: str,
    name: str,
) -> None:
    if len(inputs) == 1:
        nodes.append(h.make_node("Identity", list(inputs), [output], name=name))
        return

    summed = f"{output}_sum"
    divisor = f"{output}_divisor"
    nodes.append(h.make_node("Sum", list(inputs), [summed], name=f"{name}_sum"))
    initializers.append(
        h.make_tensor(divisor, onnx.TensorProto.FLOAT, [], [float(len(inputs))])
    )
    nodes.append(h.make_node("Div", [summed, divisor], [output], name=f"{name}_divide"))


def serialize_parallel_ensemble(
    fold_models: Sequence[Sequence[SerializedModelRepr]],
    weights: Sequence[float],
    task: str,
    *,
    classes: Sequence[int] | None = None,
) -> SerializedModelRepr:
    if task not in {"tabular_classification", "tabular_regression"}:
        raise ValueError(f"Unknown task `{task}`")
    if len(fold_models) != len(weights):
        raise ValueError("Each ensemble member must have one weight")
    if not fold_models or any(not models for models in fold_models):
        raise ValueError("Each ensemble member must contain at least one fold model")
    normalized_weights = _normalized_ensemble_weights(weights)
    serialized_models = [model for models in fold_models for model in models]
    model_protos = [model.get_model() for model in serialized_models]
    if any(len(model.graph.input) != 1 for model in model_protos):
        raise ValueError("Ensemble fold models must expose exactly one input")

    shared_input = deepcopy(model_protos[0].graph.input[0])
    shared_input.name = "ensemble_input"
    expected_input_type = shared_input.type.SerializeToString()
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    sparse_initializers: list[onnx.SparseTensorProto] = []
    value_info: list[onnx.ValueInfoProto] = []
    functions: list[onnx.FunctionProto] = []
    member_outputs: list[str] = []
    model_index = 0

    for member_index, models in enumerate(fold_models):
        fold_outputs: list[str] = []
        for fold_index, _ in enumerate(models):
            model = model_protos[model_index]
            model_index += 1
            if model.graph.input[0].type.SerializeToString() != expected_input_type:
                raise ValueError("Ensemble fold model input types must match")
            if task == "tabular_classification" and len(model.graph.output) < 2:
                raise ValueError(
                    "Classification fold models must expose labels and probabilities"
                )
            if task == "tabular_regression" and len(model.graph.output) != 1:
                raise ValueError(
                    "Regression fold models must expose exactly one prediction output"
                )

            prefix = f"falcon-ensemble/member-{member_index}/fold-{fold_index}/"
            prefixed = add_prefix(model, prefix=prefix)
            branch_input = prefixed.graph.input[0].name
            _replace_graph_input(prefixed.graph, branch_input, shared_input.name)
            nodes.extend(deepcopy(prefixed.graph.node))
            initializers.extend(deepcopy(prefixed.graph.initializer))
            sparse_initializers.extend(deepcopy(prefixed.graph.sparse_initializer))
            value_info.extend(deepcopy(prefixed.graph.value_info))
            functions.extend(deepcopy(prefixed.functions))

            source_output = (
                prefixed.graph.output[-1].name
                if task == "tabular_classification"
                else prefixed.graph.output[0].name
            )
            cast_output = f"{prefix}falcon_float_output"
            nodes.append(
                h.make_node(
                    "Cast",
                    [source_output],
                    [cast_output],
                    to=onnx.TensorProto.FLOAT,
                    name=f"{prefix}falcon_cast_output",
                )
            )
            if task == "tabular_regression":
                shape_name = f"{prefix}falcon_output_shape"
                normalized_output = f"{prefix}falcon_normalized_output"
                initializers.append(
                    h.make_tensor(shape_name, onnx.TensorProto.INT64, [1], [-1])
                )
                nodes.append(
                    h.make_node(
                        "Reshape",
                        [cast_output, shape_name],
                        [normalized_output],
                        name=f"{prefix}falcon_reshape_output",
                    )
                )
                fold_outputs.append(normalized_output)
            else:
                fold_outputs.append(cast_output)

        member_output = f"falcon-ensemble/member-{member_index}/fold_average"
        _append_average(
            nodes,
            initializers,
            fold_outputs,
            member_output,
            f"falcon-ensemble/member-{member_index}/average_folds",
        )
        weighted_output = f"falcon-ensemble/member-{member_index}/weighted"
        weight_name = f"falcon-ensemble/member-{member_index}/weight"
        initializers.append(
            h.make_tensor(
                weight_name,
                onnx.TensorProto.FLOAT,
                [],
                [normalized_weights[member_index]],
            )
        )
        nodes.append(
            h.make_node(
                "Mul",
                [member_output, weight_name],
                [weighted_output],
                name=f"falcon-ensemble/member-{member_index}/apply_weight",
            )
        )
        member_outputs.append(weighted_output)

    ensemble_output = (
        "ensemble_probabilities"
        if task == "tabular_classification"
        else "ensemble_prediction"
    )
    if len(member_outputs) == 1:
        nodes.append(
            h.make_node(
                "Identity",
                member_outputs,
                [ensemble_output],
                name="falcon-ensemble/weighted_mean",
            )
        )
    else:
        nodes.append(
            h.make_node(
                "Sum",
                member_outputs,
                [ensemble_output],
                name="falcon-ensemble/weighted_mean",
            )
        )

    outputs: list[onnx.ValueInfoProto]
    if task == "tabular_classification":
        if classes is None or not classes:
            raise ValueError("Classification ensembles require encoded class labels")
        class_values = [int(label) for label in classes]
        class_indices = "ensemble_class_indices"
        class_labels = "ensemble_labels"
        class_initializer = "ensemble_classes"
        initializers.append(
            h.make_tensor(
                class_initializer,
                onnx.TensorProto.INT64,
                [len(class_values)],
                class_values,
            )
        )
        nodes.extend(
            [
                h.make_node(
                    "ArgMax",
                    [ensemble_output],
                    [class_indices],
                    axis=1,
                    keepdims=0,
                    name="falcon-ensemble/predict_class_index",
                ),
                h.make_node(
                    "Gather",
                    [class_initializer, class_indices],
                    [class_labels],
                    axis=0,
                    name="falcon-ensemble/decode_class_label",
                ),
            ]
        )
        outputs = [
            h.make_tensor_value_info(class_labels, onnx.TensorProto.INT64, [None]),
            h.make_tensor_value_info(
                ensemble_output,
                onnx.TensorProto.FLOAT,
                [None, len(class_values)],
            ),
        ]
    else:
        outputs = [
            h.make_tensor_value_info(
                ensemble_output,
                onnx.TensorProto.FLOAT,
                [None],
            )
        ]

    graph = h.make_graph(
        nodes,
        "falcon_parallel_ensemble",
        [shared_input],
        outputs,
        initializer=initializers,
        value_info=value_info,
        sparse_initializer=sparse_initializers,
    )
    opset_imports = _opset_imports_for_models(model_protos)
    if not any(opset.domain in {"", "ai.onnx"} for opset in opset_imports):
        opset_imports.append(h.make_operatorsetid("", ONNX_OPSET_VERSION))
    ensemble_model = h.make_model(
        graph,
        producer_name="Falcon ML",
        producer_version=falcon_version,
        opset_imports=opset_imports,
        ir_version=ONNX_IR_VERSION,
    )
    ensemble_model.functions.extend(functions)
    return SerializedModelRepr(
        ensemble_model,
        n_inputs=1,
        n_outputs=len(outputs),
        initial_types=serialized_models[0].get_initial_types(),
        initial_shapes=serialized_models[0].get_initial_shapes(),
    )


def serialize_to_onnx(
    models_: list[SerializedModelRepr],
    init_types: list[ColumnTypes] | None = None,
    init_feature_names: list[Any] | None = None,
    task: str | None = None,
) -> onnx.ModelProto:
    if init_types is None:
        init_types = []
    if init_feature_names is None:
        init_feature_names = []
    if len(models_) == 0:
        raise ValueError("List of models cannot be empty")

    updated_models: list[ModelProto] = []
    models = [m.get_model() for m in models_]
    opset_imports = _opset_imports_for_models(models)
    if task == "tabular_regression" and not any(
        opset.domain in {"", "ai.onnx"} for opset in opset_imports
    ):
        opset_imports.append(h.make_operatorsetid("", ONNX_OPSET_VERSION))
    for i, model in enumerate(models):
        updated_model = make_model(
            model.graph, opset_imports=opset_imports, ir_version=ONNX_IR_VERSION
        )
        updated_model = add_prefix(updated_model, prefix=f"falcon-pl-{i}/")
        updated_models.append(updated_model)

    prev: ModelProto = updated_models[0]
    for i in range(1, len(updated_models)):
        current: ModelProto = updated_models[i]
        prev_outputs = prev.graph.output
        current_inputs = current.graph.input
        if len(prev_outputs) > len(current_inputs):
            prev_outputs = prev_outputs[: len(current_inputs)]
        if len(prev_outputs) < len(current_inputs):
            raise RuntimeError(
                "When merging, previous model should have at least as many outputs as inputs in the next model."
            )
        io_map: list[tuple[str, str]] = []
        for p, c in zip(prev_outputs, current_inputs, strict=True):
            mapping: tuple[str, str] = (p.name, c.name)
            io_map.append(mapping)

        combined_model: ModelProto = merge_models(
            prev,
            current,
            io_map=io_map,
        )

        prev = combined_model
    combined_model = prev
    if task == "tabular_regression":
        _normalize_regression_output(combined_model)
    # TODO: Rename the inputs here
    description = {}
    if task is not None:
        description["task"] = task
    _rename_inputs(combined_model, init_feature_names)
    combined_model.graph.doc_string = str(description)
    combined_model.producer_name = "Falcon ML"
    combined_model.producer_version = falcon_version
    combined_model.ir_version = ONNX_IR_VERSION
    logger.info("Serialization completed.")
    return combined_model


input_tags: dict[ColumnTypes, list[str]] = {
    ColumnTypes.NUMERIC_REGULAR: [f"{DEFAULT_PRODUCER_NAME}::numeric:v1"],
    ColumnTypes.CAT_LOW_CARD: [f"{DEFAULT_PRODUCER_NAME}::categorical_lc:v1"],
    ColumnTypes.CAT_HIGH_CARD: [f"{DEFAULT_PRODUCER_NAME}::categorical_hc:v1"],
    ColumnTypes.TEXT_UTF8: [f"{DEFAULT_PRODUCER_NAME}::text:v1"],
    ColumnTypes.DATE_YMD_ISO8601: [f"{DEFAULT_PRODUCER_NAME}::date_ymd_iso8601:v1"],
    ColumnTypes.DATETIME_YMDHMS_ISO8601: [
        f"{DEFAULT_PRODUCER_NAME}::datetime_ymdhms_iso8601:v1"
    ],
}


@dataclass
class ModelIO:
    name: str
    dtype: str
    shape: list[int | str]
    tags: list[str] | None = None


class FNNXSerializer:
    out_names: dict[str, list[str]] = {
        "tabular_classification": ["probabilities", "y_pred"],
        "tabular_regression": ["y_pred", "y_lower", "y_upper"],
    }

    artifact_dirs: list[str] = [
        "meta_artifacts",
        "ops_artifacts",
        "variant_artifacts",
    ]

    def __init__(
        self,
        models: list[SerializedModelRepr],
        init_types: list[ColumnTypes] | None = None,
        init_feature_names: list[Any] | None = None,
        task: str | None = None,
        producer_name: str | None = DEFAULT_PRODUCER_NAME,
        producer_version: str | None = falcon_version,
        producer_extra_tags: list[str] | None = None,
        description: str = "",
        schema: DatasetSchema | None = None,
    ) -> None:
        self.models: list[SerializedModelRepr] = models
        if schema is not None:
            init_types = list(schema.column_types)
            init_feature_names = list(schema.column_names)
        self.init_types: list[ColumnTypes] | None = init_types
        self.schema = schema

        self.task: str | None = task
        self.description: str = description

        self.model_proto: onnx.ModelProto = serialize_to_onnx(
            models, init_types, init_feature_names, task
        )

        self.init_feature_names: list[Any] = init_feature_names or [
            f"input_{i}" for i in range(len(self.model_proto.graph.input))
        ]

        self.inputs: list[ModelIO] = self._map_io(
            self.model_proto.graph.input, self.init_feature_names
        )
        output_names = self.out_names.get(task or "")
        if output_names is None:
            raise ValueError(f"Unknown task `{task}`")
        output_count = len(self.model_proto.graph.output)
        valid_output_counts = {2} if task == "tabular_classification" else {1, 3}
        if output_count not in valid_output_counts:
            raise ValueError(
                f"Task `{task}` produced an unexpected number of outputs: "
                f"{output_count}"
            )
        self.outputs: list[ModelIO] = self._map_io(
            self.model_proto.graph.output,
            output_names[:output_count],
        )

        if init_types is not None and len(init_types) == len(self.inputs):
            for i, t in enumerate(init_types):
                self.inputs[i].tags = copy(input_tags.get(t, []))

        self.producer_name: str | None = producer_name
        self.producer_version: str | None = producer_version
        self.producer_extra_tags: list[str] = producer_extra_tags or []
        self.metadata_container: dict[str, Any] = {
            "id": "falcon_metrics",
            "producer": DEFAULT_PRODUCER_NAME,
            "producer_version": falcon_version,
            "producer_tags": [f"{DEFAULT_PRODUCER_NAME}::{self.task}_metrics:v1"],
        }
        self.metadata_payload: dict[str, Any] = {}
        self.metadata_container["payload"] = self.metadata_payload
        self.fh: TarHandler = TarHandler()

    def _map_io(
        self, io_: Sequence[onnx.ValueInfoProto], names: Sequence[str]
    ) -> list[ModelIO]:
        processed: list[ModelIO] = []
        for el, name in zip(io_, names, strict=True):
            shape = [
                (
                    dim.dim_value
                    if dim.dim_value != 0
                    else (dim.dim_param if dim.dim_param else "batch")
                )
                for dim in el.type.tensor_type.shape.dim
            ]
            elem_type = onnx_type_map.get(el.type.tensor_type.elem_type)
            if elem_type is None:
                raise ValueError(
                    f"Unknown element type: {el.type.tensor_type.elem_type}"
                )
            processed.append(ModelIO(name, f"Array[{elem_type}]", shape))
        return processed

    def serialize(self) -> bytes:
        self._add_manifest()
        self._add_env()
        self._add_variant_config()
        for d in self.artifact_dirs:
            self.fh.add_directory(d)
        self._add_onnx()
        self.fh.add_json("dtypes.json", {})
        if len(self.metadata_payload.keys()) > 0:
            metadata = [self.metadata_container]
        else:
            metadata = []
        self.fh.add_json("meta.json", metadata)
        return self.fh.finalize()

    def _add_manifest(self) -> None:
        manifest: dict[str, Any] = {
            "variant": "pipeline",
            "description": self.description,
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
            "producer_tags": [f"{DEFAULT_PRODUCER_NAME}::{self.task}:v1"]
            + self.producer_extra_tags,
            "inputs": [],
            "outputs": [],
            "dynamic_attributes": [],
            "env_vars": [],
        }
        if self.schema is not None:
            manifest["schema"] = self.schema.to_dict()

        for io_ in self.inputs:
            inp: dict[str, Any] = {
                "name": io_.name,
                "content_type": "NDJSON",
                "dtype": io_.dtype,
                "shape": io_.shape,
            }
            if io_.tags:
                inp["tags"] = io_.tags
            manifest["inputs"].append(inp)

        for io_ in self.outputs:
            manifest["outputs"].append(
                {
                    "name": io_.name,
                    "content_type": "NDJSON",
                    "dtype": io_.dtype,
                    "shape": io_.shape,
                }
            )

        self.fh.add_json("manifest.json", manifest)

    def _add_variant_config(self) -> None:
        config: dict[str, Any] = {
            "nodes": [
                {
                    "op_instance_id": "onnx_main",
                    "extra_dynattrs": {},
                    "inputs": [i.name for i in self.inputs],
                    "outputs": [o.name for o in self.outputs],
                }
            ]
        }

        self.fh.add_json("variant_config.json", config)

    def _add_onnx(self) -> None:
        op: list[dict[str, Any]] = [
            {
                "id": "onnx_main",
                "op": "ONNX_v1",
                "inputs": [{"dtype": i.dtype, "shape": i.shape} for i in self.inputs],
                "outputs": [{"dtype": o.dtype, "shape": o.shape} for o in self.outputs],
                "dynamic_attributes": {},
                "attributes": {
                    "opsets": [
                        {
                            "domain": (
                                "ai.onnx"
                                if opset.domain in {"", "ai.onnx"}
                                else opset.domain
                            ),
                            "version": opset.version,
                        }
                        for opset in self.model_proto.opset_import
                    ],
                    "requires_ort_extensions": False,
                    "has_external_data": False,  # TODO
                    "onnx_ir_version": self.model_proto.ir_version,
                },
            }
        ]

        self.fh.add_json("ops.json", op)

        onnx_model = self.model_proto.SerializeToString()

        self.fh.add_directory("ops_artifacts/onnx_main")
        self.fh.add_file("ops_artifacts/onnx_main/model.onnx", onnx_model)

    def _add_env(self) -> None:
        self.fh.add_json("env.json", {})  # TODO


class TarHandler:
    def __init__(self) -> None:
        self.tar_buffer: io.BytesIO = io.BytesIO()
        self.tar: tarfile.TarFile = tarfile.open(fileobj=self.tar_buffer, mode="w")
        self.finalized: bool = False

    def _assert_not_finalized(self) -> None:
        if self.finalized:
            raise RuntimeError("Tar file is already finalized.")

    def add_directory(self, directory_name: str) -> None:
        self._assert_not_finalized()
        if not directory_name.endswith("/"):
            directory_name += "/"
        folder_info = tarfile.TarInfo(name=directory_name)
        folder_info.type = tarfile.DIRTYPE
        folder_info.mode = 0o755
        self.tar.addfile(tarinfo=folder_info)

    def add_file(self, file_path: str, content: bytes | str) -> None:
        self._assert_not_finalized()
        if isinstance(content, str):
            content = content.encode()
        file_data = io.BytesIO(content)
        file_info = tarfile.TarInfo(name=file_path)
        file_info.size = len(file_data.getvalue())
        file_info.mode = 0o644
        self.tar.addfile(tarinfo=file_info, fileobj=file_data)

    def add_json(self, file_path: str, data: Any) -> None:
        json_content = json.dumps(data, indent=4)
        self.add_file(file_path, json_content)

    def finalize(self) -> bytes:
        self.tar.close()
        self.tar_buffer.seek(0)
        self.finalized = True
        return self.tar_buffer.getvalue()
