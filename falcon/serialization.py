from typing import List, Optional, Dict
from copy import copy
from falcon.utils import print_
from typing import List, Tuple, Optional
from numpy import typing as npt
from onnx import ModelProto, load_from_string
from onnx.compose import add_prefix, merge_models
from onnx.helper import make_model
from falcon.config import ONNX_OPSET_VERSION, ML_ONNX_OPSET_VERSION
from falcon import __version__ as falcon_version
from falcon.types import ColumnTypes
import numpy as np
from typing import Any, Dict, Union
import onnx
from onnx import helper as h
import io
import tarfile
import json
from dataclasses import dataclass


onnx_type_map = {
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
        initial_types: List[str],
        initial_shapes: List[List[Optional[int]]],
        type_: str = "onnx",
    ):
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

    def get_initial_types(self) -> List[str]:
        return self._initial_types

    def get_initial_shapes(self) -> List[List[Optional[int]]]:
        return self._initial_shapes

    def get_type(self) -> str:
        return self._type

    def to_dict(self) -> Dict:
        return {
            "n_inputs": self._n_inputs,
            "n_outputs": self._n_outputs,
            "initial_types": self._initial_types,
            "initial_shapes": self._initial_shapes,
            "model": self._model,
            "type": self._type,
        }


def _make_self_name(name: Any) -> str:
    name = str(name)
    name = "".join([c for c in name if c.isalpha() or c.isdigit() or c == " "]).rstrip()
    name = name.replace(" ", "_")
    return name


def _rename_inputs(
    model: onnx.ModelProto, feature_names: List[Any], feature_types: List
) -> None:
    if len(feature_types) != len(model.graph.input):
        feature_types = ["9999" for _ in range(len(model.graph.input))]
    else:
        feature_types = [str(i.value) for i in feature_types]
    if len(feature_names) != len(feature_types):
        feature_names = [str(i) for i in range(len(feature_types))]
    mapping = {}
    for i, inp in enumerate(model.graph.input):
        new_name = f"falcon-input-{str(i)}_{_make_self_name(feature_names[i])}_{feature_types[i]}"
        mapping[inp.name] = new_name
        inp.name = new_name
    for node in model.graph.node:
        for key in mapping.keys():
            for ii, input in enumerate(node.input):
                if input == key:
                    node.input[ii] = mapping[key]


def serialize_to_onnx(
    models_: List[SerializedModelRepr],
    init_types: Optional[List] = None,
    init_feature_names: Optional[List] = None,
    task: Optional[str] = None,
) -> onnx.ModelProto:
    if init_types is None:
        init_types = []
    if init_feature_names is None:
        init_feature_names = []
    # print_("Serializing to onnx...")
    if len(models_) == 0:
        raise ValueError("List of models cannot be empty")

    # Updating the models by resetting the opset and adding prefix to node names
    updated_models: List[ModelProto] = []
    models = [m.get_model() for m in models_]
    for i, model in enumerate(models):
        op1 = h.make_operatorsetid("", ONNX_OPSET_VERSION)
        op2 = h.make_operatorsetid("ai.onnx.ml", ML_ONNX_OPSET_VERSION)
        op3 = h.make_operatorsetid("com.microsoft", 1)
        updated_model = make_model(model.graph, opset_imports=[op1, op2, op3])
        updated_model = add_prefix(updated_model, prefix=f"falcon-pl-{i}/")
        updated_models.append(updated_model)

    # Merging the models sequentially
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
        io_map: List[Tuple[str, str]] = []
        outputs: List[str] = [o.name for o in current.graph.output]
        for p, c in zip(prev_outputs, current_inputs):
            mapping: Tuple[str, str] = (p.name, c.name)
            io_map.append(mapping)

        # print_(f"\t -> Merging step {i} ::: io_map {io_map} ::: outputs: {outputs}")
        combined_model: ModelProto = merge_models(
            prev, current, io_map=io_map  # , outputs=outputs
        )

        prev = combined_model
    combined_model = prev
    # TODO: Rename the inputs here
    description = {}
    if task is not None:
        description["task"] = task
    _rename_inputs(combined_model, init_feature_names, init_types)
    combined_model.graph.doc_string = str(description)
    combined_model.producer_name = "Falcon ML"
    combined_model.producer_version = falcon_version
    print_("Serialization completed.")
    return combined_model


############################################################################################################
############################################# FNNX #########################################################
############################################################################################################

DEFAULT_PRODUCER_NAME = "falcon.beastbyte.ai"

input_tags = {
    ColumnTypes.NUMERIC_REGULAR: [f"{DEFAULT_PRODUCER_NAME}::numeric:v1"],
    ColumnTypes.CAT_LOW_CARD: [f"{DEFAULT_PRODUCER_NAME}::categorical_lc:v1"],
    ColumnTypes.CAT_HIGH_CARD: [f"{DEFAULT_PRODUCER_NAME}::categorical_hc:v1"],
    ColumnTypes.TEXT_UTF8: [f"{DEFAULT_PRODUCER_NAME}::text:v1"],
    ColumnTypes.DATE_YMD_ISO8601: [f"{DEFAULT_PRODUCER_NAME}::date_ymd_iso8601:v1"],
    ColumnTypes.DATETIME_YMDHMS_ISO8601: [
        "{DEFAULT_PRODUCER_NAME}::datetime_ymdhms_iso8601:v1"
    ],
}


@dataclass
class ModelIO:
    name: str
    dtype: str
    shape: List[Union[int, str]]
    tags: Optional[List[str]] = None


class FNNXSerializer:

    out_names = {
        "tabular_classification": ["probabilities", "y_pred"],
        "tabular_regression": ["y_pred"],
    }

    artifact_dirs = ["meta_artifacts", "ops_artifacts", "variant_artifacts"]

    def __init__(
        self,
        models: List[SerializedModelRepr],
        init_types: Optional[List] = None,
        init_feature_names: Optional[List] = None,
        task: Optional[str] = None,
        producer_name: Optional[str] = "falcon.beastbyte.ai",
        producer_version: Optional[str] = falcon_version,
        producer_extra_tags: Optional[List[str]] = None,
        description: str = "",
    ):
        self.models = models
        self.init_types = init_types

        self.task = task
        self.description = description

        self.model_proto = serialize_to_onnx(
            models, init_types, init_feature_names, task
        )

        self.init_feature_names = init_feature_names or [
            f"input_{i}" for i in range(len(self.model_proto.graph.input))
        ]

        self.inputs = self._map_io(
            self.model_proto.graph.input, self.init_feature_names
        )
        self.outputs = self._map_io(
            self.model_proto.graph.output, self.out_names.get(task)
        )

        if init_types is not None and len(init_types) == len(self.inputs):
            for i, t in enumerate(init_types):
                self.inputs[i].tags = copy(input_tags.get(t, []))

        self.producer_name = producer_name
        self.producer_version = producer_version
        self.producer_extra_tags = producer_extra_tags or []
        self.metadata_container = {
            "id": "falcon_metrics",
            "producer": "falcon.beastbyte.ai",
            "producer_version": falcon_version,
            "producer_tags": [f"falcon.beastbyte.ai::{self.task}_metrics:v1"],
        }
        self.metadata_payload = {}
        self.metadata_container["payload"] = self.metadata_payload
        self.fh = TarHandler()

    def _map_io(self, io, names) -> List[ModelIO]:
        processed = []
        for el, name in zip(io, names):
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
        self._add_manifest()  # always start with manifest
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

    def _add_manifest(self):
        manifest = {
            "variant": "pipeline",
            "description": self.description,
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
            "producer_tags": [f"falcon.beastbyte.ai::{self.task}:v1"]
            + self.producer_extra_tags,
            "inputs": [],
            "outputs": [],
            "dynamic_attributes": [],
            "env_vars": [],
        }

        for io in self.inputs:
            inp = {
                "name": io.name,
                "content_type": "NDJSON",
                "dtype": io.dtype,
                "shape": io.shape,
            }
            if io.tags:
                inp["tags"] = io.tags
            manifest["inputs"].append(inp)

        for io in self.outputs:
            manifest["outputs"].append(
                {
                    "name": io.name,
                    "content_type": "NDJSON",
                    "dtype": io.dtype,
                    "shape": io.shape,
                }
            )

        self.fh.add_json("manifest.json", manifest)

    def _add_variant_config(self):
        config = {
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

    def _add_onnx(self):
        op = [
            {
                "id": "onnx_main",
                "op": "ONNX_v1",
                "inputs": [{"dtype": i.dtype, "shape": i.shape} for i in self.inputs],
                "outputs": [{"dtype": o.dtype, "shape": o.shape} for o in self.outputs],
                "dynamic_attributes": {},
                "attributes": {
                    "opsets": [
                        {"domain": "ai.onnx", "version": ONNX_OPSET_VERSION},
                        {"domain": "com.microsoft", "version": 1},
                        {"domain": "ai.onnx.ml", "version": ML_ONNX_OPSET_VERSION},
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

    def _add_env(self):
        self.fh.add_json("env.json", {})  # TODO


class TarHandler:
    def __init__(self):
        self.tar_buffer = io.BytesIO()
        self.tar = tarfile.open(fileobj=self.tar_buffer, mode="w")
        self.finalized = False

    def _assert_not_finalized(self):
        if self.finalized:
            raise RuntimeError("Tar file is already finalized.")

    def add_directory(self, directory_name):
        self._assert_not_finalized()
        if not directory_name.endswith("/"):
            directory_name += "/"
        folder_info = tarfile.TarInfo(name=directory_name)
        folder_info.type = tarfile.DIRTYPE
        folder_info.mode = 0o755
        self.tar.addfile(tarinfo=folder_info)

    def add_file(self, file_path, content):
        self._assert_not_finalized()
        if isinstance(content, str):
            content = content.encode()
        file_data = io.BytesIO(content)
        file_info = tarfile.TarInfo(name=file_path)
        file_info.size = len(file_data.getvalue())
        file_info.mode = 0o644
        self.tar.addfile(tarinfo=file_info, fileobj=file_data)

    def add_json(self, file_path, data):
        json_content = json.dumps(data, indent=4)
        self.add_file(file_path, json_content)

    def finalize(self):
        self.tar.close()
        self.tar_buffer.seek(0)
        self.finalized = True
        return self.tar_buffer.getvalue()
