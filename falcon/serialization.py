from typing import List, Optional, Dict
from falcon.utils import print_
from typing import List, Tuple, Optional
from numpy import typing as npt
from onnx import ModelProto, load_from_string
from onnx.compose import add_prefix, merge_models
from onnx.helper import make_model
from falcon.config import ONNX_OPSET_VERSION, ML_ONNX_OPSET_VERSION
from falcon import __version__ as falcon_version
import numpy as np
from typing import Any, Dict, Union
import onnx
from onnx import TensorProto, helper as h, OperatorSetIdProto


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
    # print(feature_types, len(model.graph.input))
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
    print_("Serializing to onnx...")
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
        prev_outputs: List = prev.graph.output
        current_inputs: List = current.graph.input
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

        print_(f"\t -> Merging step {i} ::: io_map {io_map} ::: outputs: {outputs}")
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
