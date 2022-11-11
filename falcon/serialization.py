from typing import List, Optional, Dict
from falcon.utils import print_
import bson
from bson import BSON
from typing import List, Tuple, Optional
from numpy import typing as npt
from onnx import ModelProto, load_from_string
import onnxruntime as ort
from onnx.compose import add_prefix, merge_models
from onnx.helper import make_model
from falcon.config import ONNX_OPSET_VERSION, ML_ONNX_OPSET_VERSION
import numpy as np
from typing import Any, Dict, Union
import onnx
from onnx import TensorProto, helper as h, OperatorSetIdProto

class SerializedModelRepr:
    def __init__(
        self,
        model: bytes,
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

    def get_model(self) -> bytes:
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

def serialize_to_onnx(models_: List[SerializedModelRepr]) -> onnx.ModelProto:
    print_("Serializing to onnx...")
    if len(models_) == 0:
        raise ValueError("List of models cannot be empty")

    # Updating the models by resetting the opset and adding prefix to node names
    updated_models: List[ModelProto] = []
    models = [load_from_string(m.get_model()) for m in models_]
    for i, model in enumerate(models):
        op1 = h.make_operatorsetid("", ONNX_OPSET_VERSION)
        op2 = h.make_operatorsetid("ai.onnx.ml", ML_ONNX_OPSET_VERSION)
        updated_model = make_model(model.graph, opset_imports=[op1, op2])
        updated_model = add_prefix(updated_model, prefix=f"falcon_pl_{i}/")
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
    print_("Serialization completed.")
    return prev


def serialize_to_falcon(models: List[SerializedModelRepr]) -> bytes:
    print_("Serializing to falcon...")
    if len(models) == 0:
        raise ValueError("List of models cannot be empty")

    nodes = []
    for model_ in models:
        nodes.append(model_.to_dict())

    model_to_save = {"version": 1, "n_nodes": len(nodes), "nodes": nodes}
    encoded = bytes(bson.BSON.encode(model_to_save))
    print_("Serialization completed.")
    return encoded