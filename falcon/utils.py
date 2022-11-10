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
from falcon.types import ModelsList
import os
import sys
import warnings
import bson
from bson import BSON
from falcon.runtime import ONNXRuntime, FalconRuntime


def serialize_to_onnx(models_: ModelsList) -> onnx.ModelProto:
    print_("Serializing to onnx...")
    if len(models_) == 0:
        raise ValueError("List of models cannot be empty")
    updated_models: List[ModelProto] = []
    models = [load_from_string(m[0][0]) for m in models_]
    for i, model in enumerate(models):
        op1 = h.make_operatorsetid("", ONNX_OPSET_VERSION)
        op2 = h.make_operatorsetid("ai.onnx.ml", ML_ONNX_OPSET_VERSION)
        updated_model = make_model(model.graph, opset_imports=[op1, op2])
        updated_model = add_prefix(updated_model, prefix=f"falcon_pl_{i}/")
        updated_models.append(updated_model)
        # onnx.save(updated_model, f"{i}_model.onnx")

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


def make_node(
    n_inputs: int,
    n_outputs: int,
    initial_types: List[str],
    initial_shapes: List[List[Optional[int]]],
    model: bytes,
    type_: str = "onnx",
) -> Dict:
    return {
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "initial_types": initial_types,
        "initial_shapes": initial_shapes,
        "model": model,
        "type": type_,
    }


def serialize_to_falcon(models: ModelsList) -> bytes:
    print_("Serializing to falcon...")
    if len(models) == 0:
        raise ValueError("List of models cannot be empty")

    nodes = []
    for model_, type_ in models:
        node = make_node(
            n_inputs=model_[1],
            n_outputs=model_[2],
            initial_types=model_[3],
            initial_shapes=model_[4],
            model=model_[0],
            type_=type_,
        )
        nodes.append(node)

    model_to_save = {"version": 1, "n_nodes": len(nodes), "nodes": nodes}
    encoded = bytes(bson.BSON.encode(model_to_save))
    print_("Serialization completed.")
    return encoded


def run_falcon(model: BSON, X: npt.NDArray) -> npt.NDArray:
    runtime = FalconRuntime(model=model)
    return runtime.run(X)


def run_onnx(
    model: Union[bytes, str], X: npt.NDArray, outputs: str = "final"
) -> npt.NDArray:
    runtime = ONNXRuntime(model=model)
    return runtime.run(X, outputs=outputs)


def set_verbosity_level(level: int = 1) -> None:
    if level not in {0, 1}:
        level = 0
    os.environ["FALCON_VERBOSITY_LEVEL"] = str(level)


def print_(*args: Any) -> None:
    verbosity_level = os.getenv("FALCON_VERBOSITY_LEVEL", "1")
    if verbosity_level == "1":
        for a in args:
            print(a)


# TODO RUN TESTS
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__ # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter Notebook
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal
        else:
            return True  # Other
    except NameError:
        return False

def disable_warnings() -> None: 
    if not sys.warnoptions:
        warnings.simplefilter('ignore')
        os.environ["PYTHONWARNINGS"] = 'ignore'