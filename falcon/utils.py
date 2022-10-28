from typing import List, Tuple, Optional
from numpy import typing as npt
from onnx import ModelProto, load_from_string
import onnxruntime as ort
from onnx.compose import add_prefix, merge_models
from onnx import OperatorSetIdProto
from onnx.helper import make_model
from falcon.config import ONNX_OPSET_VERSION
import numpy as np
from typing import Any, Dict, Union
import bson
from bson import BSON
import onnx
from onnx import TensorProto, helper as h, OperatorSetIdProto
from falcon.types import ModelsList
import os
import sys
import warnings


def serialize_to_onnx(models_: ModelsList) -> onnx.ModelProto:
    print_("Serializing to onnx...")
    if len(models_) == 0:
        raise ValueError("List of models cannot be empty")
    updated_models: List[ModelProto] = []
    models = [load_from_string(m[0][0]) for m in models_]
    for i, model in enumerate(models):
        op1 = h.make_operatorsetid("", ONNX_OPSET_VERSION)
        op2 = h.make_operatorsetid("ai.onnx.ml", 2)
        updated_model = make_model(model.graph, opset_imports=[op1, op2])
        updated_model = add_prefix(updated_model, prefix=f"m{i}/")
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
    decoded_model: Dict = bson.BSON.decode(model)
    for i, node in enumerate(decoded_model["nodes"]):
        inputs = {}
        n_next_inputs: Optional[int] = None
        if i + 1 < len(decoded_model["nodes"]):
            n_next_inputs = decoded_model["nodes"][i + 1]["n_inputs"]
        if node["type"] == "onnx":
            ort_sess = ort.InferenceSession(node["model"])
        for oi, inp in enumerate(ort_sess.get_inputs()):
            dtype: Any
            if node["initial_types"][oi] == "FLOAT32":
                dtype = np.float32
            elif node["initial_types"][oi] == "STRING":
                dtype = np.str_
            elif node["initial_types"][oi] == "INT64":
                dtype = np.int64
            else:
                RuntimeError("The model has an unsupported input type")
            if not isinstance(X, np.ndarray):
                RuntimeError(
                    f"Wrong input type: Numpy array was expected, got {type(X)}"
                )
            if len(ort_sess.get_inputs()) > 1:
                inputs[str(inp.name)] = np.expand_dims(X[:, oi], 1).astype(dtype)
            else:
                if not isinstance(X, np.ndarray):
                    X = np.asarray(X)
                if len(X.shape) > len(decoded_model["nodes"][i]["initial_shapes"][oi]):
                    X = X.squeeze()
                elif len(X.shape) < len(
                    decoded_model["nodes"][i]["initial_shapes"][oi]
                ):
                    X = np.expand_dims(X, 1)
                inputs[str(inp.name)] = X.astype(dtype)
        outputs = [o.name for o in ort_sess.get_outputs()]
        if n_next_inputs is not None:
            outputs = outputs[:n_next_inputs]
        pred = ort_sess.run(outputs, inputs)
        X = pred
        if len(X) == 1:
            X = X[0]
    if len(X) == 1:
        X = X[0]
    if len(X.shape) > 1:
        X = np.squeeze(X)
    return X


def run_onnx(
    model: Union[bytes, str], X: npt.NDArray, outputs: str = "final"
) -> npt.NDArray:
    if outputs not in ["all", "final"]:
        raise ValueError(
            f"Expected `outputs` to be one of [all, final], got `{outputs}`."
        )
    ort_sess = ort.InferenceSession(model)
    inputs = {}
    for i, inp in enumerate(ort_sess.get_inputs()):
        dtype: Any
        if str(inp.type) == "tensor(float)":
            dtype = np.float32
        elif str(inp.type) == "tensor(string)":
            dtype = np.str_
        elif "int" in str(inp.type).lower():
            dtype = np.int64
        else:
            RuntimeError(
                f"The model input type should be one of [str, int, float], got f{inp.type}"
            )
        if len(ort_sess.get_inputs()) > 1:
            inputs[str(inp.name)] = np.expand_dims(X[:, i], 1).astype(dtype)
        else:
            inputs[str(inp.name)] = X.astype(dtype)

    output_names = [o.name for o in ort_sess.get_outputs()]
    if outputs == "final" and len(ort_sess.get_outputs()) > 1:
        idx_ = []
        for name in output_names:
            if not name[0] == "m":
                raise RuntimeError("One of the output nodes has an invalid name.")
            idx = int(name.split("/")[0][1:])
            idx_.append(idx)
        max_idx = max(idx_)
        output_names = [n for n in output_names if n.startswith(f"m{max_idx}")]
    pred_onnx = ort_sess.run(output_names, inputs)

    return pred_onnx


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