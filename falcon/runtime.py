from typing import Any, Union, Dict, List, Type, Optional
import onnxruntime as ort
import numpy as np
from abc import ABC, abstractmethod
import bson
from bson import BSON


class BaseRuntime(ABC):
    @abstractmethod
    def run(self, X: np.ndarray, **kwargs: Any) -> Any:
        pass


class ONNXRuntime(BaseRuntime):
    """
    Runtime for ONNX models. This runtime can only run onnx models produced by falcon.
    """
    def __init__(self, model: Union[bytes, str]):
        self.ort_session = ort.InferenceSession(model)

    def run(self, X: np.ndarray, outputs: str = "final", **kwargs: Any) -> List[np.ndarray]:
        """
        Runs the model.

        Parameters
        ----------
        X : np.ndarray
            model 
        outputs : str, optional
            when set to "all", all onnx output nodes will be returned; when "final" only the last layer outputs are returned, by default "final"

        Returns
        -------
        List[np.ndarray]
            model predictions
        """
        if outputs not in ["all", "final"]:
            raise ValueError(
                f"Expected `outputs` to be one of [all, final], got `{outputs}`."
            )

        inputs = self._get_inputs(X)
        output_names = self._get_output_names(outputs)

        return self.ort_session.run(output_names, inputs)

    def _get_inputs(self, X: np.ndarray) -> Dict:
        ort_inputs = self.ort_session.get_inputs()
        inputs = {}

        for i, inp in enumerate(ort_inputs):
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
            if len(ort_inputs) > 1:
                inputs[str(inp.name)] = np.expand_dims(X[:, i], 1).astype(dtype)
            else:
                inputs[str(inp.name)] = X.astype(dtype)
        return inputs

    def _get_output_names(self, outputs: str = "final") -> List[str]:
        ort_outputs = self.ort_session.get_outputs()
        output_names = [o.name for o in ort_outputs]
        if outputs == "final" and len(ort_outputs) > 1:
            idx_ = []
            for name in output_names:
                if not name[0:10] == "falcon_pl_":
                    raise RuntimeError("One of the output nodes has an invalid name.")
                idx = int(name.split("/")[0][10:])
                idx_.append(idx)
            max_idx = max(idx_)
            output_names = [
                n for n in output_names if n.startswith(f"falcon_pl_{max_idx}")
            ]
        return output_names


class FalconRuntime(BaseRuntime):
    """
    Runtime for falcon models.
    This runtime is added for future compatibility in case some models will not be onnx convertible. 
    Ideally, it will not have to be used.
    """
    def __init__(self, model: Union[bson.BSON, str, bytes]):
        """
        Parameters
        ----------
        model : Union[bson.BSON, str, bytes]
            Serialized model or filepath
        """

        if isinstance(model, str) and model.endswith(".falcon"):
            with (open(model, "rb")) as f:
                model = f.read()

        self.decoded_model: Dict = bson.BSON.decode(model) # type: ignore

    def run(self, X: np.ndarray, **kwargs: Any) -> Union[List, np.ndarray]:
        """
        Runs the model.

        Parameters
        ----------
        X : np.ndarray
            model inputs
            

        Returns
        -------
        Union[List, np.ndarray]
            model prediction
        """
        for i, node in enumerate(self.decoded_model["nodes"]):
            if node["type"] == "onnx":
                X = self._run_onnx_node(node, X, self._get_n_inputs_node(i + 1))
            else:
                raise RuntimeError(f"Unknown node type {node['type']}")
        if len(X) == 1:
            X = X[0]
        if len(X.shape) > 1:
            X = np.squeeze(X)
        return X

    def _get_n_inputs_node(self, i: int) -> Optional[int]:
        if len(self.decoded_model["nodes"]) > i:
            return self.decoded_model["nodes"][i]["n_inputs"]
        else:
            return None

    def _run_onnx_node(
        self, node: Dict, X: np.ndarray, n_inputs_next: Optional[int] = None
    ) -> np.ndarray:
        ort_sess = ort.InferenceSession(node["model"])
        ort_inputs = ort_sess.get_inputs()
        inputs = {}
        for oi, inp in enumerate(ort_inputs):
            dtype = self._get_np_type(node["initial_types"][oi])
            if len(ort_sess.get_inputs()) > 1:
                inputs[str(inp.name)] = np.expand_dims(X[:, oi], 1).astype(dtype)
            else:
                if len(X.shape) > len(node["initial_shapes"][oi]):
                    X = X.squeeze()
                elif len(X.shape) < len(node["initial_shapes"][oi]):
                    X = np.expand_dims(X, 1)
                inputs[str(inp.name)] = X.astype(dtype)
        outputs = [o.name for o in ort_sess.get_outputs()]
        print("next >> ", n_inputs_next)
        if n_inputs_next is not None:
            outputs = outputs[:n_inputs_next]
        pred = ort_sess.run(outputs, inputs)
        if len(pred) == 1:
            pred = pred[0]
        else:
            print(len(pred))
            print(pred)
            raise NotImplementedError(  # TODO
                "Multiple outputs from node is not yet supported."
            )
        return pred

    def _get_np_type(self, type_: str) -> Type:
        dtype: Type
        if type_ == "FLOAT32":
            dtype = np.float32
        elif type_ == "STRING":
            dtype = np.str_
        elif type_ == "INT64":
            dtype = np.int64
        else:
            RuntimeError("The model has an unsupported input type")
        return dtype
