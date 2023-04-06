from typing import Any, Union, Dict, List, Type, Optional
try:
    import onnxruntime as ort
except (ImportError, ModuleNotFoundError):
    print("ONNXRuntime is not installed. Inference modules will not work.")
    ort = None
import numpy as np
from abc import ABC, abstractmethod


class BaseRuntime(ABC):
    @abstractmethod
    def run(self, X: np.ndarray, **kwargs: Any) -> Any:
        pass


class ONNXRuntime(BaseRuntime):
    """
    Runtime for ONNX models. This runtime can only run onnx models produced by falcon.
    """

    def __init__(self, model: Union[bytes, str]):
        if ort is None:
            raise RuntimeError(
                "ONNXRuntime is not installed. Please install it with `pip install onnxruntime`."
            )
        self.ort_session = ort.InferenceSession(model)

    def run(
        self, X: np.ndarray, outputs: str = "final", **kwargs: Any
    ) -> List[np.ndarray]:
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
                if not name[0:10] in ("falcon_pl_", "falcon-pl-"):
                    raise RuntimeError("One of the output nodes has an invalid name.")
                idx = int(name.split("/")[0][10:])
                idx_.append(idx)
            max_idx = max(idx_)
            output_names = [
                n
                for n in output_names
                if (
                    n.startswith(f"falcon_pl_{max_idx}")
                    or n.startswith(f"falcon-pl-{max_idx}")
                )
            ]
        return output_names
