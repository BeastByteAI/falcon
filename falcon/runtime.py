from typing import Any, Union, Dict, List, Type, Optional

try:
    from phonnx.runtime import Runtime as _PhonnxRuntime
except (ImportError, ModuleNotFoundError):
    print("ONNXRuntime/PHONNX is not installed. Inference modules will not work.")
    _PhonnxRuntime = None
import numpy as np


class ONNXRuntime:
    """
    Runtime for ONNX models based on PHONNX.
    """

    def __init__(self, model: Union[bytes, str]):
        if _PhonnxRuntime is None:
            raise ImportError("PHONNX is not installed.")
        self.runtime = _PhonnxRuntime(model=model)

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

        return self.runtime.run(X, outputs_to_return=outputs)