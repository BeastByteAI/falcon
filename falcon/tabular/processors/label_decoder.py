from concurrent.futures import process
from multiprocessing.dummy import Process
from falcon.abstract import Processor, ONNXConvertible, PipelineElement
from typing import Any, Type, Union
from numpy.typing import NDArray
import numpy as np
from sklearn.preprocessing import LabelEncoder
from falcon.types import Float32Array, Int64Array
from skl2onnx import convert_sklearn
from onnx import TensorProto, helper as h, OperatorSetIdProto
from skl2onnx.common.data_types import StringTensorType
from falcon.config import ML_ONNX_OPSET_VERSION
from numpy import typing as npt
from falcon.serialization import SerializedModelRepr


class LabelDecoder(Processor, ONNXConvertible):
    """
    Label encoder/decoder to be used for encoding labels as integers and vice versa.
    """
    def __init__(self) -> None:
        """
        does not take any arguments
        """
        self.le = LabelEncoder()

    def fit_pipe(self, X: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # Do nothing
        """
        Since label decoder should initially be fitted and applied before the main training phase of pipeline, this method does nothing. 

        Parameters
        ----------
        X : Any
            dummy argument
        y : Any
            dummy argument
        """
        return

    def fit(self, X: npt.NDArray, y: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Fits the decoder.

        Parameters
        ----------
        X : npt.NDArray
            labels to be encoded as integers
        y : Any, optional
            dummy argument, by default None
        """
        self.le.fit(X)

    def predict(self, X: npt.NDArray, inverse: bool = True, *args: Any, **kwargs: Any) -> npt.NDArray:
        """
        Equivalent of `.transform()`.

        Parameters
        ----------
        X : npt.NDArray
            labels
        inverse : bool, optional
            if True, encode strings as integers, else convert integers back to strings, by default True

        Returns
        -------
        npt.NDArray
            encoded/decoded labels
        """
        return self.transform(X, inverse=inverse)

    def transform(self, X: npt.NDArray, inverse: bool = True, *args: Any, **kwargs: Any) -> npt.NDArray:
        """
        Encodes/decodes the labels.

        Parameters
        ----------
        X : npt.NDArray
            labels
        inverse : bool, optional
            if True, encode strings as integers, else convert integers back to strings, by default True

        Returns
        -------
        npt.NDArray
            encoded/decoded labels
        """
        if not inverse:
            return self.le.transform(X)
        else:
            return self.le.inverse_transform(X.astype(np.int64)).astype(np.str_)

    def get_input_type(self) -> Type:
        """
        Returns
        -------
        Type
            Int64Array
        """
        return Int64Array

    def get_output_type(self) -> Type:
        """
        Returns
        -------
        Type
            NDArray[str]
        """
        return NDArray[np.str_]

    def to_onnx(self) -> SerializedModelRepr:
        """
        Serializes the encoder to onnx. 

        Returns
        -------
        SerializedModelRepr
        """
        inputs = [h.make_tensor_value_info("encoded_labels", TensorProto.INT64, [None])]
        outputs = [
            h.make_tensor_value_info("decoded_labels", TensorProto.STRING, [None])
        ]
        node = h.make_node(
            "LabelEncoder",
            ["encoded_labels"],
            ["decoded_labels"],
            values_strings=[str(el) for el in self.le.classes_],
            keys_int64s=[int(i) for i in range(len(self.le.classes_))],
            name=f"labels_decoder",
            domain="ai.onnx.ml",
        )
        graph = h.make_graph([node], f"decoder", inputs, outputs)
        op = h.make_operatorsetid("ai.onnx.ml", ML_ONNX_OPSET_VERSION)
        model = h.make_model(graph, producer_name="falcon", opset_imports = [op])
        return SerializedModelRepr(model, 1, 1, ["INT64"], [[None]])

    def forward(
        self, X: npt.NDArray, *args: Any, **kwargs: Any
    ) -> npt.NDArray:  # Inside pipeline used as post-processor to decode labels back to strings
        """
        Equivalent to `.transform(X, inverse=True)`.

        Parameters
        ----------
        X : npt.NDArray
            labels to decode

        Returns
        -------
        npt.NDArray
            labels decoded to strings
        """
        return self.transform(X, inverse=True)
