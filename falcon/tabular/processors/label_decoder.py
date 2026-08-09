from typing import Any

import numpy as np
from numpy import typing as npt
from numpy.typing import NDArray
from onnx import TensorProto
from onnx import helper as h
from sklearn.preprocessing import LabelEncoder

from falcon.config import ML_ONNX_OPSET_VERSION
from falcon.serialization import SerializedModelRepr
from falcon.types import DatasetSchema, Int64Array


class LabelDecoder:
    def __init__(self) -> None:
        self.le = LabelEncoder()

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None:
        self.le.fit(y)

    def encode(self, labels: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
        return self.le.transform(labels)

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Decodes integer predictions back to the fitted labels."""
        return self.le.inverse_transform(X.astype(np.int64)).astype(np.str_)

    def get_input_type(self) -> object:
        return Int64Array

    def get_output_type(self) -> object:
        return NDArray[np.str_]

    def serialize(self) -> SerializedModelRepr:
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
            name="labels_decoder",
            domain="ai.onnx.ml",
        )
        graph = h.make_graph([node], "decoder", inputs, outputs)
        op = h.make_operatorsetid("ai.onnx.ml", ML_ONNX_OPSET_VERSION)
        model = h.make_model(graph, producer_name="falcon", opset_imports=[op])
        return SerializedModelRepr(model, 1, 1, ["INT64"], [[None]])
