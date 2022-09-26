from abc import ABC, abstractmethod
from falcon.types import SerializedModelTuple


class ONNXConvertible(ABC):
    """
    Base class for all models/pipeline_elements/pipelines that can be converted to onnx.
    """
    @abstractmethod
    def to_onnx(self) -> SerializedModelTuple:
        """
        Converted model

        Returns
        -------
        SerializedModelTuple
            Tuple of (Converted model serialized to string, number of input nodes, number of output nodes, list of initial types (one per input node), list of initial shapes (one per input node)).
        """
        pass
