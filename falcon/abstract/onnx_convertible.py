from abc import ABC, abstractmethod
from falcon.serialization import SerializedModelRepr


class ONNXConvertible(ABC):
    """
    Base class for all models/pipeline_elements/pipelines that can be converted to onnx.
    """
    @abstractmethod
    def to_onnx(self) -> SerializedModelRepr:
        """
        Converted model

        Returns
        -------
        SerializedModelRepr
        """
        pass
