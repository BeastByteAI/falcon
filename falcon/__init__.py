__version__ = "1.0.0"
__author__ = "Oleh Kostromin, Iryna Kondrashchenko"

from falcon.config import RunConfig as RunConfig
from falcon.main import AutoML as AutoML
from falcon.predictor import Predictor as Predictor

__all__ = ["AutoML", "Predictor", "RunConfig"]
