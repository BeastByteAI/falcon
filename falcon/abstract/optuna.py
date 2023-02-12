from abc import abstractmethod, ABC
from typing import Any, Union, Callable, Dict
from typing_extensions import Protocol
from numpy import typing as npt

class OptunaMixin(ABC):

    """
    Abstract mixin that should be used in order to indicate the compatibility of the model with OptunaLearner. 
    """
    @classmethod
    @abstractmethod
    def get_search_space(cls, X: Any, y: Any) -> Union[Callable, Dict]:
        """
        A class method that provides an optuna search space for the model. 
        Optionally, the search space can be adjusted based on the provided training data.

        Parameters
        ----------
        X : Any
            features
        y : Any
            targets

        Returns
        -------
        Union[Callable, Dict]
            dictionary that describes the search space, or custom objective function
        """
        pass