from abc import abstractmethod, ABC
from typing import Any, Union, Callable, Dict

class OptunaMixin(ABC):

    @classmethod
    @abstractmethod
    def get_search_space(cls, X: Any, y: Any) -> Union[Callable, Dict]:
        pass