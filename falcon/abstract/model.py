from abc import ABC, abstractmethod
from numpy import typing as npt
from typing import Any


class Model(ABC):
    """
    Base class for all models.
    """

    @abstractmethod
    def fit(self, X: npt.NDArray, y: npt.NDArray) -> Any:
        """

        Parameters
        ----------
        X : npt.NDArray
            features
        y : npt.NDArray
            targets

        Returns
        -------
        Any
            usually `None`
        """
        pass

    @abstractmethod
    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """
        Parameters
        ----------
        X : npt.NDArray
            features

        Returns
        -------
        npt.NDArray
            predictions
        """
        pass


class TransformerMixin:
    def transform(self, X: npt.NDArray) -> npt.NDArray:
        """
        Equivalent of `self.predict(X)`

        Parameters
        ----------
        X : npt.NDArray
            features

        Returns
        -------
        npt.NDArray
            transformed features
        """
        return self.predict(X)  # type: ignore
