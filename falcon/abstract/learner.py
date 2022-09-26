from abc import ABC, abstractmethod
from numpy import typing as npt
from falcon.abstract.task_pipeline import PipelineElement
from typing import Any, Optional


class Learner(PipelineElement):
    """
    Subclass of `PipelineElement`.
    Learners are task aware pipeline elements that act as wrappers around models and responsible for tuning of the hyperparameters.
    """
    def __init__(self, task: str, **kwargs: Any) -> None: 
        """
        Parameters
        ----------
        task : str
            current ML task
        """
        self.task = task

    