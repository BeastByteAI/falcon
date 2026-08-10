from typing import Any

from numpy import typing as npt

from falcon.abstract.task_pipeline import Pipeline, PipelineStep
from falcon.tabular.processors.label_decoder import LabelDecoder
from falcon.tabular.processors.multi_modal_encoder import MultiModalEncoder
from falcon.tabular.processors.scaler_and_encoder import ScalerAndEncoder
from falcon.types import DatasetSchema
from falcon.utils import logger


class SimpleTabularPipeline(Pipeline):
    """
    Default tabular pipeline.
    """

    def __init__(
        self,
        task: str,
        dataset_size: tuple[int, ...],
        learner: type[Any],
        schema: DatasetSchema | None = None,
        learner_kwargs: dict[str, Any] | None = None,
        preprocessor: str = "MultiModalEncoder",
        impute_missing: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(task=task, dataset_size=dataset_size, schema=schema)

        self.preprocessor = preprocessor
        self.impute_missing = impute_missing
        self.learner = learner
        self.learner_kwargs = learner_kwargs
        self.labels_transformer: LabelDecoder | None = None

    def _reset(self) -> None:
        self.clear_steps()
        encoder: PipelineStep
        if self.preprocessor == "MultiModalEncoder":
            encoder = MultiModalEncoder(impute_missing=self.impute_missing)
        else:
            encoder = ScalerAndEncoder(impute_missing=self.impute_missing)

        self.add_step(encoder)

        if not self.learner_kwargs:
            learner_kwargs = {}
        else:
            learner_kwargs = self.learner_kwargs
        learner = self.learner(
            task=self.task, dataset_size=self.dataset_size, **learner_kwargs
        )
        self.add_step(learner)

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None:
        logger.info("Fitting the pipeline...")
        self._reset()
        if self.task == "tabular_classification":
            self.labels_transformer = LabelDecoder()
            self.labels_transformer.fit(X, y, schema, groups=groups)
            y = self.labels_transformer.encode(y)
        super().fit(X, y, schema, groups=groups)
        if self.labels_transformer is not None:
            self.add_step(self.labels_transformer)
