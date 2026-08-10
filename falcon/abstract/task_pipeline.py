from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from numpy import typing as npt

from falcon.serialization import FNNXSerializer, SerializedModelRepr
from falcon.types import DatasetSchema


@runtime_checkable
class PipelineStep(Protocol):
    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None: ...

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]: ...

    def serialize(self) -> SerializedModelRepr: ...

    def get_input_type(self) -> object: ...

    def get_output_type(self) -> object: ...


class Pipeline:
    def __init__(
        self,
        task: str,
        dataset_size: tuple[int, ...],
        schema: DatasetSchema | None = None,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.dataset_size = dataset_size
        self.schema = schema
        self._steps: list[PipelineStep] = []

    @property
    def steps(self) -> tuple[PipelineStep, ...]:
        return tuple(self._steps)

    def clear_steps(self) -> None:
        self._steps.clear()

    def add_step(self, step: PipelineStep) -> None:
        if step is self:
            raise ValueError("Cannot add the pipeline to itself")
        if not isinstance(step, PipelineStep):
            raise TypeError(
                "Pipeline steps must implement fit, transform, serialize, and type metadata"
            )
        if self._steps and step.get_input_type() != self._steps[-1].get_output_type():
            raise RuntimeError(
                "The step cannot be added to the pipeline because its input type "
                "does not match the previous output type."
            )
        self._steps.append(step)

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> None:
        if X.ndim != 2 or X.shape[1] != schema.n_features:
            raise ValueError("Feature data does not match the dataset schema")
        self.schema = schema
        transformed = X
        for step in self._steps:
            fit_transform = getattr(step, "fit_transform", None)
            if fit_transform is None:
                step.fit(transformed, y, schema, groups=groups)
                transformed = step.transform(transformed)
            else:
                transformed = fit_transform(transformed, y, schema, groups=groups)

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        transformed = X
        for step in self._steps:
            transformed = step.transform(transformed)
        return transformed

    def save(
        self,
        feature_names: list[Any] | None = None,
        producer_extra_tags: list[str] | None = None,
        schema: DatasetSchema | None = None,
    ) -> FNNXSerializer:
        serialized_steps = [step.serialize() for step in self._steps]
        resolved_schema = schema if schema is not None else self.schema
        if resolved_schema is None:
            raise RuntimeError("A dataset schema is required to save a pipeline")

        return FNNXSerializer(
            models=serialized_steps,
            task=self.task,
            init_types=list(resolved_schema.column_types),
            init_feature_names=feature_names,
            producer_extra_tags=producer_extra_tags,
            schema=resolved_schema,
        )
