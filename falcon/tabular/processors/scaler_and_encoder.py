from typing import Any

import numpy as np
from numpy import typing as npt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from falcon.addons.sklearn.preprocessing.missing_values import (
    MissingStringImputer,
    NumericCast,
    NumericMedianImputer,
    StringCast,
)
from falcon.addons.sklearn.preprocessing.target_encoder import FalconTargetEncoder
from falcon.config import ML_ONNX_OPSET_VERSION, ONNX_OPSET_VERSION
from falcon.serialization import SerializedModelRepr
from falcon.types import ColumnTypes, DatasetSchema, Float32Array, TargetKind


class ScalerAndEncoder:
    """
    One-hot encodes low cardinality categoricals, target encodes high cardinality ones
    and standard scales numeric features.

    With `impute_missing` disabled, missing values are not filled in and the exported
    graph contains no data dependent branching; numeric features must then be complete
    at fit time and missing categories become ordinary categories.
    """

    def __init__(self, impute_missing: bool = True) -> None:
        self.impute_missing = impute_missing
        self.column_types: tuple[ColumnTypes, ...] = ()
        self.ct: ColumnTransformer

    def _get_string_imputer(self, fill_value: str) -> BaseEstimator:
        if not self.impute_missing:
            return StringCast()
        return MissingStringImputer(fill_value=fill_value)

    def _get_ohe(self) -> BaseEstimator:
        not_sparse = {"sparse_output": False}

        method = SKLPipeline(
            steps=[
                ("imputer", self._get_string_imputer("__falcon_missing__")),
                (
                    "ohe",
                    OneHotEncoder(
                        categories="auto", handle_unknown="ignore", **not_sparse
                    ),
                ),
            ]
        )
        return method

    def _get_numeric_scaler(self) -> BaseEstimator:
        imputer = NumericMedianImputer() if self.impute_missing else NumericCast()
        return SKLPipeline(
            steps=[
                ("imputer", imputer),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ]
        )

    def _get_target_encoder(self, target_kind: TargetKind) -> BaseEstimator:
        target_type = "continuous" if target_kind == "regression" else "auto"
        return SKLPipeline(
            steps=[
                ("imputer", self._get_string_imputer("__falcon_missing__")),
                (
                    "target_encoder",
                    FalconTargetEncoder(
                        target_type=target_type,
                        cv=5,
                        shuffle=True,
                        random_state=42,
                    ),
                ),
            ]
        )

    def _get_ordinal_encoder(self) -> BaseEstimator:
        return SKLPipeline(
            steps=[
                ("imputer", self._get_string_imputer("__falcon_missing__")),
                (
                    "ord_enc",
                    OrdinalEncoder(
                        categories="auto",
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                ("sc", MaxAbsScaler()),
            ]
        )

    def _align_numeric_scalers_with_onnx(self) -> None:
        for index, column_type in enumerate(self.column_types):
            if column_type != ColumnTypes.NUMERIC_REGULAR:
                continue
            pipeline = self.ct.named_transformers_[f"input {index}"]
            scaler = pipeline.named_steps["scaler"]
            # Float32 statistics keep tree split decisions identical after export.
            scaler.mean_ = scaler.mean_.astype(np.float32)
            scaler.scale_ = scaler.scale_.astype(np.float32)

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
        self.column_types = schema.column_types
        transformers = []

        for i, v in enumerate(self.column_types):
            if v == ColumnTypes.CAT_LOW_CARD:
                method = self._get_ohe()
            elif v == ColumnTypes.CAT_HIGH_CARD:
                method = self._get_target_encoder(schema.target_kind)
            elif v == ColumnTypes.NUMERIC_REGULAR:
                method = self._get_numeric_scaler()
            else:
                method = self._get_ordinal_encoder()
            t = (f"input {i}", method, [i])
            transformers.append(t)

        self.ct = ColumnTransformer(transformers)
        self.ct.fit(X, y)
        self._align_numeric_scalers_with_onnx()

    def fit_transform(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        schema: DatasetSchema,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> Float32Array:
        self.fit(X, y, schema, groups=groups)
        transformed = np.asarray(self.ct.transform(X), dtype=np.float32)
        for index, column_type in enumerate(self.column_types):
            if column_type != ColumnTypes.CAT_HIGH_CARD:
                continue
            transformer_name = f"input {index}"
            fitted_pipeline = self.ct.named_transformers_[transformer_name]
            imputed = fitted_pipeline.named_steps["imputer"].transform(X[:, [index]])
            target_encoder = fitted_pipeline.named_steps["target_encoder"]
            cross_fitted = target_encoder.cross_fit_transform(
                imputed,
                y,
                X,
                schema.target_kind,
                groups=groups,
            )
            transformed[:, self.ct.output_indices_[transformer_name]] = cross_fitted
        return transformed

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self.ct.transform(X).astype(dtype=np.float32)

    def get_input_type(self) -> object:
        return npt.NDArray[np.object_]

    def get_output_type(self) -> object:
        return Float32Array

    def serialize(self) -> SerializedModelRepr:
        """
        Each feature of the original dataset becomes its own onnx input node,
        `float32` for numeric features and `string` for everything else.
        """
        initial_types = []
        initial_types_str: list[str] = []
        initial_shapes: list[list[int | None]] = []
        for i, t in enumerate(self.column_types):
            if t in [ColumnTypes.NUMERIC_REGULAR]:
                tensor = FloatTensorType([None, 1])
                initial_types_str.append("FLOAT32")
            else:
                tensor = StringTensorType([None, 1])
                initial_types_str.append("STRING")
            initial_types.append((f"input{i}", tensor))
            initial_shapes.append([None, 1])
        return SerializedModelRepr(
            convert_sklearn(
                self.ct,
                initial_types=initial_types,
                target_opset={
                    "": ONNX_OPSET_VERSION,
                    "ai.onnx.ml": ML_ONNX_OPSET_VERSION,
                },
                options={
                    StandardScaler: {"div": "div"},
                },
            ),
            len(self.column_types),
            1,
            initial_types_str,
            initial_shapes,
        )
