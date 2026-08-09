from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy import typing as npt
from skl2onnx import update_registered_converter
from skl2onnx.common._apply_operation import apply_concat
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.proto import onnx_proto
from sklearn.base import clone
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.validation import check_is_fitted

from falcon.tabular.splitting import cross_validation_indices
from falcon.types import TargetKind


class FalconTargetEncoder(TargetEncoder):
    def fit_transform(
        self, X: npt.NDArray[Any], y: npt.NDArray[Any]
    ) -> npt.NDArray[np.float64]:
        return self.fit(X, y).transform(X)

    def cross_fit_transform(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        split_features: npt.NDArray[Any],
        target_kind: TargetKind,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        check_is_fitted(
            self, ("categories_", "encodings_", "target_mean_", "target_type_")
        )
        values = np.asarray(X)
        targets = np.asarray(y).reshape(-1)
        if values.ndim != 2 or values.shape[0] != targets.shape[0]:
            raise ValueError("Target-encoder inputs must contain one target per row")

        transformed = np.empty(
            (values.shape[0], len(self.encodings_)), dtype=np.float64
        )
        task = f"tabular_{target_kind}"
        splits = cross_validation_indices(
            split_features,
            targets,
            task=task,
            groups=groups,
            n_splits=int(self.cv),
        )
        for train_indices, validation_indices in splits:
            fold_encoder = cast(FalconTargetEncoder, clone(self))
            fold_encoder.fit(values[train_indices], targets[train_indices])
            if hasattr(self, "classes_") and not np.array_equal(
                fold_encoder.classes_, self.classes_
            ):
                raise ValueError(
                    "Every target-encoding fold must contain all target classes"
                )
            fold_values = fold_encoder.transform(values[validation_indices])
            if fold_values.shape[1] != transformed.shape[1]:
                raise ValueError(
                    "Target-encoding folds produced inconsistent feature counts"
                )
            transformed[validation_indices] = fold_values
        return transformed


def _target_encoder_shape_calculator(operator: Any) -> None:
    encoder: FalconTargetEncoder = operator.raw_operator
    check_is_fitted(encoder, ("categories_", "encodings_"))
    batch_size = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = FloatTensorType([batch_size, len(encoder.encodings_)])


def _feature_input_name(
    scope: Any,
    container: Any,
    input_name: str,
    feature_index: int,
    feature_count: int,
) -> str:
    if feature_count == 1:
        return input_name
    index_name = scope.get_unique_variable_name("target_encoder_feature_index")
    container.add_initializer(
        index_name,
        onnx_proto.TensorProto.INT64,
        [],
        [feature_index],
    )
    feature_name = scope.get_unique_variable_name("target_encoder_feature")
    container.add_node(
        "ArrayFeatureExtractor",
        [input_name, index_name],
        [feature_name],
        name=scope.get_unique_operator_name("target_encoder_feature"),
        op_domain="ai.onnx.ml",
        op_version=1,
    )
    return feature_name


def _target_encoder_converter(scope: Any, operator: Any, container: Any) -> None:
    encoder: FalconTargetEncoder = operator.raw_operator
    check_is_fitted(
        encoder, ("categories_", "encodings_", "target_mean_", "target_type_")
    )
    class_count = len(encoder.classes_) if encoder.target_type_ == "multiclass" else 1
    if len(encoder.encodings_) != len(encoder.categories_) * class_count:
        raise RuntimeError("Target encoder has inconsistent fitted mappings")

    target_means = np.asarray(encoder.target_mean_, dtype=np.float32).reshape(-1)
    encoded_outputs: list[str] = []
    input_name = operator.inputs[0].full_name
    for feature_index, categories in enumerate(encoder.categories_):
        feature_name = _feature_input_name(
            scope,
            container,
            input_name,
            feature_index,
            len(encoder.categories_),
        )
        for class_index in range(class_count):
            encoding_index = feature_index * class_count + class_index
            output_name = scope.get_unique_variable_name("target_encoded_feature")
            encoded_outputs.append(output_name)
            container.add_node(
                "LabelEncoder",
                [feature_name],
                [output_name],
                name=scope.get_unique_operator_name("target_encoder_mapping"),
                op_domain="ai.onnx.ml",
                op_version=2,
                keys_strings=np.asarray(
                    [str(category).encode("utf-8") for category in categories]
                ),
                values_floats=np.asarray(
                    encoder.encodings_[encoding_index], dtype=np.float32
                ),
                default_float=float(target_means[class_index]),
            )

    output_name = operator.outputs[0].full_name
    if len(encoded_outputs) == 1:
        container.add_node(
            "Identity",
            encoded_outputs,
            [output_name],
            name=scope.get_unique_operator_name("target_encoder_output"),
            op_domain="",
        )
    else:
        apply_concat(scope, encoded_outputs, output_name, container, axis=1)


update_registered_converter(
    FalconTargetEncoder,
    "FalconTargetEncoder",
    _target_encoder_shape_calculator,
    _target_encoder_converter,
)
