from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import numpy as np
import onnx
from numpy import typing as npt
from onnx import TensorProto, helper

from falcon.config import ONNX_OPSET_VERSION
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.serialization import SerializedModelRepr
from falcon.utils import logger

_ONNXMLTOOLS_TARGET_OPSET = 15


class GBDTModel(Protocol):
    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None: ...

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]: ...

    def serialize(self) -> SerializedModelRepr: ...


class GBDTClassifierModel(GBDTModel, Protocol):
    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]: ...


class _OnnxmltoolsGBDT:
    def __init__(
        self,
        estimator: Any,
        converter_name: str,
        prediction_dtype: npt.DTypeLike,
    ) -> None:
        self.estimator = estimator
        self._converter_name = converter_name
        self._prediction_dtype = prediction_dtype
        self._shape: list[int | None] | None = None

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self._shape = [None, *X.shape[1:]]
        fit_options: dict[str, Any] = {}
        if sample_weight is not None:
            fit_options["sample_weight"] = sample_weight
        if validation_data is not None:
            if early_stopping_rounds is None:
                raise ValueError(
                    "early_stopping_rounds is required with validation_data"
                )
            fit_options["eval_set"] = [validation_data]
            if self._converter_name == "convert_lightgbm":
                fit_options["callbacks"] = [
                    import_module("lightgbm").early_stopping(
                        early_stopping_rounds,
                        verbose=False,
                    )
                ]
            else:
                self.estimator.set_params(early_stopping_rounds=early_stopping_rounds)
                fit_options["verbose"] = False
        elif early_stopping_rounds is not None:
            raise ValueError("validation_data is required for early stopping")
        self.estimator.fit(X, y, **fit_options)

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.asarray(self.estimator.predict(X), dtype=self._prediction_dtype)

    def serialize(self) -> SerializedModelRepr:
        if self._shape is None:
            raise RuntimeError("The model must be fitted before it can be serialized")

        converter = getattr(import_module("onnxmltools"), self._converter_name)
        float_tensor_type = import_module(
            "onnxmltools.convert.common.data_types"
        ).FloatTensorType
        conversion_options: dict[str, object] = {
            "initial_types": [("model_input", float_tensor_type(self._shape))],
            # onnxmltools currently rejects newer core opsets even though these
            # converters emit only older, compatible operators.
            "target_opset": _ONNXMLTOOLS_TARGET_OPSET,
        }
        if self._converter_name == "convert_lightgbm":
            conversion_options["zipmap"] = False
        model = converter(self.estimator, **conversion_options)
        return SerializedModelRepr(
            model,
            len(model.graph.input),
            len(model.graph.output),
            ["FLOAT32"],
            [self._shape],
        )


class _OnnxmltoolsClassifier(_OnnxmltoolsGBDT):
    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float32)

    def serialize(self) -> SerializedModelRepr:
        serialized = super().serialize()
        label_shape = serialized.get_model().graph.output[0].type.tensor_type.shape
        if label_shape.dim and label_shape.dim[0].dim_value == 1:
            label_shape.dim[0].ClearField("dim_value")
        return serialized


class LightGBMClassifier(_OnnxmltoolsClassifier):
    def __init__(self, random_state: int = 42, **parameters: Any) -> None:
        defaults: dict[str, Any] = {
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": -1,
        }
        defaults.update(parameters)
        estimator = import_module("lightgbm").LGBMClassifier(**defaults)
        super().__init__(estimator, "convert_lightgbm", np.int64)


class LightGBMRegressor(_OnnxmltoolsGBDT):
    def __init__(self, random_state: int = 42, **parameters: Any) -> None:
        defaults: dict[str, Any] = {
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": -1,
        }
        defaults.update(parameters)
        estimator = import_module("lightgbm").LGBMRegressor(**defaults)
        super().__init__(estimator, "convert_lightgbm", np.float32)


class XGBoostClassifier(_OnnxmltoolsClassifier):
    def __init__(self, random_state: int = 42, **parameters: Any) -> None:
        defaults: dict[str, Any] = {
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": 0,
        }
        defaults.update(parameters)
        estimator = import_module("xgboost").XGBClassifier(**defaults)
        super().__init__(estimator, "convert_xgboost", np.int64)


class XGBoostRegressor(_OnnxmltoolsGBDT):
    def __init__(self, random_state: int = 42, **parameters: Any) -> None:
        defaults: dict[str, Any] = {
            "n_jobs": 1,
            "random_state": random_state,
            "verbosity": 0,
        }
        defaults.update(parameters)
        estimator = import_module("xgboost").XGBRegressor(**defaults)
        super().__init__(estimator, "convert_xgboost", np.float32)


class _CatBoostGBDT:
    def __init__(self, estimator: Any, prediction_dtype: npt.DTypeLike) -> None:
        self.estimator = estimator
        self._prediction_dtype = prediction_dtype
        self._shape: list[int | None] | None = None

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self._shape = [None, *X.shape[1:]]
        fit_options: dict[str, Any] = {}
        if sample_weight is not None:
            fit_options["sample_weight"] = sample_weight
        if validation_data is not None:
            if early_stopping_rounds is None:
                raise ValueError(
                    "early_stopping_rounds is required with validation_data"
                )
            fit_options.update(
                {
                    "early_stopping_rounds": early_stopping_rounds,
                    "eval_set": validation_data,
                    "use_best_model": True,
                }
            )
        elif early_stopping_rounds is not None:
            raise ValueError("validation_data is required for early stopping")
        self.estimator.fit(X, y, **fit_options)

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.asarray(
            self.estimator.predict(X), dtype=self._prediction_dtype
        ).reshape(-1)

    def serialize(self) -> SerializedModelRepr:
        if self._shape is None:
            raise RuntimeError("The model must be fitted before it can be serialized")

        with TemporaryDirectory(prefix="falcon-catboost-") as directory:
            model_path = Path(directory) / "model.onnx"
            self.estimator.save_model(str(model_path), format="onnx")
            model = onnx.load(model_path)
        return SerializedModelRepr(
            model,
            len(model.graph.input),
            len(model.graph.output),
            ["FLOAT32"],
            [self._shape],
        )


class CatBoostClassifier(_CatBoostGBDT):
    def __init__(self, random_state: int = 42, **parameters: Any) -> None:
        defaults: dict[str, Any] = {
            "allow_writing_files": False,
            "random_seed": random_state,
            "thread_count": 1,
            "verbose": False,
        }
        defaults.update(parameters)
        estimator = import_module("catboost").CatBoostClassifier(**defaults)
        super().__init__(estimator, np.int64)
        self._class_count: int | None = None

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        sample_weight: npt.NDArray[np.float64] | None = None,
        validation_data: tuple[npt.NDArray[Any], npt.NDArray[Any]] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self._class_count = len(np.unique(y))
        if self._class_count > 2:
            raise ValueError(
                "CatBoost multiclass models are disabled because ONNX export parity "
                "is not established"
            )
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            validation_data=validation_data,
            early_stopping_rounds=early_stopping_rounds,
        )

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float32)

    def serialize(self) -> SerializedModelRepr:
        serialized = super().serialize()
        model = serialized.get_model()
        zipmap = next(
            (node for node in model.graph.node if node.op_type == "ZipMap"), None
        )
        if zipmap is None or self._class_count is None:
            raise RuntimeError(
                "CatBoost did not export the expected probability output"
            )

        probability_name = zipmap.input[0]
        model.graph.node.remove(zipmap)
        label_name = "falcon_argmax_label"
        model.graph.node.append(
            helper.make_node(
                "ArgMax",
                [probability_name],
                [label_name],
                axis=1,
                keepdims=0,
                name="falcon_catboost_argmax",
            )
        )
        del model.graph.output[:]
        model.graph.output.extend(
            [
                helper.make_tensor_value_info(label_name, TensorProto.INT64, [None]),
                helper.make_tensor_value_info(
                    probability_name,
                    TensorProto.FLOAT,
                    [None, self._class_count],
                ),
            ]
        )
        if not any(opset.domain in {"", "ai.onnx"} for opset in model.opset_import):
            model.opset_import.append(helper.make_operatorsetid("", ONNX_OPSET_VERSION))
        return serialized


class CatBoostRegressor(_CatBoostGBDT):
    def __init__(self, random_state: int = 42, **parameters: Any) -> None:
        defaults: dict[str, Any] = {
            "allow_writing_files": False,
            "random_seed": random_state,
            "thread_count": 1,
            "verbose": False,
        }
        defaults.update(parameters)
        estimator = import_module("catboost").CatBoostRegressor(**defaults)
        super().__init__(estimator, np.float32)


@dataclass(frozen=True)
class GBDTModelFamily:
    classifier: type[GBDTClassifierModel]
    regressor: type[GBDTModel]


def _module_is_available(module_name: str) -> bool:
    try:
        import_module(module_name)
    except ImportError:
        return False
    return True


def _discover_gbdt_families() -> dict[str, GBDTModelFamily]:
    families: dict[str, GBDTModelFamily] = {}
    if _module_is_available("onnxmltools") and _module_is_available("lightgbm"):
        families["lightgbm"] = GBDTModelFamily(
            LightGBMClassifier,
            LightGBMRegressor,
        )
    if _module_is_available("onnxmltools") and _module_is_available("xgboost"):
        families["xgboost"] = GBDTModelFamily(
            XGBoostClassifier,
            XGBoostRegressor,
        )
    if _module_is_available("catboost"):
        families["catboost"] = GBDTModelFamily(
            CatBoostClassifier,
            CatBoostRegressor,
        )
    return families


_GBDT_FAMILIES = _discover_gbdt_families()


def get_gbdt_model_classes(
    task: str,
    *,
    n_classes: int | None = None,
) -> dict[str, type[GBDTModel]]:
    if task == TABULAR_CLASSIFICATION_TASK:
        classes: dict[str, type[GBDTModel]] = {
            name: family.classifier for name, family in _GBDT_FAMILIES.items()
        }
        if n_classes is not None and n_classes > 2 and "catboost" in classes:
            del classes["catboost"]
            logger.info(
                "CatBoost is excluded from multiclass classification because ONNX "
                "export parity is not established."
            )
        return classes
    if task == TABULAR_REGRESSION_TASK:
        return {name: family.regressor for name, family in _GBDT_FAMILIES.items()}
    raise ValueError(f"Unknown task `{task}`")


__all__ = [
    "CatBoostClassifier",
    "CatBoostRegressor",
    "GBDTClassifierModel",
    "GBDTModel",
    "GBDTModelFamily",
    "LightGBMClassifier",
    "LightGBMRegressor",
    "XGBoostClassifier",
    "XGBoostRegressor",
    "get_gbdt_model_classes",
]
