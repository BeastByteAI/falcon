from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from falcon import Predictor, RunConfig
from falcon.config import DATASET_AWARE_ORDERING_DEFAULT
from falcon.constants import TABULAR_CLASSIFICATION_TASK, TABULAR_REGRESSION_TASK
from falcon.runtime import Runtime

logger = logging.getLogger("falcon.benchmarks")


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    openml_id: int
    task: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dataset name must not be empty")
        if isinstance(self.openml_id, bool) or self.openml_id < 1:
            raise ValueError("OpenML dataset ID must be positive")
        if self.task not in {
            TABULAR_CLASSIFICATION_TASK,
            TABULAR_REGRESSION_TASK,
        }:
            raise ValueError(f"Unknown benchmark task `{self.task}`")


DATASETS = (
    BenchmarkDataset("credit-g", 31, TABULAR_CLASSIFICATION_TASK),
    BenchmarkDataset("segment", 36, TABULAR_CLASSIFICATION_TASK),
    BenchmarkDataset("diabetes", 37, TABULAR_CLASSIFICATION_TASK),
    BenchmarkDataset("spambase", 44, TABULAR_CLASSIFICATION_TASK),
    BenchmarkDataset("vehicle", 54, TABULAR_CLASSIFICATION_TASK),
    BenchmarkDataset("kc1", 1067, TABULAR_CLASSIFICATION_TASK),
    BenchmarkDataset("phoneme", 1489, TABULAR_CLASSIFICATION_TASK),
    BenchmarkDataset("adult", 1590, TABULAR_CLASSIFICATION_TASK),
    BenchmarkDataset("cpu-act", 197, TABULAR_REGRESSION_TASK),
    BenchmarkDataset("wine-quality", 287, TABULAR_REGRESSION_TASK),
)

BenchmarkScalar = str | int | float | bool | None
BenchmarkResult = dict[str, BenchmarkScalar]


class BenchmarkReport(TypedDict):
    schema_version: int
    preset: str
    dataset_aware_ordering: bool
    compare_dataset_ordering: bool
    ordering_gate_passed: bool | None
    time_limit_seconds: float | None
    random_state: int
    test_size: float
    inference_repeats: int
    autogluon_baseline: str | None
    results: list[BenchmarkResult]


def load_openml_dataset(
    dataset: BenchmarkDataset,
) -> tuple[pd.DataFrame, pd.Series[Any]]:
    features, targets = fetch_openml(
        data_id=dataset.openml_id,
        as_frame=True,
        return_X_y=True,
        parser="auto",
    )
    if not isinstance(features, pd.DataFrame) or not isinstance(targets, pd.Series):
        raise TypeError(
            f"OpenML dataset {dataset.openml_id} did not return a single-target frame"
        )

    feature_names = [str(name) for name in features.columns]
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(
            f"OpenML dataset {dataset.openml_id} has duplicate feature names"
        )
    normalized_features = features.copy()
    normalized_features.columns = feature_names
    normalized_targets = targets.copy()
    normalized_targets.name = str(targets.name or "target")
    return normalized_features, normalized_targets


def split_dataset(
    dataset: BenchmarkDataset,
    features: pd.DataFrame,
    targets: pd.Series[Any],
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series[Any], pd.Series[Any]]:
    stratify = targets if dataset.task == TABULAR_CLASSIFICATION_TASK else None
    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return (
        train_features.reset_index(drop=True),
        test_features.reset_index(drop=True),
        train_targets.reset_index(drop=True),
        test_targets.reset_index(drop=True),
    )


def score_predictions(
    dataset: BenchmarkDataset,
    targets: pd.Series[Any] | npt.ArrayLike,
    predictions: npt.ArrayLike,
) -> tuple[str, float]:
    if dataset.task == TABULAR_CLASSIFICATION_TASK:
        score = metrics.balanced_accuracy_score(
            np.asarray(targets).astype(np.str_),
            np.asarray(predictions).reshape(-1).astype(np.str_),
        )
        return "balanced_accuracy", float(score)

    score = metrics.root_mean_squared_error(
        np.asarray(targets, dtype=np.float64),
        np.asarray(predictions, dtype=np.float64).reshape(-1),
    )
    return "rmse", float(score)


def measure_artifact_latency(
    runtime: Runtime,
    features: pd.DataFrame,
    repeats: int,
) -> float:
    runtime.predict(features)
    durations: list[float] = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        runtime.predict(features)
        durations.append(time.perf_counter() - started_at)
    return float(statistics.median(durations))


def load_autogluon_predictor_type() -> type[Any] | None:
    try:
        from autogluon.tabular import TabularPredictor
    except ModuleNotFoundError as error:
        if error.name is not None and not error.name.startswith("autogluon"):
            raise
        return None
    return TabularPredictor


def benchmark_autogluon(
    predictor_type: type[Any],
    dataset: BenchmarkDataset,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_targets: pd.Series[Any],
    test_targets: pd.Series[Any],
    *,
    workspace: Path,
    time_limit: float | None,
) -> float:
    label = "__falcon_benchmark_target__"
    while label in train_features.columns:
        label = f"_{label}"
    train_data = train_features.copy()
    train_data[label] = train_targets.to_numpy()
    problem_type = "regression"
    if dataset.task == TABULAR_CLASSIFICATION_TASK:
        problem_type = "binary" if train_targets.nunique() == 2 else "multiclass"
    predictor = predictor_type(
        label=label,
        problem_type=problem_type,
        path=str(workspace / f"autogluon-{dataset.openml_id}"),
        verbosity=0,
    )
    fit_options: dict[str, Any] = {
        "train_data": train_data,
        "presets": "medium_quality",
    }
    if time_limit is not None:
        fit_options["time_limit"] = time_limit
    predictor.fit(**fit_options)
    _, score = score_predictions(
        dataset, test_targets, predictor.predict(test_features)
    )
    return score


def benchmark_dataset(
    dataset: BenchmarkDataset,
    *,
    workspace: Path,
    preset: str,
    time_limit: float | None,
    random_state: int,
    test_size: float,
    inference_repeats: int,
    autogluon_predictor_type: type[Any] | None,
    dataset_aware_ordering: bool = DATASET_AWARE_ORDERING_DEFAULT,
) -> BenchmarkResult:
    features, targets = load_openml_dataset(dataset)
    train_features, test_features, train_targets, test_targets = split_dataset(
        dataset,
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
    )

    started_at = time.perf_counter()
    predictor = Predictor(
        dataset.task,
        preset=preset,
        config=RunConfig(dataset_aware_ordering=dataset_aware_ordering),
        time_limit=time_limit,
        random_state=random_state,
        eval_strategy=None,
    ).fit((train_features, train_targets))
    wall_time = time.perf_counter() - started_at
    metric, score = score_predictions(
        dataset,
        test_targets,
        predictor.predict(test_features),
    )

    artifact_path = workspace / f"falcon-{dataset.openml_id}.fnnx"
    artifact = predictor.save(artifact_path)
    runtime = Runtime(str(artifact_path))
    result: BenchmarkResult = {
        "status": "ok",
        "dataset": dataset.name,
        "openml_id": dataset.openml_id,
        "task": dataset.task,
        "dataset_aware_ordering": dataset_aware_ordering,
        "metric": metric,
        "score": score,
        "wall_time_seconds": float(wall_time),
        "artifact_size_bytes": len(artifact),
        "artifact_inference_latency_seconds": measure_artifact_latency(
            runtime,
            test_features,
            inference_repeats,
        ),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
    }
    if autogluon_predictor_type is not None:
        result["autogluon_score"] = benchmark_autogluon(
            autogluon_predictor_type,
            dataset,
            train_features,
            test_features,
            train_targets,
            test_targets,
            workspace=workspace,
            time_limit=time_limit,
        )
    return result


def _relative_ordering_improvement(
    task: str,
    static_score: float,
    dataset_aware_score: float,
) -> float:
    improvement = dataset_aware_score - static_score
    if task == TABULAR_REGRESSION_TASK:
        improvement = -improvement
    scale = max(abs(static_score), float(np.finfo(np.float64).eps))
    return improvement / scale


def benchmark_dataset_ordering(
    dataset: BenchmarkDataset,
    *,
    workspace: Path,
    preset: str,
    time_limit: float | None,
    random_state: int,
    test_size: float,
    inference_repeats: int,
    autogluon_predictor_type: type[Any] | None,
) -> BenchmarkResult:
    static_result = benchmark_dataset(
        dataset,
        workspace=workspace,
        preset=preset,
        time_limit=time_limit,
        random_state=random_state,
        test_size=test_size,
        inference_repeats=inference_repeats,
        autogluon_predictor_type=None,
        dataset_aware_ordering=False,
    )
    dataset_aware_result = benchmark_dataset(
        dataset,
        workspace=workspace,
        preset=preset,
        time_limit=time_limit,
        random_state=random_state,
        test_size=test_size,
        inference_repeats=inference_repeats,
        autogluon_predictor_type=autogluon_predictor_type,
        dataset_aware_ordering=True,
    )
    static_score = float(cast(float, static_result["score"]))
    dataset_aware_score = float(cast(float, dataset_aware_result["score"]))
    dataset_aware_result.update(
        {
            "static_score": static_score,
            "dataset_aware_score": dataset_aware_score,
            "ordering_relative_improvement": _relative_ordering_improvement(
                dataset.task,
                static_score,
                dataset_aware_score,
            ),
            "static_wall_time_seconds": static_result["wall_time_seconds"],
        }
    )
    return dataset_aware_result


def ordering_quality_gate(results: Sequence[BenchmarkResult]) -> bool:
    if not results or any(result.get("status") != "ok" for result in results):
        return False
    improvements: list[float] = []
    for result in results:
        value = result.get("ordering_relative_improvement")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return False
        improvements.append(float(value))
    return statistics.fmean(improvements) >= 0


def _validate_run_settings(
    datasets: Sequence[BenchmarkDataset],
    preset: str,
    time_limit: float | None,
    test_size: float,
    inference_repeats: int,
    dataset_aware_ordering: bool,
    compare_dataset_ordering: bool,
) -> None:
    if not datasets:
        raise ValueError("At least one dataset must be selected")
    if not preset:
        raise ValueError("preset must not be empty")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be a positive finite number")
    if not math.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    if (
        isinstance(inference_repeats, bool)
        or not isinstance(inference_repeats, int)
        or inference_repeats < 1
    ):
        raise ValueError("inference_repeats must be at least 1")
    if not isinstance(dataset_aware_ordering, bool):
        raise ValueError("dataset_aware_ordering must be a boolean")
    if not isinstance(compare_dataset_ordering, bool):
        raise ValueError("compare_dataset_ordering must be a boolean")


def _write_report(path: Path, report: BenchmarkReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def run_benchmarks(
    output_path: Path,
    *,
    datasets: Sequence[BenchmarkDataset] = DATASETS,
    preset: str = "balanced",
    time_limit: float | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
    inference_repeats: int = 5,
    include_autogluon: bool = False,
    dataset_aware_ordering: bool = DATASET_AWARE_ORDERING_DEFAULT,
    compare_dataset_ordering: bool = False,
) -> BenchmarkReport:
    _validate_run_settings(
        datasets,
        preset,
        time_limit,
        test_size,
        inference_repeats,
        dataset_aware_ordering,
        compare_dataset_ordering,
    )
    autogluon_predictor_type = (
        load_autogluon_predictor_type() if include_autogluon else None
    )
    if include_autogluon and autogluon_predictor_type is None:
        logger.warning("AutoGluon is not installed; its baseline will be skipped.")
    report: BenchmarkReport = {
        "schema_version": 2,
        "preset": preset,
        "dataset_aware_ordering": dataset_aware_ordering,
        "compare_dataset_ordering": compare_dataset_ordering,
        "ordering_gate_passed": None,
        "time_limit_seconds": time_limit,
        "random_state": random_state,
        "test_size": test_size,
        "inference_repeats": inference_repeats,
        "autogluon_baseline": (
            "medium_quality"
            if autogluon_predictor_type is not None
            else "unavailable"
            if include_autogluon
            else None
        ),
        "results": [],
    }
    _write_report(output_path, report)

    with tempfile.TemporaryDirectory(prefix="falcon-benchmark-") as temporary_dir:
        workspace = Path(temporary_dir)
        for index, dataset in enumerate(datasets, start=1):
            logger.info(
                "Benchmarking %s (%d/%d)",
                dataset.name,
                index,
                len(datasets),
            )
            try:
                benchmark = (
                    benchmark_dataset_ordering
                    if compare_dataset_ordering
                    else benchmark_dataset
                )
                benchmark_options: dict[str, Any] = {
                    "workspace": workspace,
                    "preset": preset,
                    "time_limit": time_limit,
                    "random_state": random_state,
                    "test_size": test_size,
                    "inference_repeats": inference_repeats,
                    "autogluon_predictor_type": autogluon_predictor_type,
                }
                if not compare_dataset_ordering:
                    benchmark_options["dataset_aware_ordering"] = dataset_aware_ordering
                result = benchmark(
                    dataset,
                    **benchmark_options,
                )
            except Exception as error:
                logger.exception("Benchmark failed for %s", dataset.name)
                result = {
                    "status": "failed",
                    "dataset": dataset.name,
                    "openml_id": dataset.openml_id,
                    "task": dataset.task,
                    "error": f"{type(error).__name__}: {error}",
                }
            report["results"].append(result)
            if compare_dataset_ordering:
                report["ordering_gate_passed"] = ordering_quality_gate(
                    report["results"]
                )
            _write_report(output_path, report)
    return report


def _positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return value


def _test_fraction(raw_value: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError("expected a number between zero and one")
    return value


def _positive_integer(raw_value: str) -> int:
    value = int(raw_value)
    if value < 1:
        raise argparse.ArgumentTypeError("expected an integer of at least 1")
    return value


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Falcon OpenML benchmark.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results.json"),
    )
    parser.add_argument(
        "--preset", choices=("fast", "balanced", "best"), default="balanced"
    )
    parser.add_argument("--time-limit", type=_positive_float)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=_test_fraction, default=0.2)
    parser.add_argument("--inference-repeats", type=_positive_integer, default=5)
    parser.add_argument("--autogluon", action="store_true")
    parser.add_argument(
        "--dataset-aware-ordering",
        action=argparse.BooleanOptionalAction,
        default=DATASET_AWARE_ORDERING_DEFAULT,
    )
    parser.add_argument("--compare-dataset-ordering", action="store_true")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(dataset.name for dataset in DATASETS),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    selected_names = set(arguments.dataset or ())
    datasets = (
        tuple(dataset for dataset in DATASETS if dataset.name in selected_names)
        if selected_names
        else DATASETS
    )
    report = run_benchmarks(
        arguments.output,
        datasets=datasets,
        preset=arguments.preset,
        time_limit=arguments.time_limit,
        random_state=arguments.random_state,
        test_size=arguments.test_size,
        inference_repeats=arguments.inference_repeats,
        include_autogluon=arguments.autogluon,
        dataset_aware_ordering=arguments.dataset_aware_ordering,
        compare_dataset_ordering=arguments.compare_dataset_ordering,
    )
    return int(any(result["status"] == "failed" for result in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
