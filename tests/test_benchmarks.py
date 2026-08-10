from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from benchmarks import run as benchmark_runner
from falcon.config import RunConfig


class _FakePredictor:
    initialization: dict[str, object] = {}
    ordering_modes: list[bool] = []

    def __init__(
        self,
        task: str,
        preset: str,
        config: RunConfig,
        time_limit: float | None,
        random_state: int,
        eval_strategy: None,
    ) -> None:
        self.initialization = {
            "task": task,
            "preset": preset,
            "dataset_aware_ordering": config.dataset_aware_ordering,
            "time_limit": time_limit,
            "random_state": random_state,
            "eval_strategy": eval_strategy,
        }
        type(self).initialization = self.initialization
        type(self).ordering_modes.append(config.dataset_aware_ordering)

    def fit(
        self,
        data: tuple[pd.DataFrame, pd.Series[Any]],
    ) -> _FakePredictor:
        assert len(data[0]) == len(data[1])
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.str_]]:
        return np.where(data["feature"].to_numpy() % 2 == 0, "even", "odd")

    def save(self, path: str | Path) -> bytes:
        artifact = b"fake-fnnx-artifact"
        Path(path).write_bytes(artifact)
        return artifact


class _FakeRuntime:
    prediction_calls = 0

    def __init__(self, model_path: str) -> None:
        assert Path(model_path).read_bytes() == b"fake-fnnx-artifact"

    def predict(self, data: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.str_]]:
        type(self).prediction_calls += 1
        return np.where(data["feature"].to_numpy() % 2 == 0, "even", "odd")


class _FakeAutoGluonPredictor:
    initialization: dict[str, object] = {}
    fit_options: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).initialization = kwargs

    def fit(self, **kwargs: object) -> _FakeAutoGluonPredictor:
        type(self).fit_options = kwargs
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.str_]]:
        return np.where(data["feature"].to_numpy() % 2 == 0, "even", "odd")


def test_fixed_suite_has_ten_openml_datasets_for_both_tasks() -> None:
    assert len(benchmark_runner.DATASETS) == 10
    assert len({dataset.openml_id for dataset in benchmark_runner.DATASETS}) == 10
    assert {dataset.task for dataset in benchmark_runner.DATASETS} == {
        "tabular_classification",
        "tabular_regression",
    }


def test_benchmark_dataset_records_falcon_and_optional_autogluon_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    features = pd.DataFrame({"feature": np.arange(40)})
    targets = pd.Series(
        np.where(features["feature"] % 2 == 0, "even", "odd"),
        name="target",
    )
    dataset = benchmark_runner.BenchmarkDataset(
        "parity",
        123,
        "tabular_classification",
    )
    monkeypatch.setattr(
        benchmark_runner,
        "load_openml_dataset",
        lambda requested: (features, targets),
    )
    monkeypatch.setattr(benchmark_runner, "Predictor", _FakePredictor)
    monkeypatch.setattr(benchmark_runner, "Runtime", _FakeRuntime)
    _FakeRuntime.prediction_calls = 0
    _FakePredictor.ordering_modes = []

    result = benchmark_runner.benchmark_dataset(
        dataset,
        workspace=tmp_path,
        preset="fast",
        time_limit=7.5,
        random_state=17,
        test_size=0.25,
        inference_repeats=3,
        autogluon_predictor_type=_FakeAutoGluonPredictor,
        dataset_aware_ordering=True,
    )

    assert result["status"] == "ok"
    assert result["metric"] == "balanced_accuracy"
    assert result["score"] == pytest.approx(1.0)
    assert result["autogluon_score"] == pytest.approx(1.0)
    assert result["artifact_size_bytes"] == len(b"fake-fnnx-artifact")
    assert float(cast(float, result["wall_time_seconds"])) >= 0.0
    assert float(cast(float, result["artifact_inference_latency_seconds"])) >= 0.0
    assert result["train_rows"] == 30
    assert result["test_rows"] == 10
    assert _FakeRuntime.prediction_calls == 4
    assert _FakePredictor.initialization == {
        "task": "tabular_classification",
        "preset": "fast",
        "dataset_aware_ordering": True,
        "time_limit": 7.5,
        "random_state": 17,
        "eval_strategy": None,
    }
    assert result["dataset_aware_ordering"] is True
    assert _FakeAutoGluonPredictor.fit_options["presets"] == "medium_quality"
    assert _FakeAutoGluonPredictor.fit_options["time_limit"] == 7.5


def test_regression_score_is_rmse() -> None:
    dataset = benchmark_runner.BenchmarkDataset(
        "regression",
        456,
        "tabular_regression",
    )

    metric, score = benchmark_runner.score_predictions(
        dataset,
        pd.Series([1.0, 2.0, 3.0]),
        np.asarray([1.0, 4.0, 3.0]),
    )

    assert metric == "rmse"
    assert score == pytest.approx(np.sqrt(4.0 / 3.0))


def test_ordering_improvement_uses_the_task_metric_direction() -> None:
    assert benchmark_runner._relative_ordering_improvement(
        "tabular_classification",
        static_score=0.8,
        dataset_aware_score=0.9,
    ) == pytest.approx(0.125)
    assert benchmark_runner._relative_ordering_improvement(
        "tabular_regression",
        static_score=2.0,
        dataset_aware_score=1.0,
    ) == pytest.approx(0.5)


def test_ordering_comparison_runs_both_modes_and_applies_quality_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    features = pd.DataFrame({"feature": np.arange(40)})
    targets = pd.Series(
        np.where(features["feature"] % 2 == 0, "even", "odd"),
        name="target",
    )
    dataset = benchmark_runner.BenchmarkDataset(
        "ordering",
        321,
        "tabular_classification",
    )
    monkeypatch.setattr(
        benchmark_runner,
        "load_openml_dataset",
        lambda requested: (features, targets),
    )
    monkeypatch.setattr(benchmark_runner, "Predictor", _FakePredictor)
    monkeypatch.setattr(benchmark_runner, "Runtime", _FakeRuntime)
    _FakePredictor.ordering_modes = []

    result = benchmark_runner.benchmark_dataset_ordering(
        dataset,
        workspace=tmp_path,
        preset="balanced",
        time_limit=None,
        random_state=11,
        test_size=0.25,
        inference_repeats=1,
        autogluon_predictor_type=None,
    )

    assert _FakePredictor.ordering_modes == [False, True]
    assert result["static_score"] == pytest.approx(1.0)
    assert result["dataset_aware_score"] == pytest.approx(1.0)
    assert result["ordering_relative_improvement"] == pytest.approx(0.0)
    assert benchmark_runner.ordering_quality_gate([result])
    assert not benchmark_runner.ordering_quality_gate(
        [
            {
                "status": "ok",
                "ordering_relative_improvement": -0.01,
            }
        ]
    )
    assert not benchmark_runner.ordering_quality_gate(
        [{"status": "failed", "ordering_relative_improvement": 1.0}]
    )


def test_runner_persists_successes_and_failures_as_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    datasets = (
        benchmark_runner.BenchmarkDataset(
            "working",
            1,
            "tabular_classification",
        ),
        benchmark_runner.BenchmarkDataset(
            "broken",
            2,
            "tabular_regression",
        ),
    )

    def fake_benchmark(
        dataset: benchmark_runner.BenchmarkDataset,
        **kwargs: object,
    ) -> benchmark_runner.BenchmarkResult:
        assert kwargs["autogluon_predictor_type"] is None
        if dataset.name == "broken":
            raise RuntimeError("download unavailable")
        return {
            "status": "ok",
            "dataset": dataset.name,
            "openml_id": dataset.openml_id,
            "task": dataset.task,
            "metric": "balanced_accuracy",
            "score": 0.75,
            "wall_time_seconds": 1.0,
            "artifact_size_bytes": 100,
            "artifact_inference_latency_seconds": 0.01,
            "train_rows": 8,
            "test_rows": 2,
        }

    monkeypatch.setattr(benchmark_runner, "benchmark_dataset", fake_benchmark)
    monkeypatch.setattr(
        benchmark_runner,
        "load_autogluon_predictor_type",
        lambda: None,
    )
    output_path = tmp_path / "nested" / "results.json"

    report = benchmark_runner.run_benchmarks(
        output_path,
        datasets=datasets,
        preset="balanced",
        time_limit=None,
        random_state=42,
        test_size=0.2,
        inference_repeats=5,
        include_autogluon=True,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert report["autogluon_baseline"] == "unavailable"
    assert report["dataset_aware_ordering"] is False
    assert report["ordering_gate_passed"] is None
    results = report["results"]
    assert isinstance(results, list)
    assert results[0]["status"] == "ok"
    assert results[1] == {
        "status": "failed",
        "dataset": "broken",
        "openml_id": 2,
        "task": "tabular_regression",
        "error": "RuntimeError: download unavailable",
    }


def test_runner_records_dataset_ordering_comparison_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    datasets = (
        benchmark_runner.BenchmarkDataset(
            "first",
            11,
            "tabular_classification",
        ),
        benchmark_runner.BenchmarkDataset(
            "second",
            12,
            "tabular_regression",
        ),
    )
    compared: list[str] = []

    def fake_comparison(
        dataset: benchmark_runner.BenchmarkDataset,
        **kwargs: object,
    ) -> benchmark_runner.BenchmarkResult:
        assert kwargs["autogluon_predictor_type"] is None
        compared.append(dataset.name)
        return {
            "status": "ok",
            "dataset": dataset.name,
            "ordering_relative_improvement": 0.01,
        }

    monkeypatch.setattr(
        benchmark_runner,
        "benchmark_dataset_ordering",
        fake_comparison,
    )
    output_path = tmp_path / "ordering.json"

    report = benchmark_runner.run_benchmarks(
        output_path,
        datasets=datasets,
        compare_dataset_ordering=True,
    )

    assert compared == ["first", "second"]
    assert report["compare_dataset_ordering"] is True
    assert report["ordering_gate_passed"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"test_size": 0.0}, "test_size"),
        ({"test_size": 1.0}, "test_size"),
        ({"inference_repeats": 0}, "inference_repeats"),
        ({"datasets": ()}, "dataset"),
        ({"dataset_aware_ordering": 1}, "dataset_aware_ordering"),
        ({"compare_dataset_ordering": 1}, "compare_dataset_ordering"),
    ],
)
def test_runner_rejects_invalid_settings(
    options: dict[str, Any],
    message: str,
    tmp_path: Path,
) -> None:
    arguments: dict[str, Any] = {
        "datasets": benchmark_runner.DATASETS[:1],
        "preset": "fast",
        "time_limit": None,
        "random_state": 42,
        "test_size": 0.2,
        "inference_repeats": 3,
        "include_autogluon": False,
    }
    arguments.update(options)

    with pytest.raises(ValueError, match=message):
        benchmark_runner.run_benchmarks(
            tmp_path / "results.json",
            **arguments,
        )
