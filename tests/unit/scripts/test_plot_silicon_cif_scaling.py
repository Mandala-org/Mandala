from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path("scripts/plot_silicon_cif_scaling.py").resolve()
    spec = importlib.util.spec_from_file_location("plot_silicon_cif_scaling", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_benchmark_file_retains_completed_sections_and_ignores_empty_one(
    tmp_path,
):
    benchmark_file = tmp_path / "benchmark.txt"
    benchmark_file.write_text(
        "\n".join(
            [
                "=== atoms=8 cif=/tmp/si8.cif ===",
                "data_preparation: repeats=10 min=0.1s max=0.3s avg=0.2s stddev=0.04s raw_seconds=[...]",
                "data_preparation_peak_gpu_memory=0.1 GiB",
                "model_evaluation: repeats=10 min=0.4s max=0.6s avg=0.5s stddev=0.05s raw_seconds=[...]",
                "model_evaluation_peak_gpu_memory=0.2 GiB",
                "matrix_construction: repeats=10 min=0.01s max=0.03s avg=0.02s stddev=0.005s raw_seconds=[...]",
                "matrix_construction_peak_gpu_memory=0.3 GiB",
                "=== atoms=32768 cif=/tmp/si32768.cif ===",
                "FAILED atoms=32768: RuntimeError: out of memory",
            ]
        ),
        encoding="utf-8",
    )

    results = _load_module().parse_benchmark_file(benchmark_file)

    assert len(results) == 1
    result = results[0]
    assert result.atom_count == 8
    assert result.timings["model_evaluation"].mean_seconds == 0.5
    assert result.peak_memory_gib["matrix_construction"] == 0.3


def test_parse_benchmark_file_keeps_partial_measurements(tmp_path):
    benchmark_file = tmp_path / "partial.txt"
    benchmark_file.write_text(
        "\n".join(
            [
                "=== atoms=64 cif=/tmp/si64.cif ===",
                "data_preparation: repeats=10 min=0.1s max=0.3s avg=0.2s stddev=0.04s raw_seconds=[...]",
            ]
        ),
        encoding="utf-8",
    )

    results = _load_module().parse_benchmark_file(benchmark_file)

    assert len(results) == 1
    assert results[0].atom_count == 64
    assert set(results[0].timings) == {"data_preparation"}


def test_parse_benchmark_file_accepts_chunked_inference_measurements(tmp_path):
    benchmark_file = tmp_path / "chunked.txt"
    benchmark_file.write_text(
        "\n".join(
            [
                "=== atoms=32768 cif=/tmp/si32768.cif ===",
                "chunked_inference_and_matrix_construction: repeats=10 min=1s max=3s avg=2s stddev=0.2s raw_seconds=[...]",
                "chunked_inference_and_matrix_construction_peak_gpu_memory=42.0 GiB",
            ]
        ),
        encoding="utf-8",
    )

    result = _load_module().parse_benchmark_file(benchmark_file)[0]

    assert (
        result.timings["chunked_inference_and_matrix_construction"].mean_seconds == 2.0
    )
    assert result.peak_memory_gib["chunked_inference_and_matrix_construction"] == 42.0
