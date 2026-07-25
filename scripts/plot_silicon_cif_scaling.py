#!/usr/bin/env python
"""Plot stacked timing and GPU-memory breakdowns from CIF scaling benchmarks."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "eval_outputs/silicon_cif_scaling_b200.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_outputs/silicon_cif_scaling_plots"

PHASES = (
    "data_preparation",
    "model_evaluation",
    "matrix_construction",
    "chunked_inference_and_matrix_construction",
)
PHASE_LABELS = {
    "data_preparation": "Data preparation",
    "model_evaluation": "Model evaluation",
    "matrix_construction": "Matrix construction",
    "chunked_inference_and_matrix_construction": "Chunked inference + matrices",
}
PHASE_COLORS = {
    "data_preparation": "#4c78a8",
    "model_evaluation": "#f58518",
    "matrix_construction": "#54a24b",
    "chunked_inference_and_matrix_construction": "#e45756",
}
SECTION_RE = re.compile(r"^=== atoms=(?P<atoms>\d+)\s+cif=.* ===$")
TIME_RE = re.compile(
    r"^(?P<phase>data_preparation|model_evaluation|matrix_construction|"
    r"chunked_inference_and_matrix_construction): "
    r"repeats=(?P<repeats>\d+) "
    r"min=(?P<minimum>[0-9.eE+-]+)s "
    r"max=(?P<maximum>[0-9.eE+-]+)s "
    r"avg=(?P<mean>[0-9.eE+-]+)s "
    r"stddev=(?P<stddev>[0-9.eE+-]+)s"
)
MEMORY_RE = re.compile(
    r"^(?P<phase>data_preparation|model_evaluation|matrix_construction|"
    r"chunked_inference_and_matrix_construction)"
    r"_peak_gpu_memory=(?P<gib>[0-9.eE+-]+) GiB$"
)


@dataclass(frozen=True)
class TimingMeasurement:
    repeats: int
    minimum_seconds: float
    maximum_seconds: float
    mean_seconds: float
    stddev_seconds: float


@dataclass
class BenchmarkResult:
    atom_count: int
    timings: dict[str, TimingMeasurement] = field(default_factory=dict)
    peak_memory_gib: dict[str, float] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create stacked timing and memory scaling plots from "
            "benchmark_silicon_cif_scaling.py output."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--show-uncertainty",
        action="store_true",
        help="Show combined one-sigma timing uncertainty bars.",
    )
    return parser.parse_args()


def parse_benchmark_file(path: Path) -> list[BenchmarkResult]:
    """Parse completed or partially completed benchmark sections.

    Sections with no timing or memory measurement are ignored. This naturally
    excludes an interrupted size while retaining every completed measurement.
    """
    results: dict[int, BenchmarkResult] = {}
    current: BenchmarkResult | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        section_match = SECTION_RE.match(line)
        if section_match is not None:
            atom_count = int(section_match.group("atoms"))
            current = BenchmarkResult(atom_count=atom_count)
            results[atom_count] = current
            continue
        if current is None:
            continue

        time_match = TIME_RE.match(line)
        if time_match is not None:
            phase = time_match.group("phase")
            current.timings[phase] = TimingMeasurement(
                repeats=int(time_match.group("repeats")),
                minimum_seconds=float(time_match.group("minimum")),
                maximum_seconds=float(time_match.group("maximum")),
                mean_seconds=float(time_match.group("mean")),
                stddev_seconds=float(time_match.group("stddev")),
            )
            continue

        memory_match = MEMORY_RE.match(line)
        if memory_match is not None:
            current.peak_memory_gib[memory_match.group("phase")] = float(
                memory_match.group("gib")
            )

    return [
        result
        for _atom_count, result in sorted(results.items())
        if result.timings or result.peak_memory_gib
    ]


def _available_phases(
    results: list[BenchmarkResult],
    attribute: str,
) -> list[str]:
    return [
        phase
        for phase in PHASES
        if any(getattr(result, attribute).get(phase) is not None for result in results)
    ]


def _stacked_bars(
    ax: plt.Axes,
    *,
    positions: np.ndarray,
    results: list[BenchmarkResult],
    phases: list[str],
    values_by_phase: dict[str, np.ndarray],
) -> np.ndarray:
    totals = np.zeros(len(results), dtype=float)
    for phase in phases:
        values = values_by_phase[phase]
        ax.bar(
            positions,
            values,
            bottom=totals,
            width=0.68,
            color=PHASE_COLORS[phase],
            edgecolor="white",
            linewidth=0.75,
            label=PHASE_LABELS[phase],
        )
        totals += values
    return totals


def _annotate_totals(
    ax: plt.Axes,
    positions: np.ndarray,
    totals: np.ndarray,
    *,
    logarithmic: bool,
) -> None:
    if len(totals) == 0:
        return
    for position, total in zip(positions, totals, strict=True):
        label_y = (
            total * 1.08
            if logarithmic
            else total + max(float(np.max(totals)) * 0.012, 0.01)
        )
        ax.text(
            position,
            label_y,
            f"{total:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _configure_log_axis(
    ax: plt.Axes,
    values_by_phase: dict[str, np.ndarray],
    totals: np.ndarray,
) -> None:
    positive_values = [
        float(value)
        for phase_values in values_by_phase.values()
        for value in phase_values
        if value > 0.0
    ]
    if not positive_values:
        raise ValueError("Cannot use a logarithmic scale without positive values")
    ax.set_yscale("log")
    ax.set_ylim(min(positive_values) * 0.5, float(np.max(totals)) * 1.35)


def plot_times(
    results: list[BenchmarkResult],
    output_path: Path,
    dpi: int,
    *,
    show_uncertainty: bool,
) -> None:
    phases = _available_phases(results, "timings")
    if not phases:
        return
    positions = np.arange(len(results), dtype=float)
    values_by_phase = {
        phase: np.array(
            [
                result.timings[phase].mean_seconds if phase in result.timings else 0.0
                for result in results
            ],
            dtype=float,
        )
        for phase in phases
    }
    error_by_phase = {
        phase: np.array(
            [
                result.timings[phase].stddev_seconds if phase in result.timings else 0.0
                for result in results
            ],
            dtype=float,
        )
        for phase in phases
    }
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.88)
    totals = _stacked_bars(
        axis,
        positions=positions,
        results=results,
        phases=phases,
        values_by_phase=values_by_phase,
    )
    if show_uncertainty:
        total_stddev = np.sqrt(
            sum(np.square(error_by_phase[phase]) for phase in phases)
        )
        axis.errorbar(
            positions,
            totals,
            yerr=total_stddev,
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.1,
            label="Combined 1 sigma timing uncertainty",
        )
    _configure_log_axis(axis, values_by_phase, totals)
    _annotate_totals(axis, positions, totals, logarithmic=True)
    axis.set_xticks(positions, [str(result.atom_count) for result in results])
    axis.set_xlabel("Atoms in periodic Silicon supercell")
    axis.set_ylabel("Total time per evaluation (s)")
    axis.set_title("Mandala Silicon inference scaling: time breakdown", pad=14)
    axis.grid(axis="y", alpha=0.28)
    axis.legend(loc="upper left")
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_memory(results: list[BenchmarkResult], output_path: Path, dpi: int) -> None:
    phases = _available_phases(results, "peak_memory_gib")
    if not phases:
        return
    positions = np.arange(len(results), dtype=float)
    values_by_phase = {
        phase: np.array(
            [result.peak_memory_gib.get(phase, 0.0) for result in results],
            dtype=float,
        )
        for phase in phases
    }

    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.20, top=0.88)
    totals = _stacked_bars(
        axis,
        positions=positions,
        results=results,
        phases=phases,
        values_by_phase=values_by_phase,
    )
    _configure_log_axis(axis, values_by_phase, totals)
    _annotate_totals(axis, positions, totals, logarithmic=True)
    axis.set_xticks(positions, [str(result.atom_count) for result in results])
    axis.set_xlabel("Atoms in periodic Silicon supercell")
    axis.set_ylabel("Sum of logged per-phase peak allocations (GiB)")
    axis.set_title("Mandala Silicon inference scaling: GPU-memory breakdown", pad=14)
    axis.grid(axis="y", alpha=0.28)
    axis.legend(loc="upper left")
    figure.text(
        0.5,
        0.055,
        "Phase peaks are recorded separately; stacked totals compare phase costs and are not simultaneous live memory.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {input_path}")

    results = parse_benchmark_file(input_path)
    if not results:
        raise ValueError(f"No completed benchmark measurements found in {input_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    time_path = output_dir / "silicon_cif_scaling_time.png"
    memory_path = output_dir / "silicon_cif_scaling_memory.png"
    plot_times(
        results,
        time_path,
        args.dpi,
        show_uncertainty=args.show_uncertainty,
    )
    plot_memory(results, memory_path, args.dpi)

    available_counts = ", ".join(str(result.atom_count) for result in results)
    print(f"parsed_atom_counts=[{available_counts}]", flush=True)
    print(f"wrote={time_path}", flush=True)
    print(f"wrote={memory_path}", flush=True)


if __name__ == "__main__":
    main()
