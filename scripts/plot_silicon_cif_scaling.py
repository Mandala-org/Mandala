#!/usr/bin/env python
"""Plot total scaling and phase contributions from CIF scaling benchmarks."""

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
FIGURE_SIZE = (6.2, 4.2)

PHASES = ("data_preparation", "model_evaluation", "matrix_construction")
PHASE_LABELS = {
    "data_preparation": "Data preparation",
    "model_evaluation": "Model evaluation",
    "matrix_construction": "Matrix construction",
}
PHASE_COLORS = {
    "data_preparation": "#4c78a8",
    "model_evaluation": "#f58518",
    "matrix_construction": "#54a24b",
}
SECTION_RE = re.compile(r"^=== atoms=(?P<atoms>\d+)\s+cif=.* ===$")
TIME_RE = re.compile(
    r"^(?P<phase>data_preparation|model_evaluation|matrix_construction): "
    r"repeats=(?P<repeats>\d+) "
    r"min=(?P<minimum>[0-9.eE+-]+)s "
    r"max=(?P<maximum>[0-9.eE+-]+)s "
    r"avg=(?P<mean>[0-9.eE+-]+)s "
    r"stddev=(?P<stddev>[0-9.eE+-]+)s"
)
MEMORY_RE = re.compile(
    r"^(?P<phase>data_preparation|model_evaluation|matrix_construction)"
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


def _annotate_values(
    ax: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    *,
    logarithmic: bool,
    suffix: str = "",
) -> None:
    if len(values) == 0:
        return
    for position, value in zip(positions, values, strict=True):
        label_y = (
            value * 1.08
            if logarithmic
            else value + max(float(np.max(values)) * 0.012, 0.01)
        )
        ax.text(
            position,
            label_y,
            f"{value:.3g}{suffix}",
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


def _phase_values(
    results: list[BenchmarkResult],
    phases: list[str],
    attribute: str,
) -> dict[str, np.ndarray]:
    if attribute == "timings":
        return {
            phase: np.array(
                [
                    (
                        result.timings[phase].mean_seconds
                        if phase in result.timings
                        else 0.0
                    )
                    for result in results
                ],
                dtype=float,
            )
            for phase in phases
        }
    if attribute == "peak_memory_gib":
        return {
            phase: np.array(
                [result.peak_memory_gib.get(phase, 0.0) for result in results],
                dtype=float,
            )
            for phase in phases
        }
    raise ValueError(f"Unsupported benchmark attribute: {attribute}")


def _configure_common_x_axis(
    axis: plt.Axes,
    positions: np.ndarray,
    results: list[BenchmarkResult],
) -> None:
    axis.set_xticks(positions, [str(result.atom_count) for result in results])
    axis.set_xlabel("Atoms in periodic Silicon supercell")
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)


def plot_time_total(
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
    values_by_phase = _phase_values(results, phases, "timings")
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
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.86)
    totals = sum(values_by_phase.values(), np.zeros(len(results), dtype=float))
    axis.bar(
        positions,
        totals,
        width=0.68,
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.75,
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
        )
    _configure_log_axis(axis, {"total": totals}, totals)
    _annotate_values(axis, positions, totals, logarithmic=True)
    _configure_common_x_axis(axis, positions, results)
    axis.set_ylabel("Total time per evaluation (s)")
    axis.set_title("Mandala Silicon inference scaling: total time", pad=14)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_time_contributions(
    results: list[BenchmarkResult],
    output_path: Path,
    dpi: int,
) -> None:
    phases = _available_phases(results, "timings")
    if not phases:
        return
    positions = np.arange(len(results), dtype=float)
    values_by_phase = _phase_values(results, phases, "timings")
    totals = sum(values_by_phase.values(), np.zeros(len(results), dtype=float))
    percentages = {
        phase: np.divide(
            values,
            totals,
            out=np.zeros_like(values),
            where=totals > 0.0,
        )
        * 100.0
        for phase, values in values_by_phase.items()
    }

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.86)
    _stacked_bars(
        axis,
        positions=positions,
        results=results,
        phases=phases,
        values_by_phase=percentages,
    )
    _configure_common_x_axis(axis, positions, results)
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Contribution to total time (%)")
    axis.set_title("Mandala Silicon inference scaling: time contributions", pad=14)
    axis.legend(loc="upper left")
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_memory_max(
    results: list[BenchmarkResult],
    output_path: Path,
    dpi: int,
) -> None:
    phases = _available_phases(results, "peak_memory_gib")
    if not phases:
        return
    positions = np.arange(len(results), dtype=float)
    values_by_phase = _phase_values(results, phases, "peak_memory_gib")
    maxima = np.maximum.reduce(list(values_by_phase.values()))

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.86)
    axis.bar(
        positions,
        maxima,
        width=0.68,
        color="#f58518",
        edgecolor="white",
        linewidth=0.75,
    )
    _configure_log_axis(axis, {"maximum": maxima}, maxima)
    _annotate_values(axis, positions, maxima, logarithmic=True)
    _configure_common_x_axis(axis, positions, results)
    axis.set_ylabel("Maximum phase peak allocation (GiB)")
    axis.set_title("Mandala Silicon inference scaling: peak GPU memory", pad=14)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_memory_contributions(
    results: list[BenchmarkResult],
    output_path: Path,
    dpi: int,
) -> None:
    phases = _available_phases(results, "peak_memory_gib")
    if not phases:
        return
    positions = np.arange(len(results), dtype=float)
    values_by_phase = _phase_values(results, phases, "peak_memory_gib")
    maxima = np.maximum.reduce(list(values_by_phase.values()))
    percentages = {
        phase: np.divide(
            values,
            maxima,
            out=np.zeros_like(values),
            where=maxima > 0.0,
        )
        * 100.0
        for phase, values in values_by_phase.items()
    }

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.86)
    width = 0.22
    offsets = (np.arange(len(phases), dtype=float) - (len(phases) - 1) / 2) * width
    for offset, phase in zip(offsets, phases, strict=True):
        axis.bar(
            positions + offset,
            percentages[phase],
            width=width,
            color=PHASE_COLORS[phase],
            edgecolor="white",
            linewidth=0.75,
            label=PHASE_LABELS[phase],
        )
    _configure_common_x_axis(axis, positions, results)
    axis.set_ylim(0.0, 110.0)
    axis.set_ylabel("Phase peak relative to maximum (%)")
    axis.set_title("Mandala Silicon inference scaling: memory contributions", pad=14)
    axis.legend(loc="upper left")
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
    time_total_path = output_dir / "silicon_cif_scaling_time_total.png"
    time_contributions_path = output_dir / "silicon_cif_scaling_time_contributions.png"
    memory_max_path = output_dir / "silicon_cif_scaling_memory_max.png"
    memory_contributions_path = (
        output_dir / "silicon_cif_scaling_memory_contributions.png"
    )
    plot_time_total(
        results,
        time_total_path,
        args.dpi,
        show_uncertainty=args.show_uncertainty,
    )
    plot_time_contributions(results, time_contributions_path, args.dpi)
    plot_memory_max(results, memory_max_path, args.dpi)
    plot_memory_contributions(results, memory_contributions_path, args.dpi)

    available_counts = ", ".join(str(result.atom_count) for result in results)
    print(f"parsed_atom_counts=[{available_counts}]", flush=True)
    print(f"wrote={time_total_path}", flush=True)
    print(f"wrote={time_contributions_path}", flush=True)
    print(f"wrote={memory_max_path}", flush=True)
    print(f"wrote={memory_contributions_path}", flush=True)


if __name__ == "__main__":
    main()
