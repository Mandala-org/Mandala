#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations_with_replacement
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.snapshot import Snapshot  # noqa: E402


DEFAULT_PROBLEM_EDGE = (1, 1, 0, 2, 17)
DEFAULT_PROBLEM_KEY = "Zn-Se"
DEFAULT_COMPETING_EDGE = (1, 0, 0, 2, 16)
DEFAULT_COMPETING_KEY = "Zn-Se"
DEFAULT_MATRIX_NAMES = ("hamiltonian", "overlap", "density")
MATRIX_COLORS = {
    "hamiltonian": "#1f77b4",
    "overlap": "#ff7f0e",
    "density": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot block magnitude versus edge distance for one or more matrices "
            "in the ZnCu2Sn_SeS_2_scale_1_010 snapshot."
        )
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("data/small/ZnCu2Sn_SeS_2_scale_1_010"),
        help="Snapshot directory containing HS.out and the OpenMX info file.",
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=None,
        help="Optional explicit matrix path. Overrides --snapshot-path discovery.",
    )
    parser.add_argument(
        "--info-path",
        type=Path,
        default=None,
        help="Optional explicit info path. Overrides --snapshot-path discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/tmp_scale_1_010_block_mag_vs_distance"),
        help="Directory for the output figure and summary JSON.",
    )
    parser.add_argument(
        "--matrix-names",
        type=str,
        default="hamiltonian,overlap,density",
        help=(
            "Comma-separated matrix names to plot. Use any subset of "
            "hamiltonian,overlap,density."
        ),
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        help="Snapshot convention used during loading.",
    )
    parser.add_argument(
        "--cutoff-radius",
        type=float,
        default=11.0,
        help="Target cutoff radius to mirror the benchmark preprocessing.",
    )
    parser.add_argument(
        "--apply-cutoff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the benchmark target cutoff before plotting.",
    )
    parser.add_argument(
        "--symmetrize-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symmetrize targets the same way dataset preprocessing does.",
    )
    parser.add_argument(
        "--problem-key",
        type=str,
        default=DEFAULT_PROBLEM_KEY,
        help="Directed pair key of the problematic target edge.",
    )
    parser.add_argument(
        "--problem-edge",
        type=int,
        nargs=5,
        metavar=("SX", "SY", "SZ", "SRC", "DST"),
        default=DEFAULT_PROBLEM_EDGE,
        help="Problematic 5D edge to highlight in red.",
    )
    parser.add_argument(
        "--competing-key",
        type=str,
        default=DEFAULT_COMPETING_KEY,
        help="Directed pair key of the competing graph edge.",
    )
    parser.add_argument(
        "--competing-edge",
        type=int,
        nargs=5,
        metavar=("SX", "SY", "SZ", "SRC", "DST"),
        default=DEFAULT_COMPETING_EDGE,
        help="Competing 5D edge to highlight in orange.",
    )
    return parser.parse_args()


def _parse_matrix_names(raw_value: str) -> list[str]:
    matrix_names = [name.strip() for name in raw_value.split(",") if name.strip()]
    valid_names = {"hamiltonian", "overlap", "density"}
    if not matrix_names:
        raise ValueError("At least one matrix name must be provided.")
    invalid_names = [name for name in matrix_names if name not in valid_names]
    if invalid_names:
        raise ValueError(
            "Unsupported matrix name(s): " + ", ".join(sorted(set(invalid_names)))
        )
    return matrix_names


def _discover_info_file(snapshot_dir: Path) -> Path | None:
    preferred = [
        "ZnCuSeS.out",
        "info.dat",
        "info.txt",
        "SiO2.out",
    ]
    for name in preferred:
        candidate = snapshot_dir / name
        if candidate.exists():
            return candidate
    for candidate in sorted(snapshot_dir.glob("*.out")):
        if candidate.name in {"HS.out", "log.out"}:
            continue
        return candidate
    return None


def _discover_snapshot_paths(
    snapshot_path: Path,
    matrix_path: Path | None,
    info_path: Path | None,
) -> tuple[Path, Path]:
    if matrix_path is not None and info_path is not None:
        return matrix_path, info_path
    if matrix_path is not None or info_path is not None:
        raise ValueError("Provide both --matrix-path and --info-path together.")
    if snapshot_path.is_file():
        matrix_candidate = snapshot_path
        info_candidate = _discover_info_file(snapshot_path.parent)
        if info_candidate is None:
            raise FileNotFoundError(
                f"Could not infer info file next to matrix file: {snapshot_path}"
            )
        return matrix_candidate, info_candidate
    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"Snapshot path does not exist: {snapshot_path}")
    matrix_candidate = snapshot_path / "HS.out"
    if not matrix_candidate.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_candidate}")
    info_candidate = _discover_info_file(snapshot_path)
    if info_candidate is None:
        raise FileNotFoundError(
            f"Could not infer info file under snapshot directory: {snapshot_path}"
        )
    return matrix_candidate, info_candidate


def _load_processed_snapshot(
    matrix_path: Path,
    info_path: Path,
    *,
    convention: str,
    apply_cutoff: bool,
    cutoff_radius: float,
    symmetrize_targets: bool,
) -> Snapshot:
    snap = Snapshot.from_openmx(
        matrix_path=matrix_path,
        info_path=info_path,
        convention=convention,
    )
    if apply_cutoff:
        snap = snap.filter_by_distance(cutoff_radius)
    if symmetrize_targets:
        snap = snap.symmetrize_matrices(
            hamiltonian=True,
            overlap=True,
            density=True,
        )
    return snap


def _unordered_pair_key(a: str, b: str, order_index: dict[str, int]) -> str:
    if order_index[a] <= order_index[b]:
        return f"{a}-{b}"
    return f"{b}-{a}"


def _block_sum_squares(blocks: torch.Tensor) -> torch.Tensor:
    return torch.sum(blocks * blocks, dim=(1, 2))


def _block_mean_squares(blocks: torch.Tensor) -> torch.Tensor:
    return torch.mean(blocks * blocks, dim=(1, 2))


def _edge_distances(
    edges: torch.Tensor,
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> torch.Tensor:
    src = edges[3].to(dtype=torch.long)
    dst = edges[4].to(dtype=torch.long)
    disp = positions[dst] - positions[src]
    if box is not None:
        shift = edges[:3].T.to(device=positions.device, dtype=positions.dtype)
        disp = disp + shift @ box
    return torch.linalg.norm(disp, dim=-1)


def _collect_subplot_data(
    snap: Snapshot,
    *,
    matrix_name: str,
    problem_key: str,
    problem_edge: tuple[int, int, int, int, int],
    competing_key: str,
    competing_edge: tuple[int, int, int, int, int],
    value_mode: str,
    highlight_edges: bool,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, object]]:
    matrix = getattr(snap, matrix_name)
    positions = snap.positions
    box = snap.box
    if positions is None:
        raise RuntimeError("Snapshot positions are missing.")

    element_order = list(dict.fromkeys(matrix.atoms))
    order_index = {el: idx for idx, el in enumerate(element_order)}
    subplot_order = [
        f"{a}-{b}" for a, b in combinations_with_replacement(element_order, 2)
    ]

    subplot_data: dict[str, dict[str, list[float]]] = {
        key: {
            "distances": [],
            "magnitudes": [],
            "is_problem": [],
            "is_competing": [],
            "directed_keys": [],
            "edges": [],
        }
        for key in subplot_order
    }
    problem_found = False
    competing_found = False
    problem_record: dict[str, object] | None = None
    competing_record: dict[str, object] | None = None

    for directed_key, blocks in matrix.pair_blocks.items():
        if directed_key not in matrix.pair_edges:
            raise RuntimeError(f"Missing pair_edges entry for key {directed_key}")
        edges = matrix.pair_edges[directed_key]
        if value_mode == "sum_squares":
            values = _block_sum_squares(blocks)
        elif value_mode == "mean_squares":
            values = _block_mean_squares(blocks)
        else:
            raise ValueError(f"Unsupported value_mode: {value_mode}")
        values = values.detach().cpu()
        dists = _edge_distances(edges, positions, box).detach().cpu()

        el_a, el_b = directed_key.split("-")
        subplot_key = _unordered_pair_key(el_a, el_b, order_index)
        payload = subplot_data[subplot_key]

        for idx, edge_row in enumerate(edges.t().tolist()):
            edge_5d = tuple(int(x) for x in edge_row)
            is_problem = (
                highlight_edges
                and directed_key == problem_key
                and edge_5d == problem_edge
            )
            is_competing = (
                highlight_edges
                and directed_key == competing_key
                and edge_5d == competing_edge
            )
            payload["distances"].append(float(dists[idx].item()))
            payload["magnitudes"].append(float(values[idx].item()))
            payload["is_problem"].append(bool(is_problem))
            payload["is_competing"].append(bool(is_competing))
            payload["directed_keys"].append(directed_key)
            payload["edges"].append(edge_5d)
            if is_problem:
                problem_found = True
                problem_record = {
                    "directed_key": directed_key,
                    "subplot_key": subplot_key,
                    "edge": edge_5d,
                    "distance": float(dists[idx].item()),
                    "magnitude": float(values[idx].item()),
                    "local_idx": idx,
                }
            if is_competing:
                competing_found = True
                competing_record = {
                    "directed_key": directed_key,
                    "subplot_key": subplot_key,
                    "edge": edge_5d,
                    "distance": float(dists[idx].item()),
                    "magnitude": float(values[idx].item()),
                    "local_idx": idx,
                }

    summary = {
        "matrix_name": matrix_name,
        "element_order": element_order,
        "subplot_order": subplot_order,
        "problem_key": problem_key,
        "problem_edge": list(problem_edge),
        "competing_key": competing_key,
        "competing_edge": list(competing_edge),
        "problem_found": problem_found,
        "problem_record": problem_record,
        "competing_found": competing_found,
        "competing_record": competing_record,
    }
    return subplot_data, summary


def _plot_multi_matrix(
    subplot_data_by_matrix: dict[str, dict[str, dict[str, list[float]]]],
    summary_by_matrix: dict[str, dict[str, object]],
    output_path: Path,
    *,
    y_label: str,
    title: str,
) -> None:
    first_summary = next(iter(summary_by_matrix.values()))
    subplot_order = first_summary["subplot_order"]
    fig, axes = plt.subplots(3, 5, figsize=(19.2, 9.6), constrained_layout=True)
    axes_flat = list(axes.flat)

    legend_handles = []
    legend_labels = []
    for matrix_name, color in MATRIX_COLORS.items():
        if matrix_name not in subplot_data_by_matrix:
            continue
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="none", color=color, markersize=8)
        )
        legend_labels.append(matrix_name)

    for ax, subplot_key in zip(axes_flat, subplot_order, strict=True):
        any_data = False
        for matrix_name, payload_by_subplot in subplot_data_by_matrix.items():
            payload = payload_by_subplot[subplot_key]
            distances = payload["distances"]
            magnitudes = payload["magnitudes"]
            is_problem = payload["is_problem"]
            is_competing = payload["is_competing"]
            if not distances:
                continue
            any_data = True
            color = MATRIX_COLORS.get(matrix_name, "#666666")

            blue_x = [
                d
                for d, bad, competing in zip(
                    distances, is_problem, is_competing, strict=True
                )
                if not bad and not competing
            ]
            blue_y = [
                m
                for m, bad, competing in zip(
                    magnitudes, is_problem, is_competing, strict=True
                )
                if not bad and not competing
            ]
            ax.scatter(blue_x, blue_y, s=18, c=color, alpha=0.7)

        if not any_data:
            ax.set_title(f"{subplot_key} (no data)")
            ax.set_xlabel("Edge distance")
            ax.set_ylabel(y_label)
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            continue

        ax.set_title(f"{subplot_key}")
        ax.set_xlabel("Edge distance")
        ax.set_ylabel(y_label)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        if subplot_key == subplot_order[0] and legend_handles:
            ax.legend(
                legend_handles,
                legend_labels,
                loc="lower left",
                frameon=True,
                framealpha=0.9,
                fontsize=9,
            )

    fig.suptitle(title, fontsize=18)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot(
    subplot_data: dict[str, dict[str, list[float]]],
    summary: dict[str, object],
    output_path: Path,
    *,
    y_label: str,
    title: str,
) -> None:
    subplot_order = summary["subplot_order"]
    fig, axes = plt.subplots(3, 5, figsize=(30, 15), constrained_layout=True)
    axes_flat = list(axes.flat)

    for ax, subplot_key in zip(axes_flat, subplot_order, strict=True):
        payload = subplot_data[subplot_key]
        distances = payload["distances"]
        magnitudes = payload["magnitudes"]
        is_problem = payload["is_problem"]
        is_competing = payload["is_competing"]

        if not distances:
            ax.set_title(f"{subplot_key} (no data)")
            ax.set_xlabel("Edge distance")
            ax.set_ylabel(y_label)
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            continue

        blue_x = [
            d
            for d, bad, competing in zip(
                distances, is_problem, is_competing, strict=True
            )
            if not bad and not competing
        ]
        blue_y = [
            m
            for m, bad, competing in zip(
                magnitudes, is_problem, is_competing, strict=True
            )
            if not bad and not competing
        ]
        red_x = [d for d, bad in zip(distances, is_problem, strict=True) if bad]
        red_y = [m for m, bad in zip(magnitudes, is_problem, strict=True) if bad]
        orange_x = [
            d for d, competing in zip(distances, is_competing, strict=True) if competing
        ]
        orange_y = [
            m
            for m, competing in zip(magnitudes, is_competing, strict=True)
            if competing
        ]

        ax.scatter(blue_x, blue_y, s=20, c="#1f77b4", alpha=0.85)
        if red_x:
            ax.scatter(red_x, red_y, s=55, c="#d62728", alpha=0.95, zorder=5)
            ax.annotate(
                "problem edge",
                (red_x[0], red_y[0]),
                xytext=(8, 8),
                textcoords="offset points",
                color="#d62728",
                fontsize=9,
            )
        if orange_x:
            ax.scatter(orange_x, orange_y, s=55, c="#ff7f0e", alpha=0.95, zorder=5)
            ax.annotate(
                "competing edge",
                (orange_x[0], orange_y[0]),
                xytext=(8, -14),
                textcoords="offset points",
                color="#ff7f0e",
                fontsize=9,
            )

        ax.set_title(f"{subplot_key}  n={len(distances)}")
        ax.set_xlabel("Edge distance")
        ax.set_ylabel(y_label)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=18)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_combined(
    subplot_data: dict[str, dict[str, list[float]]],
    summary: dict[str, object],
    output_path: Path,
    *,
    y_label: str,
    title: str,
) -> None:
    subplot_order = summary["subplot_order"]
    fig, ax = plt.subplots(1, 1, figsize=(13, 9), constrained_layout=True)
    cmap = plt.get_cmap("tab20")

    for idx, subplot_key in enumerate(subplot_order):
        payload = subplot_data[subplot_key]
        distances = payload["distances"]
        magnitudes = payload["magnitudes"]
        if not distances:
            continue
        color = cmap(idx % cmap.N)
        ax.scatter(
            distances,
            magnitudes,
            s=18,
            alpha=0.8,
            color=color,
            label=subplot_key,
        )

    ax.set_title(title)
    ax.set_xlabel("Edge distance")
    ax.set_ylabel(y_label)
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_names = _parse_matrix_names(args.matrix_names)

    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path,
        args.matrix_path,
        args.info_path,
    )
    print(f"Loading snapshot from matrix={matrix_path} info={info_path}", flush=True)
    snap = _load_processed_snapshot(
        matrix_path,
        info_path,
        convention=args.convention,
        apply_cutoff=bool(args.apply_cutoff),
        cutoff_radius=float(args.cutoff_radius),
        symmetrize_targets=bool(args.symmetrize_targets),
    )

    subplot_data_by_matrix: dict[str, dict[str, dict[str, list[float]]]] = {}
    summary_by_matrix: dict[str, dict[str, object]] = {}
    output_labels: list[str] = []
    for matrix_name in matrix_names:
        subplot_data, summary = _collect_subplot_data(
            snap,
            matrix_name=matrix_name,
            problem_key=args.problem_key,
            problem_edge=tuple(int(x) for x in args.problem_edge),
            competing_key=args.competing_key,
            competing_edge=tuple(int(x) for x in args.competing_edge),
            value_mode="sum_squares",
            highlight_edges=False,
        )
        subplot_data_by_matrix[matrix_name] = subplot_data
        summary_by_matrix[matrix_name] = summary
        output_labels.append(matrix_name)

    figure_basename = "_".join(output_labels) + "_block_magnitude_vs_distance_3x5.png"
    figure_path = output_dir / figure_basename
    _plot_multi_matrix(
        subplot_data_by_matrix,
        summary_by_matrix,
        figure_path,
        y_label="Block sum of squares",
        title=f"scale_1_010 block sum of squares vs edge distance ({', '.join(output_labels)})",
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_by_matrix, indent=2, sort_keys=True))

    print(f"Wrote {figure_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    for matrix_name in matrix_names:
        summary = summary_by_matrix[matrix_name]
        if summary["problem_found"]:
            problem_record = summary["problem_record"]
            print(
                f"[{matrix_name}] Problem edge found: "
                f"{problem_record['directed_key']} "
                f"edge={tuple(problem_record['edge'])} "
                f"distance={problem_record['distance']:.6f} "
                f"magnitude={problem_record['magnitude']:.6f}",
                flush=True,
            )
        elif matrix_name == "hamiltonian":
            print(
                "Problem edge was not found in the plotted Hamiltonian target state.",
                flush=True,
            )
        if summary["competing_found"]:
            competing_record = summary["competing_record"]
            print(
                f"[{matrix_name}] Competing edge found: "
                f"{competing_record['directed_key']} "
                f"edge={tuple(competing_record['edge'])} "
                f"distance={competing_record['distance']:.6f} "
                f"magnitude={competing_record['magnitude']:.6f}",
                flush=True,
            )
        elif matrix_name == "hamiltonian":
            print(
                "Competing edge was not found in the plotted Hamiltonian target state.",
                flush=True,
            )


if __name__ == "__main__":
    main()
