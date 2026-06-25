#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.snapshot import Snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pair-resolved Hamiltonian edge distance distributions for a "
            "single snapshot."
        )
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        required=True,
        help="Snapshot directory containing HS.out and the OpenMX info file.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
    )
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Optional explicit x-axis max in Angstrom.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/zncusnses_pair_edge_rdfs_single_snapshot"),
    )
    return parser.parse_args()


def _discover_snapshot_paths(snapshot_dir: Path) -> tuple[Path, Path]:
    matrix_path = snapshot_dir / "HS.out"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    for name in ("ZnCuSeS.out", "Si.out", "SiO2.out", "info.dat", "info.txt"):
        candidate = snapshot_dir / name
        if candidate.exists():
            return matrix_path, candidate

    for candidate in sorted(snapshot_dir.glob("*.out")):
        if candidate.name != "HS.out":
            return matrix_path, candidate

    raise FileNotFoundError(
        f"Could not infer info file under snapshot path: {snapshot_dir}"
    )


def _load_snapshot_distances(
    snapshot_path: Path,
    *,
    convention: str,
) -> dict[str, np.ndarray]:
    matrix_path, info_path = _discover_snapshot_paths(snapshot_path)
    snap = Snapshot.from_openmx(
        matrix_path=matrix_path,
        info_path=info_path,
        convention=convention,
        symmetrize_density=True,
        cutoff_radius=None,
        cfg=None,
    ).symmetrize_matrices(
        hamiltonian=True,
        overlap=True,
        density=True,
    )
    return {
        key: distances.detach().cpu().numpy()
        for key, distances in snap._edge_distances(snap.hamiltonian).items()
    }


def _plot_panel(
    distances_by_key: dict[str, np.ndarray],
    *,
    output_path: Path,
    bins: int,
    max_distance: float | None,
    title: str,
) -> dict[str, dict[str, float]]:
    keys = sorted(distances_by_key)
    if not keys:
        raise ValueError("No pair keys available to plot.")

    if max_distance is None:
        max_distance = max(
            float(values.max())
            for values in distances_by_key.values()
            if values.size > 0
        )
    if max_distance <= 0:
        raise ValueError("max_distance must be positive after inference.")

    n = len(keys)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 2.8 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    edges = np.linspace(0.0, max_distance, bins + 1)
    bin_width = edges[1] - edges[0]
    stats: dict[str, dict[str, float]] = {}

    for ax, key in zip(axes.flat, keys):
        values = distances_by_key[key]
        counts, _ = np.histogram(values, bins=edges)
        density = counts / max(float(values.size), 1.0) / bin_width
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, density, color="#5fa8ff", lw=1.5)
        ax.fill_between(centers, 0.0, density, color="#5fa8ff", alpha=0.18)
        ax.set_title(key)
        ax.set_xlim(0.0, max_distance)
        ax.grid(alpha=0.25)
        stats[key] = {
            "count": float(values.size),
            "min": float(values.min()),
            "p10": float(np.percentile(values, 10.0)),
            "p50": float(np.percentile(values, 50.0)),
            "p90": float(np.percentile(values, 90.0)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
        ax.text(
            0.98,
            0.97,
            f"n={values.size}\nmean={values.mean():.2f} Å\np90={np.percentile(values, 90.0):.2f} Å",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(title)
    for row_axes in axes:
        row_axes[0].set_ylabel("Probability density")
    for ax in axes[-1]:
        if ax.has_data():
            ax.set_xlabel("Edge length (Angstrom)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return stats


def main() -> None:
    args = parse_args()
    distances = _load_snapshot_distances(
        args.snapshot_path,
        convention=args.convention,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "pair_edge_rdf_single_snapshot.png"
    summary_path = args.output_dir / "pair_edge_rdf_single_snapshot_summary.json"
    stats = _plot_panel(
        distances,
        output_path=output_path,
        bins=args.bins,
        max_distance=args.max_distance,
        title=f"Hamiltonian edge distance distributions: {args.snapshot_path.name}",
    )
    summary_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Saved plot: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
