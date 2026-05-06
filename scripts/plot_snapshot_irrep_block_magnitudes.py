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
import torch
from e3nn.o3 import Irrep

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402
from net.irrep_tools import build_irrep_block_matrix_cache  # noqa: E402


IRREP_ORDER = [
    "0e",
    "1o",
    "1e",
    "2o",
    "2e",
    "3o",
    "3e",
    "4o",
    "4e",
    "5o",
    "5e",
    "6e",
]


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot histograms of block magnitudes per irrep for each snapshot matrix."
        )
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("data/small/ZnCuSnSeS"),
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
        default=Path("eval_outputs/ZnCuSnSeS_irrep_block_histograms"),
        help="Directory for the histogram figures and summary JSON.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        help="Matrix convention used when loading the snapshot.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins per matrix.",
    )
    return parser.parse_args()


def _discover_snapshot_paths(
    snapshot_path: Path, matrix_path: Path | None, info_path: Path | None
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


def _load_snapshot(matrix_path: Path, info_path: Path, convention: str) -> Snapshot:
    return Snapshot.from_openmx(matrix_path, info_path, convention=convention)


def _collect_block_magnitudes(
    block_matrix_cache: dict[str, object],
) -> dict[str, torch.Tensor]:
    magnitudes: dict[str, torch.Tensor] = {}
    for irrep_key, irrep_matrix in block_matrix_cache.items():
        vals: list[torch.Tensor] = []
        for blocks in irrep_matrix.pair_blocks.values():
            if blocks.numel() == 0:
                continue
            vals.append(torch.linalg.norm(blocks, dim=(1, 2)).detach().cpu())
        if vals:
            magnitudes[irrep_key] = torch.cat(vals, dim=0)
        else:
            magnitudes[irrep_key] = torch.empty(0, dtype=torch.float32)
    return magnitudes


def _summary_stats(values: torch.Tensor) -> dict[str, float | int | None]:
    if values.numel() == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    vals = values.to(dtype=torch.float64)
    return {
        "count": int(vals.numel()),
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
        "mean": float(vals.mean().item()),
        "median": float(vals.median().item()),
        "p95": float(torch.quantile(vals, 0.95).item()),
    }


def _plot_matrix_histograms(
    matrix_name: str,
    magnitudes: dict[str, torch.Tensor],
    output_path: Path,
    bins: int,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), constrained_layout=True)
    axes_flat = list(axes.flat)

    all_vals = torch.cat(
        [
            vals.to(dtype=torch.float64)
            for vals in magnitudes.values()
            if vals.numel() > 0
        ],
        dim=0,
    )
    if all_vals.numel() > 0:
        hist_bins = np.histogram_bin_edges(all_vals.numpy(), bins=bins)
    else:
        hist_bins = bins

    for ax, irrep_key in zip(axes_flat, IRREP_ORDER, strict=False):
        vals = magnitudes.get(irrep_key, torch.empty(0))
        if vals.numel() == 0:
            ax.set_title(f"{irrep_key} (no data)")
            ax.axis("off")
            continue
        arr = vals.numpy()
        ax.hist(arr, bins=hist_bins, color="#2c7fb8", alpha=0.85)
        stats = _summary_stats(vals)
        ax.set_title(
            f"{irrep_key}  n={stats['count']}  mean={stats['mean']:.3g}  "
            f"p95={stats['p95']:.3g}"
        )
        ax.set_xlabel("Block Frobenius norm")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)

    fig.suptitle(f"{matrix_name} irrep block magnitudes", fontsize=18)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = setup_argparse()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path, args.matrix_path, args.info_path
    )
    print(f"Loading snapshot from matrix={matrix_path} info={info_path}", flush=True)
    snapshot = _load_snapshot(matrix_path, info_path, args.convention)
    mapper = BlockIrrepMapper(snapshot.hamiltonian.orbital_cfg)
    all_irreps = [Irrep(ir) for ir in IRREP_ORDER]

    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for matrix_name in ("hamiltonian", "overlap", "density"):
        matrix = getattr(snapshot, matrix_name)
        cache = build_irrep_block_matrix_cache(matrix, mapper, all_irreps)
        magnitudes = _collect_block_magnitudes(cache)
        summary[matrix_name] = {
            irrep_key: _summary_stats(vals) for irrep_key, vals in magnitudes.items()
        }
        output_path = output_dir / f"{matrix_name}_irrep_block_magnitudes.png"
        _plot_matrix_histograms(matrix_name, magnitudes, output_path, args.bins)
        print(f"Wrote {output_path}", flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
