"""
Analyze raw Hamiltonian data symmetry by shift pairs.

For each canonical shift s = (sx, sy, sz) in [-k, k]^3:
  - plot H(s)
  - plot transpose-partner from reverse shift: H(-s)^T
  - plot difference: H(s) - H(-s)^T

Shift pairs are not repeated:
  if (sx, sy, sz) and (-sx, -sy, -sz) are a pair, only one is plotted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.snapshot import Snapshot
from common import extract_partial_hamiltonian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze raw Hamiltonian data by shifts"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/small/H2O/original/H2O.matrix",
        help="Path to OpenMX matrix file",
    )
    parser.add_argument(
        "--info-path",
        type=str,
        default="data/small/H2O/original/H2O.info.out",
        help="Path to OpenMX info file",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        help="Basis convention for loading (default: e3nn)",
    )
    parser.add_argument(
        "--k-range",
        type=int,
        default=1,
        help="Shift range k; iterate over [-k, k]^3 (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: studies/minimal_overfit_study/analysis_data_k<k>)",
    )
    parser.add_argument(
        "--dynamic-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use percentile-based dynamic color ranges (default: True)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile for dynamic color range (default: 99.0)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Numerical floor for color-range stability (default: 1e-12)",
    )
    return parser.parse_args()


def is_canonical_shift(sx: int, sy: int, sz: int) -> bool:
    """Keep one representative from pair (s, -s)."""
    s = (sx, sy, sz)
    r = (-sx, -sy, -sz)
    return s <= r


def compute_symmetric_vmax(arr: np.ndarray, percentile: float, eps: float) -> float:
    abs_arr = np.abs(arr).reshape(-1)
    if abs_arr.size == 0:
        return 1.0
    vmax = float(np.percentile(abs_arr, percentile))
    if not np.isfinite(vmax) or vmax < eps:
        vmax = float(np.max(abs_arr))
    return max(vmax, eps)


def save_shift_plot(
    H_s: np.ndarray,
    H_rev_t: np.ndarray,
    diff: np.ndarray,
    shift: tuple[int, int, int],
    reverse_shift: tuple[int, int, int],
    output_path: Path,
    dynamic_range: bool,
    percentile: float,
    eps: float,
) -> dict:
    if dynamic_range:
        main_vmax = compute_symmetric_vmax(
            np.concatenate([H_s.reshape(-1), H_rev_t.reshape(-1)]), percentile, eps
        )
        diff_vmax = compute_symmetric_vmax(diff, percentile, eps)
    else:
        main_vmax = 1.0
        diff_vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(H_s, cmap="bwr", vmin=-main_vmax, vmax=main_vmax)
    axes[0].set_title(f"H[s], s={shift}")
    axes[0].set_xlabel("Orbital j")
    axes[0].set_ylabel("Orbital i")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(H_rev_t, cmap="bwr", vmin=-main_vmax, vmax=main_vmax)
    axes[1].set_title(f"H[-s]^T, -s={reverse_shift}")
    axes[1].set_xlabel("Orbital j")
    axes[1].set_ylabel("Orbital i")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap="bwr", vmin=-diff_vmax, vmax=diff_vmax)
    axes[2].set_title(f"Diff: H[s] - H[-s]^T\nMAE={np.mean(np.abs(diff)):.3e}")
    axes[2].set_xlabel("Orbital j")
    axes[2].set_ylabel("Orbital i")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "main_vmax": main_vmax,
        "diff_vmax": diff_vmax,
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("RAW DATA ANALYSIS")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Info path: {args.info_path}")
    print(f"Convention: {args.convention}")
    print(f"k-range: {args.k_range}")

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / f"analysis_data_k{args.k_range}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    snapshot = Snapshot.from_openmx(
        matrix_path=Path(args.data_path),
        info_path=Path(args.info_path),
        convention=args.convention,
        symmetrize_density=True,
        cutoff_radius=None,
        dtype=torch.float32,
    )

    H = snapshot.hamiltonian
    atoms = list(H.atoms)
    orbital_cfg = H.orbital_cfg

    summary_lines = []
    n_saved = 0
    n_skipped_empty = 0

    for sx in range(-args.k_range, args.k_range + 1):
        for sy in range(-args.k_range, args.k_range + 1):
            for sz in range(-args.k_range, args.k_range + 1):
                if not is_canonical_shift(sx, sy, sz):
                    continue

                rs = (-sx, -sy, -sz)
                H_s, _, _ = extract_partial_hamiltonian(
                    H, None, atoms, orbital_cfg, sx, sy, sz, partial_train=None
                )
                H_rs, _, _ = extract_partial_hamiltonian(
                    H, None, atoms, orbital_cfg, rs[0], rs[1], rs[2], partial_train=None
                )
                H_rs_t = H_rs.T
                diff = H_s - H_rs_t

                # Skip if both compared matrices are fully zero.
                if (not np.any(H_s)) and (not np.any(H_rs_t)):
                    n_skipped_empty += 1
                    continue

                file_name = f"H_vs_transpose_sx{sx:+d}_sy{sy:+d}_sz{sz:+d}.png"
                file_path = output_dir / file_name
                stats = save_shift_plot(
                    H_s=H_s,
                    H_rev_t=H_rs_t,
                    diff=diff,
                    shift=(sx, sy, sz),
                    reverse_shift=rs,
                    output_path=file_path,
                    dynamic_range=args.dynamic_range,
                    percentile=args.percentile,
                    eps=args.eps,
                )
                n_saved += 1
                print(
                    f"Saved {file_name} | MAE={stats['mae']:.3e}, max_abs={stats['max_abs']:.3e}"
                )
                summary_lines.append(
                    f"s=({sx:+d},{sy:+d},{sz:+d})  -s=({rs[0]:+d},{rs[1]:+d},{rs[2]:+d})  "
                    f"mae={stats['mae']:.6e}  max_abs={stats['max_abs']:.6e}  "
                    f"main_vmax={stats['main_vmax']:.6e}  diff_vmax={stats['diff_vmax']:.6e}"
                )

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Raw Hamiltonian transpose-pair analysis\n")
        f.write(f"k-range={args.k_range}\n")
        f.write(f"saved_plots={n_saved}\n")
        f.write(f"skipped_empty={n_skipped_empty}\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("-" * 80)
    print(f"Saved plots: {n_saved}")
    print(f"Skipped empty shift pairs: {n_skipped_empty}")
    print(f"Summary: {summary_path}")
    print("[OK] analyze_data finished.")


if __name__ == "__main__":
    main()
