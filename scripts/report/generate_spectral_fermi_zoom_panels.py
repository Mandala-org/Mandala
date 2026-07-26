#!/usr/bin/env python
"""Generate review-only band/DOS plots with a Fermi-level zoom panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "eval_outputs/paper_spectral_fermi_zoom_panels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", required=True)
    parser.add_argument("--zoom-min", type=float, default=-2.0)
    parser.add_argument("--zoom-max", type=float, default=2.0)
    return parser.parse_args()


def _load_band(directory: Path, kind: str) -> dict:
    candidates = sorted(directory.glob(f"band_structure_{kind}*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No band_structure_{kind}*.pt in {directory}")
    return torch.load(candidates[0], map_location="cpu", weights_only=False)


def _plot_bands(ax, gt: dict, pred: dict, *, alpha: float = 1.0) -> None:
    x = gt["linear_k"].detach().cpu().numpy()
    gt_e = (gt["eigenvalues"] - gt["fermi_level"]).detach().cpu().numpy()
    pred_e = (pred["eigenvalues"] - gt["fermi_level"]).detach().cpu().numpy()
    ax.plot(x, gt_e, color="black", linewidth=0.65, alpha=alpha)
    ax.plot(x, pred_e, color="#d62728", linewidth=0.65, linestyle="--", alpha=alpha)
    for tick in gt["tick_positions"].detach().cpu().numpy():
        ax.axvline(tick, color="0.82", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xticks(gt["tick_positions"].detach().cpu().numpy())
    ax.set_xticklabels(
        [
            r"$\Gamma$" if str(label).upper() in {"G", "GAMMA"} else label
            for label in gt["tick_labels"]
        ]
    )


def main() -> None:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = _load_band(evaluation_dir, "gt")
    pred = _load_band(evaluation_dir, "pred")
    dos = torch.load(
        evaluation_dir / "dos_comparison.pt", map_location="cpu", weights_only=False
    )

    figure, (band_ax, dos_ax) = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [2.25, 1.0]}
    )
    _plot_bands(band_ax, gt, pred)
    band_ax.set_ylim(-15.0, 25.0)
    band_ax.set_ylabel(r"$E-E_F$ (eV)")
    band_ax.set_title("Band structure")

    inset = inset_axes(
        band_ax, width="48%", height="44%", loc="upper right", borderpad=1.2
    )
    _plot_bands(inset, gt, pred)
    inset.set_ylim(args.zoom_min, args.zoom_max)
    inset.set_xticklabels([])
    inset.set_ylabel(r"$E-E_F$ (eV)", fontsize=8)
    inset.tick_params(labelsize=8)
    mark_inset(band_ax, inset, loc1=2, loc2=4, fc="none", ec="0.35", linewidth=0.8)

    grid_true = np.asarray(dos["grid_true"], dtype=float) - float(dos["fermi_true_ev"])
    grid_pred = np.asarray(dos["grid_pred"], dtype=float) - float(dos["fermi_true_ev"])
    dos_ax.plot(
        dos["dos_true"], grid_true, color="black", linewidth=1.1, label="Reference"
    )
    dos_ax.plot(
        dos["dos_pred"],
        grid_pred,
        color="#d62728",
        linestyle="--",
        linewidth=1.1,
        label="Prediction",
    )
    dos_ax.axhline(0.0, color="black", linewidth=0.8)
    dos_ax.set_ylim(-15.0, 25.0)
    dos_ax.set_xlabel("DOS (states/eV)")
    dos_ax.set_yticklabels([])
    dos_ax.set_title("Density of states")
    dos_ax.legend(frameon=False, loc="upper right")

    figure.suptitle(args.title)
    figure.tight_layout()
    output_path = output_dir / f"{evaluation_dir.name}_fermi_zoom.png"
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
