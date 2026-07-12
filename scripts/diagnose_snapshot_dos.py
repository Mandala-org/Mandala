"""Create plain Matplotlib diagnostics from an evaluation bundle's saved spectra."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HARTREE_TO_EV = 27.211386245988


def gaussian_dos(eigenvalues: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    result = np.zeros_like(grid)
    for chunk in np.array_split(np.asarray(eigenvalues).reshape(-1), 32):
        result += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / sigma) ** 2).sum(
            axis=1
        )
    return result / (np.sqrt(2.0 * np.pi) * sigma)


def openmx_scf_eigenvalues(path: Path) -> np.ndarray:
    text = path.read_text(errors="ignore")
    match = re.search(
        r"Eigenvalues\s+Up-spin\s+Down-spin\s*(.*?)(?:\n\s*\n|$)", text, re.S
    )
    if match is None:
        raise ValueError(f"Could not find the OpenMX SCF eigenvalue table in {path}")
    values = []
    for line in match.group(1).splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].isdigit():
            values.append(float(fields[1]) * HARTREE_TO_EV)
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--openmx-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt = torch.load(
        args.evaluation_dir / "band_structure_gt_gt_overlap.pt",
        map_location="cpu",
        weights_only=False,
    )
    pred = torch.load(
        args.evaluation_dir / "band_structure_pred_gt_overlap.pt",
        map_location="cpu",
        weights_only=False,
    )
    bad_gt = torch.load(
        args.evaluation_dir / "tetrahedron_dos_cache_gt.pt",
        map_location="cpu",
        weights_only=False,
    )
    bad_pred = torch.load(
        args.evaluation_dir / "tetrahedron_dos_cache_pred.pt",
        map_location="cpu",
        weights_only=False,
    )
    ef = float(gt["fermi_level"]) * HARTREE_TO_EV
    gt_ev = gt["eigenvalues"].double().numpy() * HARTREE_TO_EV
    pred_ev = pred["eigenvalues"].double().numpy() * HARTREE_TO_EV
    openmx_ev = openmx_scf_eigenvalues(args.openmx_output)
    grid = np.linspace(-12.0, 18.0, 3001)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for sigma in (0.05, 0.1, 0.2, 0.4):
        axes[0].plot(
            grid - ef,
            gaussian_dos(gt_ev[0], grid, sigma),
            label=f"GT Γ, σ={sigma:g} eV",
        )
    axes[0].plot(
        grid - ef,
        gaussian_dos(openmx_ev, grid, 0.2),
        "k--",
        lw=1.5,
        label="OpenMX SCF table, σ=0.2 eV",
    )
    axes[0].set_title(
        "Ground-truth Γ-point DOS: broadening sensitivity and OpenMX check"
    )
    axes[0].legend(ncol=2)
    axes[1].plot(
        grid - ef,
        gaussian_dos(gt_ev.reshape(-1), grid, 0.2) / gt_ev.shape[0],
        label="GT band-path sample average",
    )
    axes[1].plot(
        np.asarray(bad_gt["grid_ev"]) - ef,
        np.asarray(bad_gt["dos"]),
        label="Original cached tetrahedron DOS (numerically invalid)",
    )
    axes[1].set_title(
        "Sampling-method comparison (log scale exposes failed tetrahedron result)"
    )
    axes[1].set_yscale("log")
    axes[1].legend()
    for ax in axes:
        ax.set_ylabel("states / eV / cell")
        ax.grid(alpha=0.25)
        ax.set_xlim(-10, 10)
    axes[-1].set_xlabel("Energy − OpenMX $E_F$ (eV)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "ground_truth_dos_methods.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for ax, sigma in zip(axes, (0.1, 0.2)):
        ax.plot(
            grid - ef,
            gaussian_dos(gt_ev[0], grid, sigma),
            label="Ground truth Γ",
            lw=1.7,
        )
        ax.plot(
            grid - ef,
            gaussian_dos(pred_ev[0], grid, sigma),
            label="Prediction Γ",
            lw=1.3,
        )
        ax.set_title(f"Γ-point Gaussian DOS, σ={sigma:g} eV")
        ax.set_ylabel("states / eV / cell")
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_xlim(-10, 10)
    axes[-1].set_xlabel("Energy − OpenMX $E_F$ (eV)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "prediction_vs_ground_truth_gamma_dos.png", dpi=180)
    plt.close(fig)

    gt_gamma = gt_ev[0]
    pred_gamma = pred_ev[0]
    delta = pred_gamma - gt_gamma
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(gt_gamma - ef, pred_gamma - ef, s=5, alpha=0.45)
    lim = [
        min(gt_gamma.min(), pred_gamma.min()) - ef,
        max(gt_gamma.max(), pred_gamma.max()) - ef,
    ]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set(
        xlabel="GT eigenvalue − $E_F$ (eV)",
        ylabel="Pred eigenvalue − $E_F$ (eV)",
        title="Γ-point ordered eigenvalues",
    )
    axes[1].plot(gt_gamma - ef, delta, ".", ms=2.5)
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set(
        xlabel="GT eigenvalue − $E_F$ (eV)",
        ylabel="Prediction − GT (eV)",
        title="Spectral error despite high matrix-element R²",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "gamma_eigenvalue_errors.png", dpi=180)
    plt.close(fig)

    occupied = gt_gamma <= ef
    summary = {
        "fermi_ev": ef,
        "basis_states": int(gt_gamma.size),
        "openmx_scf_states_parsed": int(openmx_ev.size),
        "gt_gamma_range_ev": [float(gt_gamma.min()), float(gt_gamma.max())],
        "pred_gamma_range_ev": [float(pred_gamma.min()), float(pred_gamma.max())],
        "gamma_eigenvalue_mae_ev_all": float(np.mean(np.abs(delta))),
        "gamma_eigenvalue_mae_ev_below_fermi": float(np.mean(np.abs(delta[occupied]))),
        "gamma_eigenvalue_max_error_ev": float(np.max(np.abs(delta))),
        "original_tetra_gt_integral_in_window": float(
            np.trapezoid(np.asarray(bad_gt["dos"]), np.asarray(bad_gt["grid_ev"]))
        ),
        "original_tetra_pred_integral_in_window": float(
            np.trapezoid(np.asarray(bad_pred["dos"]), np.asarray(bad_pred["grid_ev"]))
        ),
    }
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
