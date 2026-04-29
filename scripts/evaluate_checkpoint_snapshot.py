#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.factory import DatasetFactory  # noqa: E402
from data.openmx_info_parser import parse_info_out  # noqa: E402
from net.artifacts import _as_dense, _crop_dense_to_max_atoms  # noqa: E402
from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from utils.units import HARTREE_TO_EV  # noqa: E402


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Mandala checkpoint on a single snapshot."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/worthy-sweep-84/best_model.pt"),
        help="Path to the Lightning checkpoint.",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("data/big/silicon/2700K"),
        help="Snapshot directory, or a direct matrix file path.",
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=None,
        help="Optional explicit matrix file path. Overrides --snapshot-path discovery.",
    )
    parser.add_argument(
        "--info-path",
        type=Path,
        default=None,
        help="Optional explicit info file path. Overrides --snapshot-path discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/worthy-sweep-84_silicon_2700K"),
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for inference.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        help="Matrix convention passed to the dataset loader.",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=6,
        help="Number of atoms kept in the Hamiltonian matrix comparison cutout.",
    )
    parser.add_argument(
        "--dos-sigma",
        type=float,
        default=0.2,
        help="Gaussian broadening used for DOS.",
    )
    parser.add_argument(
        "--dos-bin-width",
        type=float,
        default=0.1,
        help="Energy bin width used for DOS.",
    )
    parser.add_argument(
        "--dos-energy-min",
        type=float,
        default=-7.0,
        help="Lower x-axis bound for the DOS plot in eV.",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default="Silicon 2700K",
        help="Label appended to the Hamiltonian comparison title.",
    )
    parser.add_argument(
        "--plot-clim",
        type=float,
        default=0.02,
        help="Symmetric color limit for the Hamiltonian comparison panels.",
    )
    return parser.parse_args()


def _patch_config_unpickling() -> None:
    if getattr(Config, "_mandala_legacy_unpickle_patch", False):
        return

    def __setstate__(self, state: Any) -> None:
        state_map: dict[str, Any] = {}
        if isinstance(state, tuple) and len(state) == 2:
            dict_state, slot_state = state
            if isinstance(dict_state, dict):
                state_map.update(dict_state)
            if isinstance(slot_state, dict):
                state_map.update(slot_state)
        elif isinstance(state, dict):
            state_map.update(state)

        defaults = Config()
        for name in Config.__dataclass_fields__:
            if name in state_map:
                object.__setattr__(self, name, state_map[name])
            else:
                object.__setattr__(self, name, getattr(defaults, name))

    Config.__setstate__ = __setstate__  # type: ignore[attr-defined]
    Config._mandala_legacy_unpickle_patch = True  # type: ignore[attr-defined]


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    _patch_config_unpickling()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(checkpoint)!r}")
    return checkpoint


def _restore_config(checkpoint: dict[str, Any]) -> Config:
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    cfg = hyper_parameters.get("cfg")
    if not isinstance(cfg, Config):
        raise ValueError(
            "Checkpoint does not contain a Config instance in hyper_parameters['cfg']."
        )
    cfg = copy.deepcopy(cfg)
    if isinstance(cfg.dtype, str):
        cfg.dtype = getattr(torch, cfg.dtype)
    if isinstance(cfg.matrix_targets, str):
        cfg.matrix_targets = [cfg.matrix_targets]
    return cfg


def _resolve_snapshot_cache_dir(cfg: Config, output_dir: Path) -> str | None:
    cache_dir = getattr(cfg, "snapshot_cache_dir", None)
    if not cache_dir:
        fallback = output_dir / "snapshot_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)

    cache_path = Path(cache_dir)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        test_file = cache_path / ".mandala_write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return str(cache_path)
    except OSError:
        fallback = output_dir / "snapshot_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"--- snapshot_cache_dir={cache_path} is not writable here; using local cache {fallback} ---"
        )
        return str(fallback)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def _discover_snapshot_paths(
    snapshot_path: Path, matrix_path: Path | None, info_path: Path | None
) -> tuple[Path, Path]:
    if matrix_path is not None and info_path is not None:
        return matrix_path, info_path
    if matrix_path is not None or info_path is not None:
        raise ValueError("Provide both --matrix-path and --info-path together.")

    if snapshot_path.is_file():
        matrix_candidate = snapshot_path
        info_candidates = [
            snapshot_path.parent / "info.dat",
            snapshot_path.parent / "info.txt",
            snapshot_path.parent / "SiO2.out",
        ]
        for candidate in info_candidates:
            if candidate.exists():
                return matrix_candidate, candidate
        raise FileNotFoundError(
            f"Could not infer info file next to matrix file: {snapshot_path}"
        )

    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"Snapshot path does not exist: {snapshot_path}")

    matrix_candidates = [
        snapshot_path / "Si_DM",
        snapshot_path / "HS.out",
    ]
    info_candidates = [
        snapshot_path / "info.dat",
        snapshot_path / "info.txt",
        snapshot_path / "SiO2.out",
    ]
    matrix_found = next((path for path in matrix_candidates if path.exists()), None)
    info_found = next((path for path in info_candidates if path.exists()), None)
    if matrix_found is None or info_found is None:
        raise FileNotFoundError(
            f"Could not resolve matrix/info files under snapshot directory: {snapshot_path}"
        )
    return matrix_found, info_found


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(value, device) for value in obj]
        return type(obj)(moved)
    if hasattr(obj, "to"):
        try:
            return obj.to(device)
        except TypeError:
            return obj.to(device=device)
    return obj


def _save_hamiltonian_comparison_plot(
    pred,
    target,
    output_path: Path,
    *,
    title: str,
    max_atoms: int,
    clim: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_dense = _crop_dense_to_max_atoms(
        _as_dense(pred), pred.atoms, pred.orbital_cfg, max_atoms
    )
    target_dense = _crop_dense_to_max_atoms(
        _as_dense(target), target.atoms, target.orbital_cfg, max_atoms
    )
    diff = pred_dense - target_dense
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (target_dense, "Ground Truth"),
        (pred_dense, "Prediction"),
        (diff, "Difference"),
    ]
    for ax, (mat, label) in zip(axes, panels):
        im = ax.imshow(mat.cpu().numpy(), cmap="bwr", vmin=-clim, vmax=clim)
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_dos_only_plot(
    h_pred,
    h_true,
    s_true,
    fermi_level_ev: float | None,
    num_electrons: float | None,
    output_path: Path,
    *,
    sigma: float,
    bin_width: float,
    energy_min: float,
    title: str,
) -> dict[str, float]:
    from net.artifacts import (
        compute_dos_from_eigenvalues,
        compute_generalized_eigenvalues,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    eig_pred = compute_generalized_eigenvalues(h_pred, s_true)
    eig_gt = compute_generalized_eigenvalues(h_true, s_true)
    abs_err = torch.abs(eig_pred - eig_gt)
    rel_err = abs_err / (torch.abs(eig_gt) + 1e-12)

    eig_min = float(torch.min(torch.min(eig_pred), torch.min(eig_gt)).item())
    eig_max = float(torch.max(torch.max(eig_pred), torch.max(eig_gt)).item())
    span = max(eig_max - eig_min, 1e-6)
    margin = 0.1 * span + 0.05
    e_min = max(float(energy_min), eig_min - margin)
    e_max = eig_max + margin

    grid, dos_pred = compute_dos_from_eigenvalues(
        eig_pred, sigma=sigma, bin_width=bin_width, e_min=e_min, e_max=e_max
    )
    _, dos_gt = compute_dos_from_eigenvalues(
        eig_gt, sigma=sigma, bin_width=bin_width, e_min=e_min, e_max=e_max
    )

    dx = float(grid[1] - grid[0]) if grid.numel() > 1 else float(bin_width)
    cum_gt = torch.zeros_like(dos_gt)
    cum_pred = torch.zeros_like(dos_pred)
    if grid.numel() > 1:
        cum_gt[1:] = torch.cumsum(0.5 * (dos_gt[:-1] + dos_gt[1:]) * dx, dim=0)
        cum_pred[1:] = torch.cumsum(0.5 * (dos_pred[:-1] + dos_pred[1:]) * dx, dim=0)

    electron_fill_energy_ev = None
    if num_electrons is not None and grid.numel() > 0:
        target = grid.new_tensor(num_electrons)
        fill_idx = int(torch.argmin(torch.abs(cum_gt - target)).item())
        electron_fill_energy_ev = float(grid[fill_idx].item())

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(grid.cpu().numpy(), dos_gt.cpu().numpy(), label="Ground Truth", lw=1.8)
    ax.plot(grid.cpu().numpy(), dos_pred.cpu().numpy(), label="Prediction", lw=1.4)
    ax.set_title(title)
    ax.set_xlabel("Energy")
    ax.set_ylabel("DOS")
    ax.set_xlim(left=e_min, right=e_max)
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(
        grid.cpu().numpy(),
        cum_gt.cpu().numpy(),
        color="tab:green",
        lw=1.2,
        ls="--",
        label="GT cumulative",
    )
    ax2.plot(
        grid.cpu().numpy(),
        cum_pred.cpu().numpy(),
        color="tab:orange",
        lw=1.0,
        ls="--",
        label="Pred cumulative",
    )
    ax2.set_ylabel("Integrated DOS / electrons")

    if fermi_level_ev is not None:
        ax.axvline(fermi_level_ev, color="black", ls=":", lw=1.5, label="Fermi level")
        idx = int(
            torch.argmin(torch.abs(grid - grid.new_tensor(fermi_level_ev))).item()
        )
        ax2.scatter(
            [fermi_level_ev],
            [float(cum_gt[idx].item())],
            color="tab:green",
            s=40,
            zorder=5,
        )
        ax2.scatter(
            [fermi_level_ev],
            [float(cum_pred[idx].item())],
            color="tab:orange",
            s=36,
            zorder=5,
        )
        ax.text(
            0.02,
            0.98,
            f"E_F = {fermi_level_ev:.3f} eV",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
    if num_electrons is not None:
        ax.text(
            0.02,
            0.90,
            f"N_e = {num_electrons:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
        ax2.axhline(
            num_electrons,
            color="tab:blue",
            ls=":",
            lw=1.0,
            alpha=0.75,
            label="Electron count",
        )
    if electron_fill_energy_ev is not None:
        ax.axvline(
            electron_fill_energy_ev,
            color="tab:purple",
            ls="-.",
            lw=1.5,
            label="E(N_e)",
        )
        ax.text(
            0.02,
            0.82,
            f"E(N_e) = {electron_fill_energy_ev:.3f} eV",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "eig_abs_mean": float(abs_err.mean().item()),
        "eig_abs_max": float(abs_err.max().item()),
        "eig_rel_mean": float(rel_err.mean().item()),
        "eig_rel_max": float(rel_err.max().item()),
        "electron_fill_energy_ev": electron_fill_energy_ev,
    }


def main() -> None:
    args = setup_argparse()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path, args.matrix_path, args.info_path
    )
    checkpoint = _load_checkpoint(args.checkpoint)
    cfg = _restore_config(checkpoint)
    device = _resolve_device(args.device)

    if "hamiltonian" not in cfg.matrix_targets:
        raise ValueError(
            f"Checkpoint matrix_targets={cfg.matrix_targets!r} do not include 'hamiltonian'."
        )

    # Keep evaluation simple and deterministic.
    cfg.dataset_device = None
    cfg.snapshot_cache_dir = _resolve_snapshot_cache_dir(cfg, output_dir)

    factory = DatasetFactory(cfg, convention=args.convention)
    factory.add_snapshot(matrix_path, info_path, purpose="train")
    dataset, _, mapper = factory.create()
    x, y = dataset[0]
    x = _move_to_device(x, device)
    y = _move_to_device(y, device)

    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        predictions_irreps = model(x)
        predictions_matrix = {
            name: pred.to_blocks(model.mapper)
            for name, pred in predictions_irreps.items()
        }

    h_pred = predictions_matrix["hamiltonian"]
    h_true = y["hamiltonian"]
    s_true = y["overlap"]
    info = parse_info_out(info_path)
    fermi_level_ev = None
    if info.fermi_level is not None:
        fermi_level_ev = float(info.fermi_level.item() * HARTREE_TO_EV)
    num_electrons = float(y["num_electrons"].item())

    matrix_plot_path = output_dir / "hamiltonian_first_atoms_comparison.png"
    dos_plot_path = output_dir / "dos_comparison.png"

    _save_hamiltonian_comparison_plot(
        h_pred,
        h_true,
        matrix_plot_path,
        title=f"Hamiltonian comparison: {args.plot_title}",
        max_atoms=args.max_atoms,
        clim=args.plot_clim,
    )
    dos_metrics = _save_dos_only_plot(
        h_pred,
        h_true,
        s_true,
        fermi_level_ev,
        num_electrons,
        dos_plot_path,
        sigma=args.dos_sigma,
        bin_width=args.dos_bin_width,
        energy_min=args.dos_energy_min,
        title=f"DOS comparison: {matrix_path.parent.name}",
    )

    print("checkpoint:", args.checkpoint)
    print("matrix_path:", matrix_path)
    print("info_path:", info_path)
    print("device:", device)
    print("matrix_plot:", matrix_plot_path)
    print("dos_plot:", dos_plot_path)
    print("fermi_level_ev:", fermi_level_ev)
    print("num_electrons:", num_electrons)
    print("dos_metrics:", dos_metrics)


if __name__ == "__main__":
    main()
