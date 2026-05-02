#!/usr/bin/env python

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.kspace_snapshot import build_band_path  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402
from net.artifacts import (  # noqa: E402
    compute_dos_from_eigenvalues,
    compute_generalized_eigenvalues,
)
from utils.units import HARTREE_TO_EV  # noqa: E402

DEFAULT_PATH_STRING = "GXWKGLUWLK,UX"


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and plot the ground-truth band structure for a snapshot."
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("data/big/silicon/300K"),
        help="Snapshot directory containing Si_DM and info.dat.",
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
        default=Path("eval_outputs/ground_truth_silicon_300K_band_structure"),
        help="Directory for the plot and serialized band-structure payload.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        help="Matrix convention used when loading the snapshot.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1200,
        help="Number of interpolated k-points along the path.",
    )
    parser.add_argument(
        "--path-string",
        type=str,
        default=DEFAULT_PATH_STRING,
        help="Band path string in fractional reciprocal coordinates.",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default="Ground-truth band structure",
        help="Figure title.",
    )
    parser.add_argument(
        "--emin-ev",
        type=float,
        default=-10.0,
        help="Lower plot bound in eV after subtracting the Fermi level.",
    )
    parser.add_argument(
        "--emax-ev",
        type=float,
        default=10.0,
        help="Upper plot bound in eV after subtracting the Fermi level.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Number of k-points processed per chunk during the band calculation.",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.8,
        help="Opacity of the band lines in the plot.",
    )
    parser.add_argument(
        "--dos-sigma",
        type=float,
        default=0.5,
        help="Gaussian broadening sigma in eV for the DOS plot.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached band_structure.pt and recompute the expensive band structure.",
    )
    parser.add_argument(
        "--overlap-psd-cleanup",
        action="store_true",
        help="Enable overlap PSD cleanup before the generalized eigensolve.",
    )
    parser.add_argument(
        "--overlap-jitter",
        action="store_true",
        help="Allow diagonal jitter retries if the overlap Cholesky fails.",
    )
    return parser.parse_args()


def _discover_snapshot_paths(
    snapshot_path: Path, matrix_path: Path | None, info_path: Path | None
) -> tuple[Path, Path]:
    if matrix_path is not None and info_path is not None:
        return matrix_path, info_path
    if matrix_path is not None or info_path is not None:
        raise ValueError("Provide both --matrix-path and --info-path together.")
    matrix_candidate = snapshot_path / "Si_DM"
    info_candidate = snapshot_path / "info.dat"
    if not matrix_candidate.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_candidate}")
    if not info_candidate.exists():
        raise FileNotFoundError(f"Info file not found: {info_candidate}")
    return matrix_candidate, info_candidate


def _silicon_fcc_special_points() -> dict[str, list[float]]:
    return {
        "G": [0.0, 0.0, 0.0],
        "X": [0.0, 0.5, 0.5],
        "W": [0.25, 0.75, 0.5],
        "K": [0.375, 0.75, 0.375],
        "L": [0.5, 0.5, 0.5],
        "U": [0.25, 0.625, 0.625],
    }


def _openmx_band_segments(
    info_path: Path,
) -> list[tuple[int, torch.Tensor, torch.Tensor, str, str]]:
    text = info_path.read_text(errors="ignore")
    m = re.search(r"<Band\.kpath(.*?)Band\.kpath>", text, re.S)
    if m is None:
        return []
    segments: list[tuple[int, torch.Tensor, torch.Tensor, str, str]] = []
    for raw in m.group(1).strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        npts = int(parts[0])
        start = torch.tensor(
            [float(parts[1]), float(parts[2]), float(parts[3])], dtype=torch.float64
        )
        end = torch.tensor(
            [float(parts[4]), float(parts[5]), float(parts[6])], dtype=torch.float64
        )
        start_label = str(parts[7])
        end_label = str(parts[8])
        segments.append((npts, start, end, start_label, end_label))
    return segments


def _band_path_from_openmx_info(
    info_path: Path,
) -> tuple[str, dict[str, list[float]]] | None:
    segments = _openmx_band_segments(info_path)
    if not segments:
        return None

    labels: list[str] = []
    special_points: dict[str, list[float]] = {}
    for _npts, start, end, start_label, end_label in segments:
        if not labels:
            labels.append(start_label)
        elif labels[-1] != start_label:
            labels.append(start_label)
        labels.append(end_label)
        special_points[start_label] = [float(x) for x in start.tolist()]
        special_points[end_label] = [float(x) for x in end.tolist()]

    return "".join(labels), special_points


def _save_band_structure_plot(
    payload: Any,
    output_path: Path,
    *,
    title: str,
    emin_ev: float,
    emax_ev: float,
    line_alpha: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    energies_ev = payload.eigenvalues.detach().cpu() * HARTREE_TO_EV
    fermi_level_ev = None
    if payload.fermi_level is not None:
        fermi_level_ev = float(
            payload.fermi_level.detach().cpu().item() * HARTREE_TO_EV
        )
        energies_ev = energies_ev - fermi_level_ev

    linear_k = payload.linear_k.detach().cpu()
    tick_positions = payload.tick_positions.detach().cpu()
    tick_labels = [_display_k_label(label) for label in payload.tick_labels]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.0))
    for band_idx in range(energies_ev.shape[1]):
        ax.plot(
            linear_k.numpy(),
            energies_ev[:, band_idx].numpy(),
            color="#1f5aa6",
            lw=1.1,
            alpha=line_alpha,
        )
    for xpos in tick_positions.tolist():
        ax.axvline(xpos, color="0.80", lw=0.8, zorder=0)
    ax.axhline(0.0, color="black", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlim(float(linear_k[0].item()), float(linear_k[-1].item()))
    ax.set_ylim(emin_ev, emax_ev)
    ax.set_xticks(tick_positions.numpy())
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    if fermi_level_ev is not None:
        ax.text(
            0.98,
            0.03,
            f"Fermi level = {fermi_level_ev:.3f} eV",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_dos_plot(
    grid_ev: torch.Tensor,
    dos: torch.Tensor,
    *,
    output_path: Path,
    title: str,
    num_electrons: float | None,
    fermi_level_ev: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cumulative = torch.zeros_like(grid_ev)
    if grid_ev.numel() > 1:
        cumulative[1:] = torch.cumsum(
            0.5 * (dos[:-1] + dos[1:]) * (grid_ev[1:] - grid_ev[:-1]), dim=0
        )

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.8))
    ax.plot(grid_ev.cpu().numpy(), dos.cpu().numpy(), color="#1f5aa6", lw=1.8)
    ax.set_title(title)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        grid_ev.cpu().numpy(),
        cumulative.cpu().numpy(),
        color="tab:green",
        lw=1.2,
        ls="--",
        label="Integrated DOS",
    )
    ax2.set_ylabel("Integrated DOS / electrons")

    if fermi_level_ev is not None:
        ax.axvline(fermi_level_ev, color="black", ls=":", lw=1.5, label="Fermi level")
        idx = int(
            torch.argmin(torch.abs(grid_ev - grid_ev.new_tensor(fermi_level_ev))).item()
        )
        ax2.scatter(
            [fermi_level_ev],
            [float(cumulative[idx].item())],
            color="black",
            s=36,
            zorder=5,
        )
    if num_electrons is not None:
        ax2.axhline(float(num_electrons), color="tab:green", ls=":", lw=1.0, alpha=0.8)
        ax2.text(
            0.02,
            0.95,
            f"N_e = {num_electrons:.3f}",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
    if fermi_level_ev is not None:
        ax.text(
            0.98,
            0.03,
            f"Fermi level = {fermi_level_ev:.3f} eV",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _log(message: str) -> None:
    print(message, flush=True)


def _display_k_label(label: str) -> str:
    label_str = str(label).strip()
    if label_str in {"G", "Gamma", r"$\Gamma$", "$\\Gamma$"}:
        return r"$\Gamma$"
    return label_str


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    return f"{minutes}m {remainder:.1f}s"


def _band_structure_to_payload(
    band_structure: Any,
    *,
    matrix_path: Path,
    info_path: Path,
    path_string: str,
) -> dict[str, Any]:
    return {
        "eigenvalues": band_structure.eigenvalues.detach().cpu(),
        "kpoints_abs": band_structure.kpoints_abs.detach().cpu(),
        "linear_k": band_structure.linear_k.detach().cpu(),
        "tick_positions": band_structure.tick_positions.detach().cpu(),
        "tick_labels": list(band_structure.tick_labels),
        "fractional_kpoints": (
            None
            if band_structure.fractional_kpoints is None
            else band_structure.fractional_kpoints.detach().cpu()
        ),
        "fermi_level": (
            None
            if band_structure.fermi_level is None
            else band_structure.fermi_level.detach().cpu()
        ),
        "matrix_path": str(matrix_path),
        "info_path": str(info_path),
        "path_string": path_string,
        "fermi_level_source": "dos",
        "overlap_psd_cleanup": bool(
            getattr(band_structure, "overlap_psd_cleanup", False)
        ),
        "overlap_jitter": bool(getattr(band_structure, "overlap_jitter", False)),
    }


def _band_structure_from_payload(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(
        eigenvalues=payload["eigenvalues"],
        kpoints_abs=payload["kpoints_abs"],
        linear_k=payload["linear_k"],
        tick_positions=payload["tick_positions"],
        tick_labels=list(payload["tick_labels"]),
        fractional_kpoints=payload.get("fractional_kpoints"),
        fermi_level=payload.get("fermi_level"),
    )


def _cache_payload_is_compatible(
    payload: dict[str, Any],
    *,
    path_string: str,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
) -> bool:
    cached_path = str(payload.get("path_string", "") or "")
    cached_labels = list(payload.get("tick_labels", []) or [])
    if cached_path != path_string:
        return False
    # Older cache payloads used placeholder labels like X0/X1. Recompute those.
    if cached_labels == ["X0", "X1"]:
        return False
    if bool(payload.get("overlap_psd_cleanup", False)) != bool(overlap_psd_cleanup):
        return False
    if bool(payload.get("overlap_jitter", False)) != bool(overlap_jitter):
        return False
    if str(payload.get("fermi_level_source", "") or "") != "dos":
        return False
    return True


def _fermi_level_from_dos(
    grid_ev: torch.Tensor,
    dos: torch.Tensor,
    num_electrons: float | None,
) -> float | None:
    if num_electrons is None:
        return None
    if grid_ev.numel() == 0:
        return None
    if grid_ev.numel() == 1:
        return float(grid_ev[0].item())

    cumulative = torch.zeros_like(grid_ev)
    cumulative[1:] = torch.cumsum(
        0.5 * (dos[:-1] + dos[1:]) * (grid_ev[1:] - grid_ev[:-1]), dim=0
    )
    target = float(num_electrons)
    if target <= float(cumulative[0].item()):
        return float(grid_ev[0].item())
    if target >= float(cumulative[-1].item()):
        return float(grid_ev[-1].item())

    idx = int(
        torch.searchsorted(
            cumulative, torch.tensor(target, device=grid_ev.device)
        ).item()
    )
    lo = max(idx - 1, 0)
    hi = min(idx, grid_ev.numel() - 1)
    if hi == lo:
        return float(grid_ev[lo].item())
    lo_c = float(cumulative[lo].item())
    hi_c = float(cumulative[hi].item())
    if abs(hi_c - lo_c) < 1e-12:
        return float(grid_ev[lo].item())
    t = (target - lo_c) / (hi_c - lo_c)
    return float((grid_ev[lo] + t * (grid_ev[hi] - grid_ev[lo])).item())


def _effective_dos_electron_target(snapshot: Snapshot) -> float | None:
    num_electrons = float(snapshot.get_number_of_electrons().detach().cpu().item())
    occupancies = getattr(getattr(snapshot, "info", None), "occupancies", None)
    if occupancies is None or not isinstance(occupancies, torch.Tensor):
        return num_electrons
    if occupancies.ndim != 2 or occupancies.shape[1] != 2:
        return num_electrons

    occ_cpu = occupancies.detach().cpu()
    if occ_cpu.numel() == 0:
        return num_electrons

    # OpenMX non-spin-polarized output stores identical up/down occupancies,
    # while the generalized eigensolve here works on the spatial-orbital problem.
    # In that case, integrating the DOS to N_e/2 gives the correct chemical potential.
    if torch.allclose(occ_cpu[:, 0], occ_cpu[:, 1], atol=1e-6, rtol=1e-6):
        return 0.5 * num_electrons
    return num_electrons


def _compute_dos_and_fermi(
    snapshot: Snapshot,
    *,
    psd_cleanup: bool,
    allow_jitter: bool,
    dos_sigma_ev: float,
) -> tuple[torch.Tensor, torch.Tensor, float | None, float | None, float | None]:
    eigenvalues = compute_generalized_eigenvalues(
        snapshot.hamiltonian,
        snapshot.overlap,
        psd_cleanup=psd_cleanup,
        allow_jitter=allow_jitter,
    )
    eigenvalues_ev = eigenvalues * HARTREE_TO_EV
    eig_min = float(torch.min(eigenvalues_ev).item())
    eig_max = float(torch.max(eigenvalues_ev).item())
    span = max(eig_max - eig_min, 1e-6)
    margin = 0.1 * span + 0.05
    e_min = eig_min - margin
    e_max = eig_max + margin
    grid_ev, dos = compute_dos_from_eigenvalues(
        eigenvalues_ev,
        sigma=dos_sigma_ev,
        bin_width=0.1,
        e_min=e_min,
        e_max=e_max,
    )
    num_electrons = float(snapshot.get_number_of_electrons().detach().cpu().item())
    dos_electron_target = _effective_dos_electron_target(snapshot)
    fermi_level_ev = _fermi_level_from_dos(grid_ev, dos, dos_electron_target)
    return grid_ev, dos, num_electrons, dos_electron_target, fermi_level_ev


def main() -> None:
    t0 = time.perf_counter()
    args = setup_argparse()
    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path,
        args.matrix_path,
        args.info_path,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("=== Band structure evaluation starting ===")
    _log(f"matrix_path: {matrix_path}")
    _log(f"info_path: {info_path}")
    _log(f"output_dir: {output_dir}")
    _log(f"path_string: {args.path_string}")
    _log(f"num_points: {args.num_points}")
    _log(f"chunk_size: {args.chunk_size}")
    _log(f"line_alpha: {args.line_alpha}")
    _log(f"dos_sigma_ev: {args.dos_sigma}")

    t_load = time.perf_counter()
    _log("[1/6] Loading snapshot ...")
    snapshot = Snapshot.from_openmx(
        matrix_path,
        info_path,
        convention=args.convention,
    )
    _log(
        "[1/6] Loaded snapshot in "
        f"{_format_seconds(time.perf_counter() - t_load)} "
        f"(atoms={len(snapshot.hamiltonian.atoms)}, basis={snapshot.hamiltonian.basis})"
    )

    t_path = time.perf_counter()
    _log("[2/6] Building k-path ...")
    openmx_path = _band_path_from_openmx_info(info_path)
    path_string = args.path_string
    special_points = _silicon_fcc_special_points()
    path_source = "hardcoded silicon FCC points"
    if openmx_path is not None:
        openmx_path_string, openmx_special_points = openmx_path
        special_points = openmx_special_points
        path_source = "OpenMX Band.kpath points"
        if args.path_string == DEFAULT_PATH_STRING:
            path_string = openmx_path_string

    (
        fractional_kpoints,
        kpoints_abs,
        linear_k,
        tick_positions,
        tick_labels,
    ) = build_band_path(
        snapshot.box,
        path=path_string,
        special_points=special_points,
        npoints=args.num_points,
    )
    _log(
        "[2/6] Built k-path in "
        f"{_format_seconds(time.perf_counter() - t_path)} "
        f"(num_kpoints={kpoints_abs.shape[0]}, labels={tick_labels}, source={path_source})"
    )

    t_shifts = time.perf_counter()
    _log("[3/6] Collecting translation shifts ...")
    shifts = snapshot.get_translation_shifts().to(device=snapshot.box.device)
    _log(
        "[3/6] Collected shifts in "
        f"{_format_seconds(time.perf_counter() - t_shifts)} "
        f"(num_shifts={shifts.shape[0]})"
    )

    t_dos = time.perf_counter()
    _log("[4/6] Computing DOS and Fermi level ...")
    grid_ev, dos, num_electrons, dos_electron_target, fermi_level_ev = (
        _compute_dos_and_fermi(
            snapshot,
            psd_cleanup=args.overlap_psd_cleanup,
            allow_jitter=args.overlap_jitter,
            dos_sigma_ev=args.dos_sigma,
        )
    )
    dos_path = output_dir / "dos.png"
    _save_dos_plot(
        grid_ev,
        dos,
        output_path=dos_path,
        title=f"DOS: {args.plot_title}",
        num_electrons=num_electrons,
        fermi_level_ev=fermi_level_ev,
    )
    _log(
        "[4/6] Computed DOS in "
        f"{_format_seconds(time.perf_counter() - t_dos)} "
        f"(N_e={num_electrons:.3f}, DOS_target={dos_electron_target:.3f}, E_F={fermi_level_ev:.3f} eV)"
    )

    cache_path = output_dir / "band_structure.pt"
    if cache_path.exists() and not args.force_recompute:
        t_cache = time.perf_counter()
        _log("[5/6] Loading cached band structure ...")
        cache_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if _cache_payload_is_compatible(
            cache_payload,
            path_string=path_string,
            overlap_psd_cleanup=args.overlap_psd_cleanup,
            overlap_jitter=args.overlap_jitter,
        ):
            band_structure = _band_structure_from_payload(cache_payload)
            if fermi_level_ev is not None:
                band_structure.fermi_level = torch.tensor(
                    fermi_level_ev / HARTREE_TO_EV,
                    dtype=band_structure.eigenvalues.dtype,
                    device=band_structure.eigenvalues.device,
                )
            _log(
                "[5/6] Loaded cached band structure in "
                f"{_format_seconds(time.perf_counter() - t_cache)} "
                f"(num_bands={band_structure.eigenvalues.shape[1]})"
            )
        else:
            _log("[5/6] Cached band structure is stale; recomputing ...")
            band_structure = None
    else:
        band_structure = None

    if band_structure is None:
        t_band = time.perf_counter()
        _log("[5/6] Computing chunked band structure ...")
        band_structure = snapshot.get_band_structure(
            kpoints_abs=kpoints_abs,
            fractional_kpoints=fractional_kpoints,
            linear_k=linear_k,
            tick_positions=tick_positions,
            tick_labels=tick_labels,
            shifts=shifts,
            chunk_size=args.chunk_size,
            show_progress=True,
            psd_cleanup=args.overlap_psd_cleanup,
            allow_jitter=args.overlap_jitter,
        )
        band_structure.fermi_level = (
            None
            if fermi_level_ev is None
            else torch.tensor(
                fermi_level_ev / HARTREE_TO_EV,
                dtype=band_structure.eigenvalues.dtype,
                device=band_structure.eigenvalues.device,
            )
        )
        _log(
            "[5/6] Computed chunked band structure in "
            f"{_format_seconds(time.perf_counter() - t_band)} "
            f"(num_bands={band_structure.eigenvalues.shape[1]})"
        )
        torch.save(
            _band_structure_to_payload(
                band_structure,
                matrix_path=matrix_path,
                info_path=info_path,
                path_string=path_string,
            ),
            cache_path,
        )
        _log(f"[5/6] Cached band structure at {cache_path}")

    plot_path = output_dir / "band_structure.png"
    t_plot = time.perf_counter()
    _log("[6/6] Saving plot and serialized payload ...")
    _save_band_structure_plot(
        band_structure,
        plot_path,
        title=args.plot_title,
        emin_ev=args.emin_ev,
        emax_ev=args.emax_ev,
        line_alpha=args.line_alpha,
    )
    _log("[6/6] Saved outputs in " f"{_format_seconds(time.perf_counter() - t_plot)}")

    _log("=== Band structure evaluation finished ===")
    _log(f"plot_path: {plot_path}")
    _log(f"dos_path: {dos_path}")
    _log(f"num_kpoints: {int(band_structure.eigenvalues.shape[0])}")
    _log(f"num_bands: {int(band_structure.eigenvalues.shape[1])}")
    if fermi_level_ev is not None:
        _log(f"fermi_level_ev: {fermi_level_ev:.6f}")
    _log(f"num_electrons: {num_electrons:.6f}")
    if dos_electron_target is not None:
        _log(f"dos_electron_target: {dos_electron_target:.6f}")
    _log(f"total_runtime: {_format_seconds(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
