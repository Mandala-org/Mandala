#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402
from core.orbital_irrep_config import OrbitalIrrepConfig  # noqa: E402
from data.factory import DatasetFactory  # noqa: E402
from data.openmx_info_parser import parse_info_out  # noqa: E402
from data.block_matrix import IrrepsBlockData  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402
from data.structure_inference import (  # noqa: E402
    build_model_input_from_structure,
    load_orbital_cfg_from_reference_info,
    load_structure_from_cif,
)
from data.kspace_snapshot import build_band_path  # noqa: E402
from data.kspace_snapshot import shiftspace_to_kspace_dense  # noqa: E402
from analysis import evaluation as analysis_eval  # noqa: E402
from net.artifacts import _as_dense, _crop_dense_to_max_atoms  # noqa: E402
from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from utils.units import HARTREE_TO_EV  # noqa: E402
import yaml  # noqa: E402


_BAND_MP_STATE: dict[str, Any] | None = None


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a snapshot or CIF structure with matrix, DOS, and band plots."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["snapshot", "cif"],
        help="Use a ground-truth snapshot comparison or prediction-only CIF evaluation.",
    )
    parser.add_argument("--snapshot-path", type=Path, default=None)
    parser.add_argument("--matrix-path", type=Path, default=None)
    parser.add_argument("--info-path", type=Path, default=None)
    parser.add_argument("--cif-path", type=Path, default=None)
    parser.add_argument("--reference-info-path", type=Path, default=None)
    parser.add_argument("--orbital-set", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--convention", type=str, default="e3nn")
    parser.add_argument("--max-atoms", type=int, default=6)
    parser.add_argument("--cif-max-atoms", type=int, default=8)
    parser.add_argument("--plot-clim", type=float, default=None)
    parser.add_argument("--hamiltonian-clim", type=float, default=0.05)
    parser.add_argument("--density-clim", type=float, default=0.1)
    parser.add_argument("--dos-sigma", type=float, default=0.2)
    parser.add_argument("--dos-bin-width", type=float, default=0.1)
    parser.add_argument("--dos-energy-min", type=float, default=-10.0)
    parser.add_argument("--dos-energy-max", type=float, default=15.0)
    parser.add_argument("--num-points", type=int, default=240)
    parser.add_argument("--path-string", type=str, default="GXWKGLUWLK,UX")
    parser.add_argument(
        "--use-gt-overlap-for-eigs",
        action="store_true",
        help="Use the ground-truth overlap matrix for DOS and band-structure eigensolves in snapshot mode.",
    )
    parser.add_argument("--band-emin-ev", type=float, default=-8.0)
    parser.add_argument("--band-emax-ev", type=float, default=8.0)
    parser.add_argument("--band-line-alpha", type=float, default=0.2)
    parser.add_argument("--correlation-max-points", type=int, default=250000)
    parser.add_argument("--correlation-alpha", type=float, default=0.03)
    parser.add_argument("--correlation-sample-seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--overlap-psd-cleanup",
        action="store_true",
        help="Enable overlap PSD cleanup before generalized eigensolves.",
    )
    parser.add_argument(
        "--overlap-jitter",
        action="store_true",
        help="Allow diagonal jitter retries if overlap Cholesky fails.",
    )
    parser.add_argument("--force-recompute-bands", action="store_true")
    parser.add_argument("--save-input", action="store_true")
    parser.add_argument("--plot-title", type=str, default=None)
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


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _cpu_copy(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        copied = [_cpu_copy(val) for val in value]
        return type(value)(copied)
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _discover_snapshot_paths(
    snapshot_path: Path | None, matrix_path: Path | None, info_path: Path | None
) -> tuple[Path, Path]:
    if matrix_path is not None and info_path is not None:
        return matrix_path, info_path
    if matrix_path is not None or info_path is not None:
        raise ValueError("Provide both --matrix-path and --info-path together.")
    if snapshot_path is None:
        raise ValueError("--snapshot-path is required in snapshot mode.")
    matrix_candidate = snapshot_path / "Si_DM"
    info_candidate = snapshot_path / "info.dat"
    if not matrix_candidate.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_candidate}")
    if not info_candidate.exists():
        raise FileNotFoundError(f"Info file not found: {info_candidate}")
    return matrix_candidate, info_candidate


def _resolve_orbital_cfg(args: argparse.Namespace, cfg: Config) -> OrbitalIrrepConfig:
    if args.reference_info_path is not None:
        return load_orbital_cfg_from_reference_info(
            args.reference_info_path, dtype=cfg.dtype
        )
    if args.orbital_set is not None:
        payload = yaml.safe_load(args.orbital_set)
        if not isinstance(payload, dict):
            raise ValueError(
                "--orbital-set must parse to a mapping like '{Si: 2s2p1d}'"
            )
        return OrbitalIrrepConfig.from_dict(payload)
    raise ValueError(
        "Need either --reference-info-path or --orbital-set to define the orbital basis."
    )


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


def _resolve_band_path(
    info_path: Path, requested_path_string: str
) -> tuple[str, dict[str, list[float]]]:
    resolved = _band_path_from_openmx_info(info_path)
    if resolved is None:
        raise ValueError(
            f"No OpenMX Band.kpath found in {info_path}; the hardcoded silicon FCC fallback was removed."
        )
    openmx_path_string, special_points = resolved
    return requested_path_string, special_points


def _display_k_label(label: str) -> str:
    label_str = str(label).strip()
    if label_str in {"G", "Gamma", r"$\Gamma$", "$\\Gamma$"}:
        return r"$\Gamma$"
    return label_str


def _symmetrize_block_matrix(mat):
    return 0.5 * (mat + mat.transpose())


def _clean_predicted_overlap_irreps(
    overlap_irreps: IrrepsBlockData, mapper: BlockIrrepMapper
) -> IrrepsBlockData:
    zero_labels = {"1e", "2o", "3e"}
    cleaned_vectors: dict[str, torch.Tensor] = {}
    for key, vec in overlap_irreps.pair_vectors.items():
        pair_irreps = mapper.get_pair_irreps(key)
        vec_clean = vec.clone()
        for slc, (_, ir) in zip(pair_irreps.slices(), pair_irreps):
            label = f"{ir.l}{'e' if ir.p == 1 else 'o'}"
            if label in zero_labels:
                vec_clean[..., slc] = 0
        cleaned_vectors[key] = vec_clean
    return IrrepsBlockData(
        atoms=overlap_irreps.atoms,
        atom_counts=overlap_irreps.atom_counts,
        pair_vectors=cleaned_vectors,
        pair_edges=overlap_irreps.pair_edges,
        lookup=overlap_irreps.lookup,
        orbital_cfg=overlap_irreps.orbital_cfg,
        basis=overlap_irreps.basis,
    )


def _fermi_level_from_dos(
    grid: torch.Tensor,
    dos: torch.Tensor,
    num_electrons: float | None,
) -> float | None:
    if num_electrons is None:
        return None
    if grid.numel() == 0:
        return None
    if grid.numel() == 1:
        return float(grid[0].item())
    cumulative = torch.zeros_like(grid)
    cumulative[1:] = torch.cumsum(
        0.5 * (dos[:-1] + dos[1:]) * (grid[1:] - grid[:-1]), dim=0
    )
    target = float(num_electrons)
    if target <= float(cumulative[0].item()):
        return float(grid[0].item())
    if target >= float(cumulative[-1].item()):
        return float(grid[-1].item())
    idx = int(
        torch.searchsorted(cumulative, torch.tensor(target, device=grid.device)).item()
    )
    lo = max(idx - 1, 0)
    hi = min(idx, grid.numel() - 1)
    if hi == lo:
        return float(grid[lo].item())
    lo_c = float(cumulative[lo].item())
    hi_c = float(cumulative[hi].item())
    if abs(hi_c - lo_c) < 1e-12:
        return float(grid[lo].item())
    t = (target - lo_c) / (hi_c - lo_c)
    return float((grid[lo] + t * (grid[hi] - grid[lo])).item())


def _save_comparison_plot(
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
    for ax, (mat, label) in zip(
        axes,
        [
            (target_dense, "Ground Truth"),
            (pred_dense, "Prediction"),
            (diff, "Difference"),
        ],
    ):
        im = ax.imshow(mat.cpu().numpy(), cmap="bwr", vmin=-clim, vmax=clim)
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_prediction_plot(
    mat,
    output_path: Path,
    *,
    title: str,
    max_atoms: int,
    clim: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dense = _crop_dense_to_max_atoms(
        _as_dense(mat), mat.atoms, mat.orbital_cfg, max_atoms
    )
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0))
    im = ax.imshow(dense.cpu().numpy(), cmap="bwr", vmin=-clim, vmax=clim)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_correlation_plot(
    pred,
    target,
    output_path: Path,
    *,
    title: str,
    max_points: int,
    alpha: float,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_dense = _as_dense(pred).flatten()
    target_dense = _as_dense(target).flatten()
    if pred_dense.numel() != target_dense.numel():
        raise ValueError(
            f"Correlation plot requires matching element counts, got {pred_dense.numel()} vs {target_dense.numel()}"
        )
    n = pred_dense.numel()
    if max_points > 0 and n > max_points:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(n, generator=generator)[: int(max_points)]
        pred_dense = pred_dense.index_select(0, perm)
        target_dense = target_dense.index_select(0, perm)
    pred_np = pred_dense.cpu().numpy()
    target_np = target_dense.cpu().numpy()
    stacked = np.concatenate([pred_np, target_np], axis=0)
    bound = float(np.quantile(np.abs(stacked), 0.9999))
    if not np.isfinite(bound) or bound <= 0.0:
        bound = float(max(abs(pred_np).max(), abs(target_np).max()))
    lo = -bound
    hi = bound
    corr = float(torch.corrcoef(torch.stack([target_dense, pred_dense]))[0, 1].item())

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
    ax.scatter(
        target_np,
        pred_np,
        s=3,
        alpha=alpha,
        color="#1f5aa6",
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.2, ls="--", alpha=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.text(
        0.02,
        0.98,
        f"r = {corr:.4f}\nN = {pred_np.size}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _band_structure_to_payload(
    band_structure: Any,
    *,
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
        "path_string": path_string,
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


def _band_cache_matches_options(
    payload: dict[str, Any],
    *,
    path_string: str,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
) -> bool:
    if str(payload.get("path_string", "") or "") != path_string:
        return False
    if bool(payload.get("overlap_psd_cleanup", False)) != bool(overlap_psd_cleanup):
        return False
    if bool(payload.get("overlap_jitter", False)) != bool(overlap_jitter):
        return False
    return True


def _build_band_structure_from_chunks(
    snapshot: Snapshot,
    *,
    path_string: str,
    special_points: dict[str, list[float]],
    num_points: int,
    chunk_size: int,
    num_workers: int,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
) -> Any:
    from data.kspace_snapshot import BandStructure, _generalized_eigenvalues_kspace
    from data.kspace_snapshot import block_matrix_to_shiftspace_dense

    fractional_kpoints, kpoints_abs, linear_k, tick_positions, tick_labels = (
        build_band_path(
            snapshot.box,
            path=path_string,
            special_points=special_points,
            npoints=num_points,
        )
    )
    shift_t = snapshot.get_translation_shifts().to(device=snapshot.box.device)
    if shift_t.numel() == 0:
        raise ValueError("Snapshot does not contain any translation shifts.")

    kpoints_abs = kpoints_abs.to(device=snapshot.box.device, dtype=snapshot.box.dtype)
    ham_shift = block_matrix_to_shiftspace_dense(snapshot.hamiltonian, shifts=shift_t)
    ovl_shift = block_matrix_to_shiftspace_dense(snapshot.overlap, shifts=shift_t)

    nk = int(kpoints_abs.shape[0])
    effective_chunk = nk if chunk_size <= 0 else min(int(chunk_size), nk)
    tasks = [
        (start, min(start + effective_chunk, nk))
        for start in range(0, nk, effective_chunk)
    ]

    if num_workers <= 1 or len(tasks) <= 1:
        eig_chunks = []
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - optional dependency
            tqdm = None
        task_iter = tasks
        if tqdm is not None and len(tasks) > 1:
            task_iter = tqdm(tasks, total=len(tasks), desc="Band chunks")
        for start, stop in task_iter:
            k_chunk = kpoints_abs[start:stop]
            ham_k = shiftspace_to_kspace_dense(
                ham_shift,
                kpoints_abs=k_chunk,
                shifts=shift_t,
                box=snapshot.box,
            )
            ovl_k = shiftspace_to_kspace_dense(
                ovl_shift,
                kpoints_abs=k_chunk,
                shifts=shift_t,
                box=snapshot.box,
            )
            eig_chunks.append(
                _generalized_eigenvalues_kspace(
                    ham_k,
                    ovl_k,
                    psd_cleanup=overlap_psd_cleanup,
                    allow_jitter=overlap_jitter,
                ).cpu()
            )
    else:
        eig_chunks = _compute_band_chunks_parallel(
            ham_shift=ham_shift.detach().cpu(),
            ovl_shift=ovl_shift.detach().cpu(),
            shift_t=shift_t.detach().cpu(),
            box=snapshot.box.detach().cpu(),
            kpoints_abs=kpoints_abs.detach().cpu(),
            tasks=tasks,
            num_workers=num_workers,
            overlap_psd_cleanup=overlap_psd_cleanup,
            overlap_jitter=overlap_jitter,
        )

    eigenvalues = torch.cat(eig_chunks, dim=0).to(dtype=ham_shift.dtype)
    linear_axis = linear_k.to(dtype=kpoints_abs.dtype)

    fermi_level = None
    if getattr(snapshot.info, "fermi_level", None) is not None:
        fermi_level = snapshot.info.fermi_level.to(dtype=eigenvalues.real.dtype).cpu()

    return BandStructure(
        eigenvalues=eigenvalues,
        kpoints_abs=kpoints_abs.detach().cpu(),
        linear_k=linear_axis.detach().cpu(),
        tick_positions=tick_positions.to(dtype=kpoints_abs.dtype).detach().cpu(),
        tick_labels=list(tick_labels),
        fractional_kpoints=fractional_kpoints.detach().cpu(),
        fermi_level=fermi_level,
    )


def _band_worker_init() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _band_chunk_worker(task: tuple[int, int]) -> torch.Tensor:
    from data.kspace_snapshot import _generalized_eigenvalues_kspace

    if _BAND_MP_STATE is None:
        raise RuntimeError("Band worker state is not initialized.")
    start, stop = task
    k_chunk = _BAND_MP_STATE["kpoints_abs"][start:stop]
    ham_k = shiftspace_to_kspace_dense(
        _BAND_MP_STATE["ham_shift"],
        kpoints_abs=k_chunk,
        shifts=_BAND_MP_STATE["shift_t"],
        box=_BAND_MP_STATE["box"],
    )
    ovl_k = shiftspace_to_kspace_dense(
        _BAND_MP_STATE["ovl_shift"],
        kpoints_abs=k_chunk,
        shifts=_BAND_MP_STATE["shift_t"],
        box=_BAND_MP_STATE["box"],
    )
    return _generalized_eigenvalues_kspace(
        ham_k,
        ovl_k,
        psd_cleanup=bool(_BAND_MP_STATE["overlap_psd_cleanup"]),
        allow_jitter=bool(_BAND_MP_STATE["overlap_jitter"]),
    ).cpu()


def _compute_band_chunks_parallel(
    *,
    ham_shift: torch.Tensor,
    ovl_shift: torch.Tensor,
    shift_t: torch.Tensor,
    box: torch.Tensor,
    kpoints_abs: torch.Tensor,
    tasks: list[tuple[int, int]],
    num_workers: int,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
) -> list[torch.Tensor]:
    global _BAND_MP_STATE

    worker_count = max(1, min(int(num_workers), len(tasks)))
    if worker_count == 1:
        _BAND_MP_STATE = {
            "ham_shift": ham_shift,
            "ovl_shift": ovl_shift,
            "shift_t": shift_t,
            "box": box,
            "kpoints_abs": kpoints_abs,
            "overlap_psd_cleanup": overlap_psd_cleanup,
            "overlap_jitter": overlap_jitter,
        }
        try:
            return [_band_chunk_worker(task) for task in tasks]
        finally:
            _BAND_MP_STATE = None

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print(
            "--- multiprocessing fork context unavailable; falling back to serial band chunks ---"
        )
        _BAND_MP_STATE = {
            "ham_shift": ham_shift,
            "ovl_shift": ovl_shift,
            "shift_t": shift_t,
            "box": box,
            "kpoints_abs": kpoints_abs,
            "overlap_psd_cleanup": overlap_psd_cleanup,
            "overlap_jitter": overlap_jitter,
        }
        try:
            return [_band_chunk_worker(task) for task in tasks]
        finally:
            _BAND_MP_STATE = None

    _BAND_MP_STATE = {
        "ham_shift": ham_shift,
        "ovl_shift": ovl_shift,
        "shift_t": shift_t,
        "box": box,
        "kpoints_abs": kpoints_abs,
        "overlap_psd_cleanup": overlap_psd_cleanup,
        "overlap_jitter": overlap_jitter,
    }
    try:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - optional dependency
            tqdm = None
        with ctx.Pool(worker_count, initializer=_band_worker_init) as pool:
            result_iter = pool.imap(_band_chunk_worker, tasks, chunksize=1)
            if tqdm is not None and len(tasks) > 1:
                result_iter = tqdm(result_iter, total=len(tasks), desc="Band chunks")
            return list(result_iter)
    finally:
        _BAND_MP_STATE = None


def _compute_or_load_band_structure(
    snapshot: Snapshot,
    cache_path: Path,
    *,
    path_string: str,
    special_points: dict[str, list[float]],
    num_points: int,
    chunk_size: int,
    num_workers: int,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
    force_recompute: bool,
) -> Any:
    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if _band_cache_matches_options(
            payload,
            path_string=path_string,
            overlap_psd_cleanup=overlap_psd_cleanup,
            overlap_jitter=overlap_jitter,
        ):
            return _band_structure_from_payload(payload)
    band_structure = _build_band_structure_from_chunks(
        snapshot,
        path_string=path_string,
        special_points=special_points,
        num_points=num_points,
        chunk_size=chunk_size,
        num_workers=num_workers,
        overlap_psd_cleanup=overlap_psd_cleanup,
        overlap_jitter=overlap_jitter,
    )
    band_structure.overlap_psd_cleanup = bool(overlap_psd_cleanup)
    band_structure.overlap_jitter = bool(overlap_jitter)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _band_structure_to_payload(band_structure, path_string=path_string), cache_path
    )
    return band_structure


def _band_cache_path(
    output_dir: Path,
    *,
    kind: str,
    use_gt_overlap_for_eigs: bool,
) -> Path:
    suffix = "_gt_overlap" if use_gt_overlap_for_eigs else ""
    return output_dir / f"band_structure_{kind}{suffix}.pt"


def _save_band_structure_comparison_plot(
    gt_band: Any,
    pred_band: Any,
    output_path: Path,
    *,
    title: str,
    emin_ev: float,
    emax_ev: float,
    line_alpha: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gt_e = gt_band.eigenvalues.detach().cpu() * HARTREE_TO_EV
    pred_e = pred_band.eigenvalues.detach().cpu() * HARTREE_TO_EV
    if gt_band.fermi_level is not None:
        gt_e = gt_e - float(gt_band.fermi_level.detach().cpu().item() * HARTREE_TO_EV)
    if pred_band.fermi_level is not None:
        pred_e = pred_e - float(
            pred_band.fermi_level.detach().cpu().item() * HARTREE_TO_EV
        )
    linear_k = gt_band.linear_k.detach().cpu()
    tick_positions = gt_band.tick_positions.detach().cpu()
    tick_labels = [_display_k_label(label) for label in gt_band.tick_labels]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    for ax, energies, color, subtitle in [
        (axes[0], gt_e, "black", "Ground truth"),
        (axes[1], pred_e, "#1f5aa6", "Prediction"),
    ]:
        for idx in range(energies.shape[1]):
            ax.plot(
                linear_k.numpy(),
                energies[:, idx].numpy(),
                color=color,
                lw=1.0 if color != "black" else 1.2,
                alpha=line_alpha,
            )
        for xpos in tick_positions.tolist():
            ax.axvline(xpos, color="0.82", lw=0.8, zorder=0)
        ax.axhline(0.0, color="black", ls="--", lw=1.0, alpha=0.7)
        ax.set_xlim(float(linear_k[0].item()), float(linear_k[-1].item()))
        ax.set_ylim(emin_ev, emax_ev)
        ax.set_xticks(tick_positions.numpy())
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.set_title(subtitle)
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].set_ylabel(r"$E - E_F$ (eV)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_band_structure_prediction_plot(
    band: Any,
    output_path: Path,
    *,
    title: str,
    emin_ev: float,
    emax_ev: float,
    line_alpha: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energies = band.eigenvalues.detach().cpu() * HARTREE_TO_EV
    if band.fermi_level is not None:
        energies = energies - float(
            band.fermi_level.detach().cpu().item() * HARTREE_TO_EV
        )
    linear_k = band.linear_k.detach().cpu()
    tick_positions = band.tick_positions.detach().cpu()
    tick_labels = [_display_k_label(label) for label in band.tick_labels]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.0))
    for idx in range(energies.shape[1]):
        ax.plot(
            linear_k.numpy(),
            energies[:, idx].numpy(),
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
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _compute_dos_data(
    h_mat,
    s_mat,
    *,
    sigma: float,
    bin_width: float,
    energy_min: float,
    energy_max: float,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
):
    from net.artifacts import (
        compute_dos_from_eigenvalues,
        compute_generalized_eigenvalues,
    )

    eig = (
        compute_generalized_eigenvalues(
            h_mat,
            s_mat,
            psd_cleanup=overlap_psd_cleanup,
            allow_jitter=overlap_jitter,
        )
        .detach()
        .cpu()
        * HARTREE_TO_EV
    )
    grid, dos = compute_dos_from_eigenvalues(
        eig,
        sigma=sigma,
        bin_width=bin_width,
        e_min=float(energy_min),
        e_max=float(energy_max),
    )
    return eig, eig, grid, dos


def _save_dos_comparison_plot(
    h_pred,
    s_pred,
    h_true,
    s_true,
    num_electrons_true: float | None,
    num_electrons_pred: float | None,
    output_path: Path,
    *,
    sigma: float,
    bin_width: float,
    energy_min: float,
    energy_max: float,
    title: str,
    error_output_path: Path | None = None,
    overlap_psd_cleanup: bool = False,
    overlap_jitter: bool = False,
) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eig_pred, _eig_pred_window, grid_pred, dos_pred = _compute_dos_data(
        h_pred,
        s_pred,
        sigma=sigma,
        bin_width=bin_width,
        energy_min=energy_min,
        energy_max=energy_max,
        overlap_psd_cleanup=overlap_psd_cleanup,
        overlap_jitter=overlap_jitter,
    )
    eig_true, _eig_true_window, grid_true, dos_true = _compute_dos_data(
        h_true,
        s_true,
        sigma=sigma,
        bin_width=bin_width,
        energy_min=energy_min,
        energy_max=energy_max,
        overlap_psd_cleanup=overlap_psd_cleanup,
        overlap_jitter=overlap_jitter,
    )

    min_len = min(eig_pred.numel(), eig_true.numel())
    abs_err = torch.abs(eig_pred[:min_len] - eig_true[:min_len])
    rel_err = abs_err / (torch.abs(eig_true[:min_len]) + 1e-12)
    fermi_true = _fermi_level_from_dos(grid_true, dos_true, num_electrons_true)
    fermi_pred = _fermi_level_from_dos(grid_pred, dos_pred, num_electrons_pred)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(
        grid_true.cpu().numpy(), dos_true.cpu().numpy(), label="Ground Truth", lw=1.8
    )
    ax.plot(grid_pred.cpu().numpy(), dos_pred.cpu().numpy(), label="Prediction", lw=1.4)
    if fermi_true is not None:
        ax.axvline(
            fermi_true,
            color="black",
            ls="--",
            lw=1.2,
            alpha=0.85,
            label=f"GT $E_F$ = {fermi_true:.3f} eV",
        )
    if fermi_pred is not None:
        ax.axvline(
            fermi_pred,
            color="#1f5aa6",
            ls=":",
            lw=1.4,
            alpha=0.9,
            label=f"Pred $E_F$ = {fermi_pred:.3f} eV",
        )
    ax.set_title(title)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=energy_min, right=energy_max)
    text_lines = []
    if num_electrons_true is not None:
        text_lines.append(f"GT N_e = {num_electrons_true:.3f}")
    if num_electrons_pred is not None:
        text_lines.append(f"Pred N_e = {num_electrons_pred:.3f}")
    if text_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if error_output_path is not None:
        error_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
        dos_error = dos_pred - dos_true
        ax.plot(
            grid_true.cpu().numpy(),
            dos_error.cpu().numpy(),
            color="#b23a48",
            lw=1.4,
            label="Prediction - Ground Truth",
        )
        ax.axhline(0.0, color="black", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(title.replace("comparison", "error"))
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("DOS Error")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=energy_min, right=energy_max)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(error_output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return {
        "eig_abs_mean": float(abs_err.mean().item()),
        "eig_abs_max": float(abs_err.max().item()),
        "eig_rel_mean": float(rel_err.mean().item()),
        "eig_rel_max": float(rel_err.max().item()),
        "fermi_gt_ev": float(fermi_true) if fermi_true is not None else float("nan"),
        "fermi_pred_ev": float(fermi_pred) if fermi_pred is not None else float("nan"),
    }


def _save_dos_prediction_plot(
    h_pred,
    s_pred,
    output_path: Path,
    *,
    sigma: float,
    bin_width: float,
    energy_min: float,
    energy_max: float,
    num_electrons: float | None,
    title: str,
    overlap_psd_cleanup: bool = False,
    overlap_jitter: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, _, grid, dos = _compute_dos_data(
        h_pred,
        s_pred,
        sigma=sigma,
        bin_width=bin_width,
        energy_min=energy_min,
        energy_max=energy_max,
        overlap_psd_cleanup=overlap_psd_cleanup,
        overlap_jitter=overlap_jitter,
    )
    fermi = _fermi_level_from_dos(grid, dos, num_electrons)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(grid.cpu().numpy(), dos.cpu().numpy(), lw=1.6, color="#1f5aa6")
    if fermi is not None:
        ax.axvline(
            fermi,
            color="#1f5aa6",
            ls=":",
            lw=1.4,
            alpha=0.9,
            label=f"$E_F$ = {fermi:.3f} eV",
        )
    ax.set_title(title)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=energy_min, right=energy_max)
    if num_electrons is not None:
        ax.text(
            0.02,
            0.98,
            f"N_e = {num_electrons:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
    if fermi is not None:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_snapshot_from_matrices(
    mats: dict[str, Any], *, positions, box, info=None
) -> Snapshot:
    return Snapshot(
        mats["hamiltonian"],
        mats["overlap"],
        mats["density"],
        positions=positions,
        box=box,
        info=info,
    )


def _run_snapshot_case(
    args: argparse.Namespace, checkpoint: dict[str, Any], cfg: Config
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path, args.matrix_path, args.info_path
    )

    cfg.dataset_device = None
    cfg.snapshot_cache_dir = _resolve_snapshot_cache_dir(cfg, output_dir)
    factory = DatasetFactory(cfg, convention=args.convention)
    factory.add_snapshot(matrix_path, info_path, purpose="train")
    dataset, _, mapper = factory.create()
    x, y = dataset[0]
    device = _resolve_device(args.device)
    x = _move_to_device(x, device)
    y = _move_to_device(y, device)

    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions_irreps = model(x)
        if "overlap" in predictions_irreps:
            predictions_irreps["overlap"] = _clean_predicted_overlap_irreps(
                predictions_irreps["overlap"], model.mapper
            )
        pred_mats = {
            name: _symmetrize_block_matrix(pred.to_blocks(model.mapper))
            for name, pred in predictions_irreps.items()
        }

    gt_mats = {name: y[name] for name in ("hamiltonian", "density", "overlap")}
    info = parse_info_out(info_path)
    positions = x["positions"]
    box = x["box"]
    gt_snapshot = _build_snapshot_from_matrices(
        gt_mats, positions=positions, box=box, info=info
    )
    resolved_path_string, special_points = analysis_eval.resolve_band_path(
        info_path, args.path_string
    )
    overlap_for_eigs = (
        gt_mats["overlap"] if args.use_gt_overlap_for_eigs else pred_mats.get("overlap")
    )
    if overlap_for_eigs is None:
        raise ValueError(
            "Checkpoint does not predict overlap and --use-gt-overlap-for-eigs was not set."
        )
    density_for_eigs = pred_mats.get("density", gt_mats["density"])
    pred_band_snapshot = _build_snapshot_from_matrices(
        {
            "hamiltonian": pred_mats["hamiltonian"],
            "density": density_for_eigs,
            "overlap": overlap_for_eigs,
        },
        positions=positions,
        box=box,
        info=info,
    )

    title = args.plot_title or matrix_path.parent.name
    ham_clim = (
        args.hamiltonian_clim
        if args.hamiltonian_clim is not None
        else (args.plot_clim if args.plot_clim is not None else 0.05)
    )
    density_clim = (
        args.density_clim
        if args.density_clim is not None
        else (args.plot_clim if args.plot_clim is not None else 0.1)
    )
    analysis_eval.save_comparison_plot(
        pred_mats["hamiltonian"],
        gt_mats["hamiltonian"],
        output_dir / "hamiltonian_first_atoms_comparison.png",
        title=f"Hamiltonian comparison: {title}",
        max_atoms=args.max_atoms,
        clim=ham_clim,
    )
    analysis_eval.save_correlation_plot(
        pred_mats["hamiltonian"],
        gt_mats["hamiltonian"],
        output_dir / "hamiltonian_correlation.png",
        title=f"Hamiltonian correlation: {title}",
        max_points=args.correlation_max_points,
        alpha=args.correlation_alpha,
        seed=args.correlation_sample_seed,
    )
    if "density" in pred_mats:
        analysis_eval.save_comparison_plot(
            pred_mats["density"],
            gt_mats["density"],
            output_dir / "density_first_atoms_comparison.png",
            title=f"Density comparison: {title}",
            max_atoms=args.max_atoms,
            clim=density_clim,
        )
        analysis_eval.save_correlation_plot(
            pred_mats["density"],
            gt_mats["density"],
            output_dir / "density_correlation.png",
            title=f"Density correlation: {title}",
            max_points=args.correlation_max_points,
            alpha=args.correlation_alpha,
            seed=args.correlation_sample_seed,
        )
    else:
        print(
            "--- Density prediction unavailable; skipping density comparison plot ---"
        )
    if "overlap" in pred_mats:
        analysis_eval.save_correlation_plot(
            pred_mats["overlap"],
            gt_mats["overlap"],
            output_dir / "overlap_correlation.png",
            title=f"Overlap correlation: {title}",
            max_points=args.correlation_max_points,
            alpha=args.correlation_alpha,
            seed=args.correlation_sample_seed,
        )
    num_electrons_true = (
        float(gt_snapshot.get_number_of_electrons().item())
        if hasattr(gt_snapshot, "get_number_of_electrons")
        else (float(y["num_electrons"].item()) if "num_electrons" in y else None)
    )
    num_electrons_pred = (
        num_electrons_true
        if args.use_gt_overlap_for_eigs
        else (
            float(pred_band_snapshot.get_number_of_electrons().item())
            if hasattr(pred_band_snapshot, "get_number_of_electrons")
            else None
        )
    )
    dos_metrics = analysis_eval.save_dos_comparison_plot(
        pred_mats["hamiltonian"],
        overlap_for_eigs,
        gt_mats["hamiltonian"],
        gt_mats["overlap"],
        num_electrons_true,
        num_electrons_pred,
        output_dir / "dos_comparison.png",
        sigma=args.dos_sigma,
        bin_width=args.dos_bin_width,
        energy_min=args.dos_energy_min,
        energy_max=args.dos_energy_max,
        title=f"DOS comparison: {title}",
        error_output_path=output_dir / "dos_error.png",
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
    )

    gt_band = analysis_eval.compute_or_load_band_structure(
        gt_snapshot,
        analysis_eval.band_cache_path(
            output_dir,
            kind="gt",
            use_gt_overlap_for_eigs=args.use_gt_overlap_for_eigs,
        ),
        path_string=resolved_path_string,
        special_points=special_points,
        num_points=args.num_points,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
        force_recompute=args.force_recompute_bands,
    )
    pred_band = analysis_eval.compute_or_load_band_structure(
        pred_band_snapshot,
        analysis_eval.band_cache_path(
            output_dir,
            kind="pred",
            use_gt_overlap_for_eigs=args.use_gt_overlap_for_eigs,
        ),
        path_string=resolved_path_string,
        special_points=special_points,
        num_points=args.num_points,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
        force_recompute=args.force_recompute_bands,
    )
    analysis_eval.save_band_structure_comparison_plot(
        gt_band,
        pred_band,
        output_dir / "band_structure_comparison.png",
        title=f"Band structure comparison: {title}",
        emin_ev=args.band_emin_ev,
        emax_ev=args.band_emax_ev,
        line_alpha=args.band_line_alpha,
    )

    for name, block in pred_mats.items():
        block.save(output_dir / f"pred_{name}.pt")
    print("mode: snapshot")
    print("checkpoint:", args.checkpoint)
    print("matrix_path:", matrix_path)
    print("info_path:", info_path)
    print("output_dir:", output_dir)
    print("dos_metrics:", dos_metrics)


def _run_cif_case(
    args: argparse.Namespace, checkpoint: dict[str, Any], cfg: Config
) -> None:
    if args.cif_path is None:
        raise ValueError("--cif-path is required in cif mode.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    orbital_cfg = _resolve_orbital_cfg(args, cfg)
    mapper = BlockIrrepMapper(
        orbital_cfg,
        diagonal=False,
        device="cpu",
        dtype=cfg.dtype,
    )
    atoms, positions, box = load_structure_from_cif(
        args.cif_path,
        dtype=cfg.dtype,
        device=device,
    )
    if args.reference_info_path is None:
        raise ValueError(
            "CIF mode now requires --reference-info-path so the OpenMX Band.kpath can be reused; the hardcoded silicon FCC fallback was removed."
        )
    resolved_path_string, special_points = analysis_eval.resolve_band_path(
        args.reference_info_path, args.path_string
    )
    x = build_model_input_from_structure(
        atoms=atoms,
        positions=positions,
        box=box,
        cfg=cfg,
        mapper=mapper,
    )
    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions_irreps = model(x)
        if "overlap" in predictions_irreps:
            predictions_irreps["overlap"] = _clean_predicted_overlap_irreps(
                predictions_irreps["overlap"], model.mapper
            )
        pred_mats = {
            name: _symmetrize_block_matrix(pred.to_blocks(model.mapper))
            for name, pred in predictions_irreps.items()
        }

    title = args.plot_title or args.cif_path.stem
    ham_clim = (
        args.hamiltonian_clim
        if args.hamiltonian_clim is not None
        else (args.plot_clim if args.plot_clim is not None else 0.05)
    )
    density_clim = (
        args.density_clim
        if args.density_clim is not None
        else (args.plot_clim if args.plot_clim is not None else 0.1)
    )
    analysis_eval.save_prediction_plot(
        pred_mats["hamiltonian"],
        output_dir / "hamiltonian_first_atoms_prediction.png",
        title=f"Hamiltonian prediction: {title}",
        max_atoms=args.cif_max_atoms,
        clim=ham_clim,
    )
    if "density" in pred_mats:
        analysis_eval.save_prediction_plot(
            pred_mats["density"],
            output_dir / "density_first_atoms_prediction.png",
            title=f"Density prediction: {title}",
            max_atoms=args.cif_max_atoms,
            clim=density_clim,
        )
    else:
        raise ValueError(
            "CIF evaluation currently requires a predicted density matrix for DOS/band plots."
        )
    pred_snapshot_for_eigs = _build_snapshot_from_matrices(
        {
            "hamiltonian": pred_mats["hamiltonian"],
            "density": pred_mats["density"],
            "overlap": pred_mats.get("overlap"),
        },
        positions=positions,
        box=box,
    )
    if pred_snapshot_for_eigs.overlap is None:
        raise ValueError(
            "CIF evaluation needs a predicted overlap matrix for DOS/band plots."
        )
    num_electrons_pred = float(pred_snapshot_for_eigs.get_number_of_electrons().item())
    analysis_eval.save_dos_prediction_plot(
        pred_mats["hamiltonian"],
        pred_snapshot_for_eigs.overlap,
        output_dir / "dos_prediction.png",
        sigma=args.dos_sigma,
        bin_width=args.dos_bin_width,
        energy_min=args.dos_energy_min,
        energy_max=args.dos_energy_max,
        num_electrons=num_electrons_pred,
        title=f"DOS prediction: {title}",
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
    )
    pred_band = analysis_eval.compute_or_load_band_structure(
        pred_snapshot_for_eigs,
        output_dir / "band_structure_pred.pt",
        path_string=resolved_path_string,
        special_points=special_points,
        num_points=args.num_points,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
        force_recompute=args.force_recompute_bands,
    )
    analysis_eval.save_band_structure_prediction_plot(
        pred_band,
        output_dir / "band_structure_prediction.png",
        title=f"Band structure prediction: {title}",
        emin_ev=args.band_emin_ev,
        emax_ev=args.band_emax_ev,
        line_alpha=args.band_line_alpha,
    )
    torch.save(
        {
            "atoms": list(atoms),
            "positions": positions.detach().cpu(),
            "box": box.detach().cpu(),
            "orbital_cfg": orbital_cfg.to_dict(),
            "checkpoint": str(args.checkpoint),
            "cif_path": str(args.cif_path),
            "matrix_targets": list(cfg.matrix_targets),
        },
        output_dir / "structure_metadata.pt",
    )
    if args.save_input:
        torch.save(_cpu_copy(x), output_dir / "model_input.pt")
    for name, block in pred_mats.items():
        block.save(output_dir / f"pred_{name}.pt")
    print("mode: cif")
    print("checkpoint:", args.checkpoint)
    print("cif_path:", args.cif_path)
    print("output_dir:", output_dir)


def main() -> None:
    args = setup_argparse()
    checkpoint = _load_checkpoint(args.checkpoint)
    cfg = _restore_config(checkpoint)
    cfg.dataset_device = None
    cfg.snapshot_cache_dir = None

    if "hamiltonian" not in set(cfg.matrix_targets):
        raise ValueError(
            f"Checkpoint matrix_targets={cfg.matrix_targets!r} do not include hamiltonian."
        )

    if args.mode == "snapshot":
        _run_snapshot_case(args, checkpoint, cfg)
    else:
        _run_cif_case(args, checkpoint, cfg)


if __name__ == "__main__":
    main()
