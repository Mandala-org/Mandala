from __future__ import annotations

import itertools
import json
import random
import sys
import multiprocessing as mp
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import IrrepsBlockData
from data.kspace_snapshot import (
    BandStructure,
    _generalized_eigenvalues_kspace,
    block_matrix_to_shiftspace_dense,
    build_band_path,
    shiftspace_to_kspace_dense,
)
from data.snapshot import Snapshot
from net.artifacts import compute_dos_from_eigenvalues, compute_generalized_eigenvalues
from utils.units import HARTREE_TO_EV

DEFAULT_PATH_STRING = "GXWKGLUWLK,UX"

_BAND_MP_STATE: dict[str, Any] | None = None
_TETRA_MP_STATE: dict[str, Any] | None = None


def openmx_band_segments(
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


def band_path_from_openmx_info(
    info_path: Path,
) -> tuple[str, dict[str, list[float]]] | None:
    segments = openmx_band_segments(info_path)
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


def resolve_band_path(
    box: torch.Tensor | None,
    info_path: Path | None,
    requested_path_string: str | None,
    special_points_override: dict[str, list[float]] | None = None,
) -> tuple[str, dict[str, list[float]] | None]:
    if special_points_override is not None:
        if not special_points_override:
            raise ValueError("special_points_override must not be empty")
        if requested_path_string is None:
            raise ValueError(
                "special_points_override requires an explicit --path-string"
            )
        return requested_path_string, special_points_override
    if info_path is not None:
        resolved = band_path_from_openmx_info(info_path)
        if resolved is None:
            raise ValueError(f"No OpenMX Band.kpath found in {info_path}")
        openmx_path_string, special_points = resolved
        if requested_path_string is None:
            return openmx_path_string, special_points
        return requested_path_string, special_points
    if requested_path_string is None:
        if box is None:
            raise ValueError("Need a snapshot box to resolve the default ASE band path")
        default_path_string, *_ = build_band_path(box, npoints=2)
        return default_path_string, None
    return requested_path_string, None


def display_k_label(label: str) -> str:
    label_str = str(label).strip()
    if label_str in {"G", "Gamma", r"$\Gamma$", "$\\Gamma$"}:
        return "Γ"
    return label_str


def _clip_energy_curve(
    energy: torch.Tensor,
    values: torch.Tensor,
    *,
    energy_min: float,
    energy_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (energy >= float(energy_min)) & (energy <= float(energy_max))
    if not torch.any(mask):
        return energy, values
    return energy[mask], values[mask]


def load_dos_reference(
    dos_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    data = np.loadtxt(dos_path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected DOS reference format in {dos_path}")
    energies = torch.tensor(data[:, 0], dtype=torch.float64)
    dos = torch.tensor(data[:, 1], dtype=torch.float64)
    cumulative = (
        torch.tensor(data[:, 2], dtype=torch.float64) if data.shape[1] >= 3 else None
    )
    return energies, dos, cumulative


def parse_kmesh_spec(spec: str) -> tuple[int, int, int]:
    raw = str(spec).strip().lower().replace(" ", "")
    parts = raw.split("x")
    if len(parts) != 3:
        raise ValueError(f"Expected kmesh like '4x4x4', got {spec!r}")
    kmesh = tuple(int(part) for part in parts)
    if min(kmesh) <= 0:
        raise ValueError(f"kmesh entries must be positive, got {spec!r}")
    return kmesh


def fractional_kmesh_points(
    kmesh: tuple[int, int, int], *, device, dtype
) -> torch.Tensor:
    grids = [torch.arange(n, device=device, dtype=dtype) / float(n) for n in kmesh]
    points = []
    for vals in itertools.product(*grids):
        points.append(
            torch.tensor(
                [float(vals[0]), float(vals[1]), float(vals[2])],
                device=device,
                dtype=dtype,
            )
        )
    return torch.stack(points, dim=0)


def symmetrize_block_matrix(mat):
    return 0.5 * (mat + mat.transpose())


def clean_predicted_overlap_irreps(
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


def effective_dos_electron_target(snapshot: Snapshot) -> float | None:
    num_electrons = float(snapshot.get_number_of_electrons().detach().cpu().item())
    occupancies = getattr(getattr(snapshot, "info", None), "occupancies", None)
    if occupancies is None or not isinstance(occupancies, torch.Tensor):
        return num_electrons
    if occupancies.ndim != 2 or occupancies.shape[1] != 2:
        return num_electrons

    occ_cpu = occupancies.detach().cpu()
    if occ_cpu.numel() == 0:
        return num_electrons
    if torch.allclose(occ_cpu[:, 0], occ_cpu[:, 1], atol=1e-6, rtol=1e-6):
        return 0.5 * num_electrons
    return num_electrons


def infer_spin_factor(snapshot: Snapshot) -> float:
    """Infer whether the eigenproblem is spatial-orbital non-spin-polarized."""
    occupancies = getattr(getattr(snapshot, "info", None), "occupancies", None)
    if not isinstance(occupancies, torch.Tensor):
        return 1.0

    occ_cpu = occupancies.detach().cpu()
    if occ_cpu.ndim == 2 and occ_cpu.shape[1] == 2 and occ_cpu.numel() > 0:
        if torch.allclose(occ_cpu[:, 0], occ_cpu[:, 1], atol=1e-6, rtol=1e-6):
            return 2.0
    return 1.0


def fermi_level_from_dos(
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


def _tetrahedron_cdf_pdf(
    energies: torch.Tensor,
    grid_ev: torch.Tensor,
    *,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    if energies.shape[-1] != 4:
        raise ValueError(
            f"Expected last dimension of energies to be 4, got {energies.shape}"
        )

    e, _ = torch.sort(energies, dim=-1)

    offsets = torch.tensor(
        [-1.5, -0.5, 0.5, 1.5],
        dtype=e.dtype,
        device=e.device,
    )
    scale = torch.clamp(torch.max(torch.abs(e), dim=-1, keepdim=True).values, min=1.0)
    e = e + offsets * eps * scale

    x = grid_ev.to(dtype=e.dtype, device=e.device)
    x = x.reshape((1,) * (e.ndim - 1) + (-1,))

    cdf = torch.zeros(e.shape[:-1] + (grid_ev.numel(),), dtype=e.dtype, device=e.device)
    pdf = torch.zeros_like(cdf)

    for i in range(4):
        ei = e[..., i]
        denom = torch.ones_like(ei)
        for j in range(4):
            if j == i:
                continue
            denom = denom * (e[..., j] - ei)

        dx = torch.clamp(x - ei.unsqueeze(-1), min=0.0)
        cdf = cdf + dx.pow(3) / denom.unsqueeze(-1)
        pdf = pdf + 3.0 * dx.pow(2) / denom.unsqueeze(-1)

    # The truncated-power expression is an exact partition of unity, but for
    # x >= e_max it obtains CDF=1 and PDF=0 by cancellation of four potentially
    # enormous terms.  That cancellation is numerically disastrous for flat or
    # nearly-flat bands (a common case in large supercells).  In particular,
    # clamping the residual PDF below turned round-off of either sign into a
    # large positive, non-normalized DOS background.  Set the analytically
    # known tails explicitly before clamping.
    e_min = e[..., :1]
    e_max = e[..., -1:]
    below = x <= e_min
    above = x >= e_max
    cdf = torch.where(below, torch.zeros_like(cdf), cdf)
    pdf = torch.where(below, torch.zeros_like(pdf), pdf)
    cdf = torch.where(above, torch.ones_like(cdf), cdf)
    pdf = torch.where(above, torch.zeros_like(pdf), pdf)
    cdf = torch.clamp(cdf, min=0.0, max=1.0)
    pdf = torch.clamp(pdf, min=0.0)
    return cdf, pdf


def _tetra_worker_init() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _tetra_batch_worker(task: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    if _TETRA_MP_STATE is None:
        raise RuntimeError("Tetrahedron worker state is not initialized.")
    start, stop = task
    batch = _TETRA_MP_STATE["tet_energies"][start:stop]
    cdf_batch, pdf_batch = _tetrahedron_cdf_pdf(batch, _TETRA_MP_STATE["grid_ev"])
    weight = float(_TETRA_MP_STATE["weight"])
    return (
        weight * torch.sum(cdf_batch, dim=0).cpu(),
        weight * torch.sum(pdf_batch, dim=0).cpu(),
    )


def compute_tetrahedron_dos_from_kmesh_eigenvalues(
    eigenvalues_ev: torch.Tensor,
    kmesh: tuple[int, int, int],
    *,
    grid_ev: torch.Tensor | None = None,
    bin_width: float = 0.05,
    e_min: float | None = None,
    e_max: float | None = None,
    spin_factor: float = 1.0,
    tetra_batch_size: int = 256,
    num_workers: int = 1,
    show_progress: bool = False,
    progress_label: str = "Tetrahedron batches",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _TETRA_MP_STATE

    nx, ny, nz = kmesh
    n_cells = nx * ny * nz

    if eigenvalues_ev.ndim == 2:
        nk, nbands = eigenvalues_ev.shape
        expected = nx * ny * nz
        if nk != expected:
            raise ValueError(
                f"kmesh {kmesh} has {expected} points, but eigenvalues have {nk}"
            )
        eig = eigenvalues_ev.reshape(nx, ny, nz, nbands)
    elif eigenvalues_ev.ndim == 4:
        if tuple(eigenvalues_ev.shape[:3]) != tuple(kmesh):
            raise ValueError(
                f"eigenvalue grid shape {tuple(eigenvalues_ev.shape[:3])} does not match kmesh {kmesh}"
            )
        eig = eigenvalues_ev
        nbands = eig.shape[-1]
    else:
        raise ValueError(
            "Expected eigenvalues_ev with shape (Nk, Nb) or (Nx, Ny, Nz, Nb), "
            f"got {tuple(eigenvalues_ev.shape)}"
        )

    eig = eig.detach().to(dtype=torch.float64, device="cpu")
    eig_min = float(torch.min(eig).item()) if e_min is None else float(e_min)
    eig_max = float(torch.max(eig).item()) if e_max is None else float(e_max)

    if grid_ev is None:
        span = max(eig_max - eig_min, 1e-8)
        margin = 0.02 * span + 0.05
        e0 = eig_min - margin
        e1 = eig_max + margin
        n_grid = int(np.ceil((e1 - e0) / bin_width)) + 1
        grid_ev = torch.linspace(e0, e1, n_grid, dtype=torch.float64)
    else:
        grid_ev = grid_ev.detach().to(dtype=torch.float64, device="cpu")

    tetrahedra = torch.tensor(
        [
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
            [0, 4, 5, 7],
            [0, 5, 1, 7],
        ],
        dtype=torch.long,
    )

    all_tet_energies: list[torch.Tensor] = []
    for ix in range(nx):
        ix1 = (ix + 1) % nx
        for iy in range(ny):
            iy1 = (iy + 1) % ny
            for iz in range(nz):
                iz1 = (iz + 1) % nz
                cube = torch.stack(
                    [
                        eig[ix, iy, iz],
                        eig[ix1, iy, iz],
                        eig[ix, iy1, iz],
                        eig[ix1, iy1, iz],
                        eig[ix, iy, iz1],
                        eig[ix1, iy, iz1],
                        eig[ix, iy1, iz1],
                        eig[ix1, iy1, iz1],
                    ],
                    dim=0,
                )
                tet_e = cube[tetrahedra].permute(0, 2, 1).reshape(-1, 4)
                all_tet_energies.append(tet_e)

    tet_energies = torch.cat(all_tet_energies, dim=0)
    dos = torch.zeros_like(grid_ev)
    cumulative = torch.zeros_like(grid_ev)
    weight = float(spin_factor) / float(6 * n_cells)

    tasks = [
        (start, min(start + tetra_batch_size, tet_energies.shape[0]))
        for start in range(0, tet_energies.shape[0], tetra_batch_size)
    ]
    worker_count = max(1, min(int(num_workers), len(tasks)))

    if worker_count <= 1 or len(tasks) <= 1:
        iterator: Any = range(0, tet_energies.shape[0], tetra_batch_size)
        if show_progress and tet_energies.shape[0] > tetra_batch_size:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(tasks), desc="Tetrahedron batches")
        for start in iterator:
            stop = min(start + tetra_batch_size, tet_energies.shape[0])
            batch = tet_energies[start:stop]
            cdf_batch, pdf_batch = _tetrahedron_cdf_pdf(batch, grid_ev)
            cumulative = cumulative + weight * torch.sum(cdf_batch, dim=0)
            dos = dos + weight * torch.sum(pdf_batch, dim=0)
        return grid_ev, dos, cumulative

    try:
        ctx = mp.get_context("fork")
    except ValueError as exc:  # pragma: no cover - platform specific
        raise RuntimeError(
            "multiprocessing fork context is required for parallel tetrahedron batches"
        ) from exc

    _TETRA_MP_STATE = {
        "tet_energies": tet_energies.detach().cpu(),
        "grid_ev": grid_ev.detach().cpu(),
        "weight": weight,
    }
    try:
        from tqdm.auto import tqdm

        with ctx.Pool(worker_count, initializer=_tetra_worker_init) as pool:
            result_iter = pool.imap(_tetra_batch_worker, tasks, chunksize=1)
            if show_progress and len(tasks) > 1:
                result_iter = tqdm(result_iter, total=len(tasks), desc=progress_label)
            for cdf_part, pdf_part in result_iter:
                cumulative = cumulative + cdf_part
                dos = dos + pdf_part
        return grid_ev, dos, cumulative
    finally:
        _TETRA_MP_STATE = None


def _dos_cache_signature(
    snapshot: Snapshot,
    *,
    kind: str,
    kmesh_spec: str,
    chunk_size: int,
    num_workers: int,
    psd_cleanup: bool,
    allow_jitter: bool,
    bin_width: float,
    tetra_batch_size: int,
    e_min: float | None,
    e_max: float | None,
) -> dict[str, Any]:
    def _path_sig(path_like: str | Path | None) -> dict[str, Any] | None:
        if path_like is None:
            return None
        path = Path(path_like)
        if not path.exists():
            return {"path": str(path), "exists": False}
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "exists": True,
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }

    info = getattr(snapshot, "info", None)
    fermi_level = None
    if getattr(info, "fermi_level", None) is not None:
        fermi_level = float(info.fermi_level.item() * HARTREE_TO_EV)
    return {
        "kind": kind,
        # Bump when the numerical DOS algorithm changes so stale bundles do
        # not silently survive --force-refresh at the outer evaluation layer.
        "algorithm_version": 3,
        "matrix_path": _path_sig(getattr(snapshot, "matrix_path", None)),
        "info_path": _path_sig(getattr(snapshot, "info_path", None)),
        "kmesh_spec": kmesh_spec,
        "chunk_size": int(chunk_size),
        "num_workers": int(num_workers),
        "psd_cleanup": bool(psd_cleanup),
        "allow_jitter": bool(allow_jitter),
        "bin_width": float(bin_width),
        "tetra_batch_size": int(tetra_batch_size),
        "e_min": None if e_min is None else float(e_min),
        "e_max": None if e_max is None else float(e_max),
        "num_atoms": len(getattr(snapshot.hamiltonian, "atoms", [])),
        "num_electrons": float(
            snapshot.get_number_of_electrons().detach().cpu().item()
        ),
        "fermi_level_ev": fermi_level,
    }


def _load_dos_cache(
    cache_path: Path,
    signature: dict[str, Any],
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        float,
        float | None,
        float,
        torch.Tensor | None,
        torch.Tensor | None,
    ]
    | None
):
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    if payload.get("signature") != signature:
        return None
    return (
        payload["grid_ev"],
        payload["dos"],
        float(payload["num_electrons"]),
        payload.get("dos_electron_target"),
        float(payload["fermi_level_ev"]),
        payload.get("eigenvalues_ev"),
        payload.get("fractional_kpoints"),
    )


def _save_dos_cache(
    cache_path: Path,
    signature: dict[str, Any],
    grid_ev: torch.Tensor,
    dos: torch.Tensor,
    num_electrons: float,
    dos_electron_target: float | None,
    fermi_level_ev: float,
    eigenvalues_ev: torch.Tensor,
    fractional_kpoints: torch.Tensor,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "grid_ev": grid_ev.detach().cpu(),
            "dos": dos.detach().cpu(),
            "num_electrons": float(num_electrons),
            "dos_electron_target": (
                None if dos_electron_target is None else float(dos_electron_target)
            ),
            "fermi_level_ev": float(fermi_level_ev),
            # Keep the already-computed spectrum available to report builders.
            # This does not add another eigensolve or alter the DOS result.
            "eigenvalues_ev": eigenvalues_ev.detach().cpu(),
            "fractional_kpoints": fractional_kpoints.detach().cpu(),
        },
        cache_path,
    )


def _kmesh_eigenvalues(
    snapshot: Snapshot,
    *,
    kmesh_spec: str,
    chunk_size: int,
    num_workers: int,
    psd_cleanup: bool,
    allow_jitter: bool,
    progress_label: str = "DOS k-mesh eigensolve",
) -> tuple[torch.Tensor, torch.Tensor]:
    kmesh = parse_kmesh_spec(kmesh_spec)
    fractional_kpoints = fractional_kmesh_points(
        kmesh,
        device=snapshot.box.device,
        dtype=snapshot.box.dtype,
    )
    reciprocal = 2.0 * torch.pi * torch.linalg.inv(snapshot.box).T
    kpoints_abs = fractional_kpoints @ reciprocal
    shift_t = snapshot.get_translation_shifts().to(device=snapshot.box.device)
    if shift_t.numel() == 0:
        raise ValueError("Snapshot does not contain any translation shifts.")

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
        for start, stop in tasks:
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
                    psd_cleanup=psd_cleanup,
                    allow_jitter=allow_jitter,
                ).cpu()
            )
    else:
        eig_chunks = compute_band_chunks_parallel(
            ham_shift=ham_shift.detach().cpu(),
            ovl_shift=ovl_shift.detach().cpu(),
            shift_t=shift_t.detach().cpu(),
            box=snapshot.box.detach().cpu(),
            kpoints_abs=kpoints_abs.detach().cpu(),
            tasks=tasks,
            num_workers=num_workers,
            overlap_psd_cleanup=psd_cleanup,
            overlap_jitter=allow_jitter,
            progress_label=progress_label,
        )

    eigenvalues_ev = (
        torch.cat(eig_chunks, dim=0).to(dtype=torch.float64) * HARTREE_TO_EV
    )
    if eigenvalues_ev.ndim != 2:
        raise ValueError(
            f"Expected band eigenvalues with shape (Nk, Nb), got {tuple(eigenvalues_ev.shape)}"
        )
    expected = int(np.prod(kmesh))
    if eigenvalues_ev.shape[0] != expected:
        raise ValueError(
            f"kmesh {kmesh} has {expected} points, but eigenvalues have {eigenvalues_ev.shape[0]}"
        )
    return eigenvalues_ev, fractional_kpoints.detach().cpu()


def compute_tetrahedron_dos_and_fermi(
    snapshot: Snapshot,
    *,
    kmesh_spec: str,
    chunk_size: int,
    num_workers: int,
    psd_cleanup: bool,
    allow_jitter: bool,
    bin_width: float,
    tetra_batch_size: int = 256,
    grid_ev: torch.Tensor | None = None,
    e_min: float | None = None,
    e_max: float | None = None,
    show_progress: bool | None = None,
    cache_path: Path | None = None,
    progress_label: str = "Tetrahedron DOS k-mesh eigensolve",
) -> tuple[torch.Tensor, torch.Tensor, float, float | None, float]:
    signature = _dos_cache_signature(
        snapshot,
        kind="tetrahedron",
        kmesh_spec=kmesh_spec,
        chunk_size=chunk_size,
        num_workers=num_workers,
        psd_cleanup=psd_cleanup,
        allow_jitter=allow_jitter,
        bin_width=bin_width,
        tetra_batch_size=tetra_batch_size,
        e_min=e_min,
        e_max=e_max,
    )
    if cache_path is not None:
        cached = _load_dos_cache(cache_path, signature)
        if cached is not None:
            grid, dos, num_electrons, dos_target, fermi, _eigs, _kpoints = cached
            return grid, dos, num_electrons, dos_target, fermi
    if show_progress is None:
        show_progress = sys.stderr.isatty()
    eigenvalues_ev, _fractional_kpoints = _kmesh_eigenvalues(
        snapshot,
        kmesh_spec=kmesh_spec,
        chunk_size=chunk_size,
        num_workers=num_workers,
        psd_cleanup=psd_cleanup,
        allow_jitter=allow_jitter,
        progress_label=progress_label,
    )
    kmesh = parse_kmesh_spec(kmesh_spec)
    grid_ev, dos, _cumulative = compute_tetrahedron_dos_from_kmesh_eigenvalues(
        eigenvalues_ev,
        kmesh,
        grid_ev=grid_ev,
        bin_width=bin_width,
        e_min=e_min,
        e_max=e_max,
        spin_factor=infer_spin_factor(snapshot),
        tetra_batch_size=tetra_batch_size,
        num_workers=num_workers,
        show_progress=show_progress,
        progress_label=f"{progress_label} / tetrahedron batches",
    )
    num_electrons = float(snapshot.get_number_of_electrons().detach().cpu().item())
    dos_electron_target = effective_dos_electron_target(snapshot)
    fermi_level_ev = None
    if getattr(snapshot.info, "fermi_level", None) is not None:
        fermi_level_ev = float(snapshot.info.fermi_level.item() * HARTREE_TO_EV)
    if fermi_level_ev is None:
        fermi_level_ev = fermi_level_from_dos(grid_ev, dos, dos_electron_target)
    if cache_path is not None:
        _save_dos_cache(
            cache_path,
            signature,
            grid_ev,
            dos,
            num_electrons,
            dos_electron_target,
            fermi_level_ev,
            eigenvalues_ev,
            _fractional_kpoints,
        )
    return grid_ev, dos, num_electrons, dos_electron_target, fermi_level_ev


def compute_gaussian_dos_from_eigenvalues_and_fermi(
    snapshot: Snapshot,
    *,
    psd_cleanup: bool,
    allow_jitter: bool,
    dos_sigma_ev: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float | None, float]:
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
    dos_electron_target = effective_dos_electron_target(snapshot)
    fermi_level_ev = float(snapshot.info.fermi_level.item() * HARTREE_TO_EV)
    return grid_ev, dos, num_electrons, dos_electron_target, fermi_level_ev


def compute_kmesh_average_dos_and_fermi(
    snapshot: Snapshot,
    *,
    kmesh_spec: str,
    chunk_size: int,
    num_workers: int,
    psd_cleanup: bool,
    allow_jitter: bool,
    dos_sigma_ev: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float | None, float]:
    eigenvalues_ev, _fractional_kpoints = _kmesh_eigenvalues(
        snapshot,
        kmesh_spec=kmesh_spec,
        chunk_size=chunk_size,
        num_workers=num_workers,
        psd_cleanup=psd_cleanup,
        allow_jitter=allow_jitter,
    )
    flat_ev = eigenvalues_ev.reshape(-1)
    eig_min = float(torch.min(flat_ev).item())
    eig_max = float(torch.max(flat_ev).item())
    span = max(eig_max - eig_min, 1e-6)
    margin = 0.1 * span + 0.05
    e_min = eig_min - margin
    e_max = eig_max + margin
    grid_ev, dos = compute_dos_from_eigenvalues(
        flat_ev,
        sigma=dos_sigma_ev,
        bin_width=0.1,
        e_min=e_min,
        e_max=e_max,
    )
    dos = dos / float(eigenvalues_ev.shape[0])
    num_electrons = float(snapshot.get_number_of_electrons().detach().cpu().item())
    dos_electron_target = effective_dos_electron_target(snapshot)
    fermi_level_ev = float(snapshot.info.fermi_level.item() * HARTREE_TO_EV)
    return grid_ev, dos, num_electrons, dos_electron_target, fermi_level_ev


def save_band_structure_plot(
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
    tick_labels = [display_k_label(label) for label in payload.tick_labels]

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


def save_dos_plot(
    grid_ev: torch.Tensor,
    dos: torch.Tensor,
    *,
    output_path: Path,
    title: str,
    num_electrons: float | None,
    fermi_level_ev: float | None,
    energy_min: float | None = None,
    energy_max: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_ev = grid_ev.detach().cpu()
    dos = dos.detach().cpu()
    if energy_min is not None and energy_max is not None:
        grid_ev, dos = _clip_energy_curve(
            grid_ev,
            dos,
            energy_min=energy_min,
            energy_max=energy_max,
        )

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.8))
    ax.plot(grid_ev.numpy(), dos.numpy(), color="#1f5aa6", lw=1.8)
    ax.set_title(title)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS")
    ax.grid(True, alpha=0.25)

    if fermi_level_ev is not None:
        ax.axvline(fermi_level_ev, color="black", ls=":", lw=1.5, label="Fermi level")
    if num_electrons is not None:
        ax.text(
            0.02,
            0.95,
            f"N_e = {num_electrons:.3f}",
            transform=ax.transAxes,
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


def save_band_and_dos_plot(
    payload: Any,
    *,
    dos_grid: torch.Tensor,
    dos: torch.Tensor,
    dos_reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None,
    output_path: Path,
    title: str,
    emin_ev: float,
    emax_ev: float,
    line_alpha: float,
    num_electrons: float | None,
    fermi_level_ev: float | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    energies_ev = payload.eigenvalues.detach().cpu() * HARTREE_TO_EV
    band_fermi_ev = fermi_level_ev
    if payload.fermi_level is not None:
        band_fermi_ev = float(payload.fermi_level.detach().cpu().item() * HARTREE_TO_EV)
        energies_ev = energies_ev - band_fermi_ev

    linear_k = payload.linear_k.detach().cpu()
    tick_positions = payload.tick_positions.detach().cpu()
    tick_labels = [display_k_label(label) for label in payload.tick_labels]
    dos_grid_shifted = dos_grid - (0.0 if band_fermi_ev is None else band_fermi_ev)
    dos_grid_shifted, dos = _clip_energy_curve(
        dos_grid_shifted.detach().cpu(),
        dos.detach().cpu(),
        energy_min=emin_ev,
        energy_max=emax_ev,
    )
    if dos_reference is not None:
        ref_energy, ref_dos, _ref_cumulative = dos_reference
        ref_energy = ref_energy.detach().cpu()
        ref_dos = ref_dos.detach().cpu()
        ref_energy, ref_dos = _clip_energy_curve(
            ref_energy,
            ref_dos,
            energy_min=emin_ev,
            energy_max=emax_ev,
        )
        dos_reference = (ref_energy, ref_dos, _ref_cumulative)

    fig, (ax_band, ax_dos) = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.0),
        gridspec_kw={"width_ratios": [2.3, 1.0]},
        sharey=True,
    )

    for band_idx in range(energies_ev.shape[1]):
        ax_band.plot(
            linear_k.numpy(),
            energies_ev[:, band_idx].numpy(),
            color="#1f5aa6",
            lw=1.05,
            alpha=line_alpha,
        )
    for xpos in tick_positions.tolist():
        ax_band.axvline(xpos, color="0.80", lw=0.8, zorder=0)
    ax_band.axhline(0.0, color="black", ls="--", lw=1.0, alpha=0.7)
    ax_band.set_xlim(float(linear_k[0].item()), float(linear_k[-1].item()))
    ax_band.set_ylim(emin_ev, emax_ev)
    ax_band.set_xticks(tick_positions.numpy())
    ax_band.set_xticklabels(tick_labels, fontsize=11)
    ax_band.set_ylabel(r"$E - E_F$ (eV)")
    ax_band.set_title(title)
    ax_band.grid(True, axis="y", alpha=0.2)
    if band_fermi_ev is not None:
        ax_band.text(
            0.98,
            0.03,
            f"Fermi level = {band_fermi_ev:.3f} eV",
            transform=ax_band.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    ax_dos.plot(
        dos.numpy(),
        dos_grid_shifted.numpy(),
        color="#1f5aa6",
        lw=1.8,
        label="Our DOS",
    )
    if dos_reference is not None:
        ref_energy, ref_dos, _ref_cumulative = dos_reference
        ax_dos.plot(
            ref_dos.numpy(),
            ref_energy.numpy(),
            color="tab:orange",
            lw=1.2,
            ls="--",
            label="OpenMX DOS",
        )
    if band_fermi_ev is not None:
        ax_dos.axhline(0.0, color="black", ls=":", lw=1.5, label="Fermi level")
    ax_dos.axvline(0.0, color="black", ls="--", lw=1.1, alpha=0.85)
    ax_dos.set_xlabel("DOS")
    ax_dos.set_title("DOS")
    ax_dos.grid(True, alpha=0.25)
    if num_electrons is not None:
        ax_dos.text(
            0.98,
            0.03,
            f"N_e = {num_electrons:.3f}",
            transform=ax_dos.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
    ax_dos.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plot(
    pred,
    target,
    output_path: Path,
    *,
    title: str,
    max_atoms: int,
    clim: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from net.artifacts import _as_dense, _crop_dense_to_max_atoms

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


def align_prediction_to_target(
    pred,
    target,
    *,
    prefer_prefix: bool = True,
) -> tuple[Any, dict[str, Any]]:
    pair_blocks: dict[str, torch.Tensor] = {}
    pair_edges: dict[str, torch.Tensor] = {}
    lookup: dict[tuple[int, int, int, int, int], tuple[str, int]] = {}
    debug: dict[str, Any] = {
        "mode": "prefix" if prefer_prefix else "lookup",
        "used_prefix": True,
        "fallback_used": False,
        "keys": {},
        "extra_edges_total": 0,
        "missing_edges_total": 0,
        "prefix_mismatch_keys": [],
    }

    for key, target_blocks in target.pair_blocks.items():
        if key not in pred.pair_blocks:
            raise ValueError(f"Key {key!r} not found in prediction.")
        pred_blocks = pred.pair_blocks[key]
        pred_edges = pred.pair_edges[key]
        target_edges = target.pair_edges[key]
        target_n = int(target_blocks.shape[0])
        pred_n = int(pred_blocks.shape[0])
        if pred_n < target_n:
            raise ValueError(
                f"Predicted blocks for key {key!r} are too short: pred_len={pred_n} target_len={target_n}"
            )

        key_debug = {
            "target_edges": target_n,
            "pred_edges": pred_n,
            "extra_edges": max(pred_n - target_n, 0),
            "used_prefix": False,
            "missing_edges": 0,
        }
        debug["extra_edges_total"] += key_debug["extra_edges"]

        use_prefix = False
        if prefer_prefix:
            prefix_edges = pred_edges[:, :target_n]
            if prefix_edges.shape == target_edges.shape and torch.equal(
                prefix_edges.detach().cpu(), target_edges.detach().cpu()
            ):
                use_prefix = True

        if use_prefix:
            kept_blocks = pred_blocks[:target_n]
            kept_edges = pred_edges[:, :target_n]
            key_debug["used_prefix"] = True
        else:
            debug["used_prefix"] = False
            debug["fallback_used"] = True
            debug["prefix_mismatch_keys"].append(str(key))
            selected_idx: list[int] = []
            missing_edges = 0
            for edge in target_edges.t().tolist():
                pred_entry = pred.lookup.get(tuple(int(v) for v in edge))
                if pred_entry is None:
                    missing_edges += 1
                    continue
                _pred_key, pred_idx = pred_entry
                selected_idx.append(int(pred_idx))
            key_debug["missing_edges"] = missing_edges
            debug["missing_edges_total"] += missing_edges
            if selected_idx:
                index_tensor = torch.tensor(
                    selected_idx, device=pred_blocks.device, dtype=torch.long
                )
                kept_blocks = pred_blocks.index_select(0, index_tensor)
                kept_edges = pred_edges.index_select(1, index_tensor)
            else:
                kept_blocks = pred_blocks[:0]
                kept_edges = pred_edges[:, :0]

        pair_blocks[key] = kept_blocks
        pair_edges[key] = kept_edges
        key_debug["kept_edges"] = int(kept_blocks.shape[0])
        debug["keys"][str(key)] = key_debug
        for idx, (sx, sy, sz, i, j) in enumerate(kept_edges.t().tolist()):
            lookup[(int(sx), int(sy), int(sz), int(i), int(j))] = (key, idx)

    aligned_pred = pred.__class__(
        atoms=pred.atoms,
        atom_counts=pred.atom_counts,
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=pred.orbital_cfg,
        basis=pred.basis,
    )
    return aligned_pred, debug


def _dense_zero_support_stats(
    pred_dense: torch.Tensor,
    target_dense: torch.Tensor,
    *,
    zero_tol: float,
) -> dict[str, float | int]:
    target_abs = torch.abs(target_dense)
    pred_abs = torch.abs(pred_dense)
    zero_mask = target_abs <= float(zero_tol)
    nonzero_pred_mask = pred_abs > float(zero_tol)
    spike_mask = zero_mask & nonzero_pred_mask
    zero_count = int(zero_mask.sum().item())
    spike_count = int(spike_mask.sum().item())
    spike_values = pred_abs[spike_mask]
    return {
        "target_zero_entries": zero_count,
        "target_zero_pred_nonzero_entries": spike_count,
        "target_zero_pred_nonzero_fraction": (
            float(spike_count / zero_count) if zero_count > 0 else 0.0
        ),
        "target_zero_pred_nonzero_mean_abs": (
            float(spike_values.mean().item()) if spike_values.numel() else 0.0
        ),
        "target_zero_pred_nonzero_max_abs": (
            float(spike_values.max().item()) if spike_values.numel() else 0.0
        ),
    }


def compute_prediction_support_diagnostics(
    pred,
    target,
    *,
    positions: torch.Tensor,
    box: torch.Tensor | None,
    zero_tol: float = 1.0e-12,
    top_k: int = 25,
) -> dict[str, Any]:
    pred_aligned, align_debug = align_prediction_to_target(pred, target)
    positions_cpu = positions.detach().cpu()
    box_cpu = box.detach().cpu() if box is not None else None

    pred_only_records: list[dict[str, Any]] = []
    missing_records: list[dict[str, Any]] = []

    for key, pred_edges in pred.pair_edges.items():
        pred_blocks = pred.pair_blocks[key].detach().cpu()
        for idx in range(pred_edges.shape[1]):
            sx, sy, sz, i, j = [int(v) for v in pred_edges[:, idx].tolist()]
            lookup_key = (sx, sy, sz, i, j)
            if lookup_key in target.lookup:
                continue
            pred_block = pred_blocks[idx]
            disp = _edge_displacement(
                positions_cpu,
                box_cpu,
                i,
                j,
                (sx, sy, sz),
            )
            pred_only_records.append(
                {
                    "pair_key": str(key),
                    "src_atom": i,
                    "dst_atom": j,
                    "shift": [sx, sy, sz],
                    "edge_index": int(idx),
                    "edge_length": float(torch.linalg.norm(disp).item()),
                    "mean_abs_pred": _block_mean_abs(pred_block),
                    "max_abs_pred": float(torch.max(torch.abs(pred_block)).item()),
                }
            )

    for key, target_edges in target.pair_edges.items():
        for idx in range(target_edges.shape[1]):
            sx, sy, sz, i, j = [int(v) for v in target_edges[:, idx].tolist()]
            lookup_key = (sx, sy, sz, i, j)
            if lookup_key in pred.lookup:
                continue
            disp = _edge_displacement(
                positions_cpu,
                box_cpu,
                i,
                j,
                (sx, sy, sz),
            )
            missing_records.append(
                {
                    "pair_key": str(key),
                    "src_atom": i,
                    "dst_atom": j,
                    "shift": [sx, sy, sz],
                    "edge_index": int(idx),
                    "edge_length": float(torch.linalg.norm(disp).item()),
                }
            )

    pred_only_records.sort(
        key=lambda rec: (-float(rec["mean_abs_pred"]), rec["src_atom"], rec["dst_atom"])
    )
    missing_records.sort(
        key=lambda rec: (rec["src_atom"], rec["dst_atom"], tuple(rec["shift"]))
    )

    pred_dense_raw = pred.to_dense().detach().cpu().to(torch.float64)
    pred_dense_aligned = pred_aligned.to_dense().detach().cpu().to(torch.float64)
    target_dense = target.to_dense().detach().cpu().to(torch.float64)
    corr_raw = float(
        torch.corrcoef(torch.stack([target_dense.flatten(), pred_dense_raw.flatten()]))[
            0, 1
        ].item()
    )
    corr_aligned = float(
        torch.corrcoef(
            torch.stack([target_dense.flatten(), pred_dense_aligned.flatten()])
        )[0, 1].item()
    )

    pred_only_lengths = [float(rec["edge_length"]) for rec in pred_only_records]
    pred_only_magnitudes = [float(rec["mean_abs_pred"]) for rec in pred_only_records]

    return {
        "alignment": align_debug,
        "counts": {
            "target_edges_total": int(
                sum(edges.shape[1] for edges in target.pair_edges.values())
            ),
            "pred_edges_total_raw": int(
                sum(edges.shape[1] for edges in pred.pair_edges.values())
            ),
            "pred_edges_total_aligned": int(
                sum(edges.shape[1] for edges in pred_aligned.pair_edges.values())
            ),
            "pred_only_edges_total": int(len(pred_only_records)),
            "missing_edges_total": int(len(missing_records)),
        },
        "correlation": {
            "raw_dense": corr_raw,
            "aligned_dense": corr_aligned,
        },
        "dense_zero_support_raw": _dense_zero_support_stats(
            pred_dense_raw,
            target_dense,
            zero_tol=zero_tol,
        ),
        "dense_zero_support_aligned": _dense_zero_support_stats(
            pred_dense_aligned,
            target_dense,
            zero_tol=zero_tol,
        ),
        "pred_only_edge_length": {
            "count": int(len(pred_only_lengths)),
            "mean": float(np.mean(pred_only_lengths)) if pred_only_lengths else 0.0,
            "median": float(np.median(pred_only_lengths)) if pred_only_lengths else 0.0,
            "max": float(np.max(pred_only_lengths)) if pred_only_lengths else 0.0,
        },
        "pred_only_mean_abs_pred": {
            "count": int(len(pred_only_magnitudes)),
            "mean": (
                float(np.mean(pred_only_magnitudes)) if pred_only_magnitudes else 0.0
            ),
            "median": (
                float(np.median(pred_only_magnitudes)) if pred_only_magnitudes else 0.0
            ),
            "max": float(np.max(pred_only_magnitudes)) if pred_only_magnitudes else 0.0,
        },
        "top_pred_only_edges": pred_only_records[:top_k],
        "top_missing_edges": missing_records[:top_k],
    }


def save_prediction_support_debug_artifacts(
    pred,
    target,
    *,
    positions: torch.Tensor,
    box: torch.Tensor | None,
    output_dir: Path,
    prefix: str,
    title: str,
    zero_tol: float = 1.0e-12,
) -> dict[str, Any]:
    diagnostics = compute_prediction_support_diagnostics(
        pred,
        target,
        positions=positions,
        box=box,
        zero_tol=zero_tol,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{prefix}_prediction_support_debug.json"
    json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    pred_aligned, _ = align_prediction_to_target(pred, target)
    save_correlation_plot(
        pred_aligned,
        target,
        output_dir / f"{prefix}_aligned_correlation.png",
        title=f"{title} (training-style aligned)",
        max_points=250000,
        alpha=0.03,
        seed=0,
    )

    pred_only_records = diagnostics["top_pred_only_edges"]
    all_pred_only = compute_prediction_support_diagnostics(
        pred,
        target,
        positions=positions,
        box=box,
        zero_tol=zero_tol,
        top_k=max(
            diagnostics["counts"]["pred_only_edges_total"],
            len(pred_only_records),
        ),
    )["top_pred_only_edges"]
    pred_only_lengths = [float(rec["edge_length"]) for rec in all_pred_only]
    pred_only_magnitudes = [float(rec["mean_abs_pred"]) for rec in all_pred_only]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    if pred_only_lengths:
        axes[0].hist(
            pred_only_lengths, bins=min(40, max(10, len(pred_only_lengths) // 4))
        )
    axes[0].set_title("Prediction-only edge lengths")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.25)
    if pred_only_magnitudes:
        log_mag = np.log10(np.clip(np.asarray(pred_only_magnitudes), 1e-16, None))
        axes[1].hist(log_mag, bins=min(40, max(10, len(log_mag) // 4)))
    axes[1].set_title("Prediction-only block mean |value|")
    axes[1].set_xlabel("log10(mean |pred block|)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f"{title}: support diagnostics")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{prefix}_support_debug.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    return diagnostics


def compute_block_error_scatter_data(
    pred,
    target,
    *,
    positions: torch.Tensor,
    box: torch.Tensor,
) -> dict[str, Any]:
    positions_cpu = positions.detach().cpu()
    box_cpu = box.detach().cpu()

    edge_length: list[float] = []
    abs_mae: list[float] = []
    rel_mae: list[float] = []
    block_magnitude: list[float] = []
    pair_key: list[str] = []
    edge_index: list[int] = []
    src_atom: list[int] = []
    dst_atom: list[int] = []
    shift_sx: list[int] = []
    shift_sy: list[int] = []
    shift_sz: list[int] = []
    matched_edges = 0
    missing_in_pred = 0

    eps = 1.0e-12
    for key in target.keys():
        if key not in pred.pair_edges:
            continue
        target_edges = target.pair_edges[key].detach().cpu()
        target_blocks = target.pair_blocks[key].detach().cpu()
        pred_lookup = pred.lookup
        for idx in range(target_edges.shape[1]):
            sx, sy, sz, i, j = [int(v) for v in target_edges[:, idx].tolist()]
            pred_entry = pred_lookup.get((sx, sy, sz, i, j))
            if pred_entry is None:
                missing_in_pred += 1
                continue
            pred_key, pred_idx = pred_entry
            pred_block = pred.pair_blocks[pred_key][pred_idx].detach().cpu()
            target_block = target_blocks[idx]
            shift = torch.tensor([sx, sy, sz], dtype=positions_cpu.dtype)
            disp = positions_cpu[j] - positions_cpu[i] + shift @ box_cpu
            length = float(torch.linalg.norm(disp).item())
            diff = pred_block - target_block
            mae = float(torch.mean(torch.abs(diff)).item())
            magnitude = float(torch.mean(torch.abs(target_block)).item())
            rel = mae / max(magnitude, eps)
            edge_length.append(length)
            abs_mae.append(mae)
            rel_mae.append(rel)
            block_magnitude.append(magnitude)
            pair_key.append(str(key))
            edge_index.append(idx)
            src_atom.append(i)
            dst_atom.append(j)
            shift_sx.append(sx)
            shift_sy.append(sy)
            shift_sz.append(sz)
            matched_edges += 1

    return {
        "pair_key": pair_key,
        "edge_index": edge_index,
        "src_atom": src_atom,
        "dst_atom": dst_atom,
        "shift_sx": shift_sx,
        "shift_sy": shift_sy,
        "shift_sz": shift_sz,
        "edge_length": edge_length,
        "abs_mae": abs_mae,
        "rel_mae": rel_mae,
        "block_magnitude": block_magnitude,
        "matched_edges": matched_edges,
        "missing_in_pred": missing_in_pred,
    }


def save_block_error_scatter_data(
    pred,
    target,
    *,
    positions: torch.Tensor,
    box: torch.Tensor,
    output_path: Path,
) -> None:
    payload = compute_block_error_scatter_data(
        pred,
        target,
        positions=positions,
        box=box,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def save_prediction_plot(
    mat,
    output_path: Path,
    *,
    title: str,
    max_atoms: int,
    clim: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from net.artifacts import _as_dense, _crop_dense_to_max_atoms

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


def _block_mean_abs(block: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(block)).item())


def _tuple3(values: Any) -> tuple[int, int, int]:
    return (int(values[0]), int(values[1]), int(values[2]))


def _node_image_label(atom_idx: int, shift: tuple[int, int, int]) -> str:
    if shift == (0, 0, 0):
        return str(atom_idx)
    return f"{atom_idx} [{shift[0]},{shift[1]},{shift[2]}]"


def _orbital_dims_for_atoms(atoms: tuple[str, ...], orbital_cfg: Any) -> list[int]:
    return [int(orbital_cfg.element_to_irreps[atom].dim) for atom in atoms]


def _dense_atom_subset(
    dense: torch.Tensor,
    atoms: tuple[str, ...],
    orbital_cfg: Any,
    atom_indices: list[int],
) -> tuple[torch.Tensor, list[int]]:
    dims_all = _orbital_dims_for_atoms(atoms, orbital_cfg)
    offsets = [0]
    for dim in dims_all[:-1]:
        offsets.append(offsets[-1] + dim)
    chunks: list[torch.Tensor] = []
    dims_selected: list[int] = []
    for row_atom in atom_indices:
        row_dim = dims_all[row_atom]
        row_start = offsets[row_atom]
        row_chunks = []
        for col_atom in atom_indices:
            col_dim = dims_all[col_atom]
            col_start = offsets[col_atom]
            row_chunks.append(
                dense[
                    row_start : row_start + row_dim,
                    col_start : col_start + col_dim,
                ]
            )
        chunks.append(torch.cat(row_chunks, dim=1))
        dims_selected.append(row_dim)
    return torch.cat(chunks, dim=0), dims_selected


def _pair_dense_block(
    dense: torch.Tensor,
    atoms: tuple[str, ...],
    orbital_cfg: Any,
    src_atom: int,
    dst_atom: int,
) -> torch.Tensor:
    dims_all = _orbital_dims_for_atoms(atoms, orbital_cfg)
    offsets = [0]
    for dim in dims_all[:-1]:
        offsets.append(offsets[-1] + dim)
    row_dim = dims_all[src_atom]
    col_dim = dims_all[dst_atom]
    row_start = offsets[src_atom]
    col_start = offsets[dst_atom]
    return dense[
        row_start : row_start + row_dim,
        col_start : col_start + col_dim,
    ]


def _cutout_axes_meta(labels: list[str], dims: list[int]) -> dict[str, Any]:
    boundaries = []
    centers = []
    offset = 0
    for dim in dims:
        boundaries.append(offset)
        centers.append(offset + 0.5 * dim - 0.5)
        offset += dim
    boundaries.append(offset)
    return {
        "labels": labels,
        "dims": dims,
        "tick_positions": centers,
        "boundaries": boundaries,
        "total_dim": offset,
    }


def _rounded_matrix_payload(mat: torch.Tensor) -> list[list[float]]:
    arr = mat.detach().cpu().to(torch.float64).numpy()
    return np.round(arr, 6).tolist()


def _make_cutout_payload(
    gt_dense: torch.Tensor,
    pred_dense: torch.Tensor,
    labels: list[str],
    dims: list[int],
    *,
    selection_label: str,
    worst_edges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    diff_dense = pred_dense - gt_dense
    abs_max = max(
        float(torch.max(torch.abs(gt_dense)).item()) if gt_dense.numel() else 0.0,
        float(torch.max(torch.abs(pred_dense)).item()) if pred_dense.numel() else 0.0,
        float(torch.max(torch.abs(diff_dense)).item()) if diff_dense.numel() else 0.0,
    )
    return {
        "selection_label": selection_label,
        "axes": _cutout_axes_meta(labels, dims),
        "gt": _rounded_matrix_payload(gt_dense),
        "pred": _rounded_matrix_payload(pred_dense),
        "diff": _rounded_matrix_payload(diff_dense),
        "abs_max": abs_max,
        "worst_edges": worst_edges or [],
    }


def _record_payload(
    record: dict[str, Any],
    *,
    include_shift: bool,
) -> dict[str, Any]:
    payload = {
        "pair_key": str(record.get("pair_key", "")),
        "src_atom": int(record.get("src_atom", 0)),
        "dst_atom": int(record.get("dst_atom", 0)),
        "abs_mae": float(record.get("abs_mae", 0.0)),
        "rel_mae": float(record.get("rel_mae", 0.0)),
    }
    if "edge_length" in record:
        payload["edge_length"] = float(record["edge_length"])
    if include_shift:
        shift = record.get("shift", (0, 0, 0))
        payload["shift"] = [int(v) for v in _tuple3(shift)]
    return payload


def _top_edge_records(
    records: list[dict[str, Any]],
    *,
    max_items: int = 10,
    include_shift: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda rec: (
            -float(rec.get("abs_mae", 0.0)),
            -float(rec.get("rel_mae", 0.0)),
            int(rec.get("src_atom", 0)),
            int(rec.get("dst_atom", 0)),
        ),
    )
    return [
        _record_payload(rec, include_shift=include_shift) for rec in ordered[:max_items]
    ]


def _edge_displacement(
    positions: torch.Tensor,
    box: torch.Tensor,
    src_atom: int,
    dst_atom: int,
    shift: tuple[int, int, int],
) -> torch.Tensor:
    shift_t = torch.tensor(shift, dtype=positions.dtype)
    return positions[dst_atom] - positions[src_atom] + shift_t @ box


def _edge_error_records(
    pred,
    target,
    *,
    positions: torch.Tensor,
    box: torch.Tensor,
) -> list[dict[str, Any]]:
    positions_cpu = positions.detach().cpu()
    box_cpu = box.detach().cpu()
    records: list[dict[str, Any]] = []
    eps = 1.0e-12
    for key in target.keys():
        if key not in pred.pair_edges:
            continue
        target_edges = target.pair_edges[key].detach().cpu()
        target_blocks = target.pair_blocks[key].detach().cpu()
        pred_lookup = pred.lookup
        for idx in range(target_edges.shape[1]):
            sx, sy, sz, i, j = [int(v) for v in target_edges[:, idx].tolist()]
            pred_entry = pred_lookup.get((sx, sy, sz, i, j))
            if pred_entry is None:
                continue
            pred_key, pred_idx = pred_entry
            pred_block = pred.pair_blocks[pred_key][pred_idx].detach().cpu()
            target_block = target_blocks[idx]
            disp = _edge_displacement(positions_cpu, box_cpu, i, j, (sx, sy, sz))
            mae = _block_mean_abs(pred_block - target_block)
            magnitude = _block_mean_abs(target_block)
            records.append(
                {
                    "pair_key": str(key),
                    "src_atom": i,
                    "dst_atom": j,
                    "shift": (sx, sy, sz),
                    "edge_index": idx,
                    "edge_length": float(torch.linalg.norm(disp).item()),
                    "abs_mae": mae,
                    "rel_mae": mae / max(magnitude, eps),
                }
            )
    return records


def _error_block_from_lookup(
    pred,
    target,
    *,
    lookup_key: tuple[int, int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    target_entry = target.lookup.get(lookup_key)
    if target_entry is None:
        raise KeyError(f"Target lookup key not found: {lookup_key!r}")
    target_key, target_idx = target_entry
    target_block = target.pair_blocks[target_key][target_idx].detach().cpu()
    pred_entry = pred.lookup.get(lookup_key)
    if pred_entry is None:
        pred_block = torch.zeros_like(target_block)
        return pred_block, target_block, pred_block - target_block, True
    pred_key, pred_idx = pred_entry
    pred_block = pred.pair_blocks[pred_key][pred_idx].detach().cpu()
    return pred_block, target_block, pred_block - target_block, False


def build_snapshot_3d_error_payload(
    pred,
    target,
    *,
    positions: torch.Tensor,
    box: torch.Tensor,
) -> dict[str, Any]:
    positions_cpu = positions.detach().cpu().to(torch.float64)
    box_cpu = box.detach().cpu().to(torch.float64)
    eps = 1.0e-12

    edges_payload: list[dict[str, Any]] = []
    ghost_index: dict[tuple[int, int, int, int], int] = {}
    ghosts_payload: list[dict[str, Any]] = []
    edge_metric_abs_values: list[float] = []
    edge_metric_rel_values: list[float] = []

    for key in target.keys():
        target_edges = target.pair_edges[key].detach().cpu()
        for idx in range(target_edges.shape[1]):
            sx, sy, sz, src_atom, dst_atom = [
                int(v) for v in target_edges[:, idx].tolist()
            ]
            pred_block, target_block, diff_block, missing_pred = (
                _error_block_from_lookup(
                    pred,
                    target,
                    lookup_key=(sx, sy, sz, src_atom, dst_atom),
                )
            )
            start = positions_cpu[src_atom]
            edge_vec = _edge_displacement(
                positions_cpu,
                box_cpu,
                src_atom,
                dst_atom,
                (sx, sy, sz),
            )
            end = start + edge_vec
            abs_mae = _block_mean_abs(diff_block)
            target_mag = _block_mean_abs(target_block)
            rel_mae = abs_mae / max(target_mag, eps)
            edges_payload.append(
                {
                    "pair_key": str(key),
                    "src_atom": src_atom,
                    "dst_atom": dst_atom,
                    "shift": [sx, sy, sz],
                    "start": [float(x) for x in start.tolist()],
                    "end": [float(x) for x in end.tolist()],
                    "edge_length": float(torch.linalg.norm(edge_vec).item()),
                    "abs_mae": float(abs_mae),
                    "rel_mae": float(rel_mae),
                    "missing_pred": bool(missing_pred),
                }
            )
            edge_metric_abs_values.append(float(abs_mae))
            edge_metric_rel_values.append(float(rel_mae))

            if (sx, sy, sz) != (0, 0, 0):
                ghost_key = (dst_atom, sx, sy, sz)
                if ghost_key not in ghost_index:
                    ghost_index[ghost_key] = len(ghosts_payload)
                    ghosts_payload.append(
                        {
                            "atom": int(dst_atom),
                            "shift": [sx, sy, sz],
                            "position": [float(x) for x in end.tolist()],
                        }
                    )

    node_diag_payload: list[dict[str, Any]] = []
    node_metric_abs_values: list[float] = []
    node_metric_rel_values: list[float] = []
    for atom_idx in range(len(target.atoms)):
        pred_block, target_block, diff_block, missing_pred = _error_block_from_lookup(
            pred,
            target,
            lookup_key=(0, 0, 0, atom_idx, atom_idx),
        )
        abs_mae = _block_mean_abs(diff_block)
        target_mag = _block_mean_abs(target_block)
        rel_mae = abs_mae / max(target_mag, eps)
        node_diag_payload.append(
            {
                "atom": int(atom_idx),
                "position": [float(x) for x in positions_cpu[atom_idx].tolist()],
                "abs_mae": float(abs_mae),
                "rel_mae": float(rel_mae),
                "missing_pred": bool(missing_pred),
            }
        )
        node_metric_abs_values.append(float(abs_mae))
        node_metric_rel_values.append(float(rel_mae))

    box_rows = [[float(x) for x in row.tolist()] for row in box_cpu]
    return {
        "atom_count": int(len(target.atoms)),
        "atoms": list(target.atoms),
        "positions": [[float(x) for x in row.tolist()] for row in positions_cpu],
        "box": box_rows,
        "edges": edges_payload,
        "ghosts": ghosts_payload,
        "node_diagonal": node_diag_payload,
        "stats": {
            "edge_abs_max": float(
                max(edge_metric_abs_values) if edge_metric_abs_values else 0.0
            ),
            "edge_rel_max": float(
                max(edge_metric_rel_values) if edge_metric_rel_values else 0.0
            ),
            "node_abs_max": float(
                max(node_metric_abs_values) if node_metric_abs_values else 0.0
            ),
            "node_rel_max": float(
                max(node_metric_rel_values) if node_metric_rel_values else 0.0
            ),
            "edge_abs_min_positive": float(
                min((v for v in edge_metric_abs_values if v > 0.0), default=1.0e-12)
            ),
            "edge_rel_min_positive": float(
                min((v for v in edge_metric_rel_values if v > 0.0), default=1.0e-12)
            ),
            "node_abs_min_positive": float(
                min((v for v in node_metric_abs_values if v > 0.0), default=1.0e-12)
            ),
            "node_rel_min_positive": float(
                min((v for v in node_metric_rel_values if v > 0.0), default=1.0e-12)
            ),
        },
    }


def save_snapshot_3d_error_payload(
    pred,
    target,
    *,
    positions: torch.Tensor,
    box: torch.Tensor,
    output_path: Path,
) -> None:
    payload = build_snapshot_3d_error_payload(
        pred,
        target,
        positions=positions,
        box=box,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _shift_resolved_closest_selection(
    records: list[dict[str, Any]],
    *,
    anchor_atom: int,
    max_neighbors: int,
) -> list[tuple[int, tuple[int, int, int]]]:
    nearest_by_atom: dict[int, dict[str, Any]] = {}
    for record in records:
        if int(record["src_atom"]) != int(anchor_atom):
            continue
        dst_atom = int(record["dst_atom"])
        if dst_atom == int(anchor_atom):
            continue
        current = nearest_by_atom.get(dst_atom)
        if current is None or float(record["edge_length"]) < float(
            current["edge_length"]
        ):
            nearest_by_atom[dst_atom] = record
    neighbors = sorted(
        nearest_by_atom.values(),
        key=lambda rec: (float(rec["edge_length"]), int(rec["dst_atom"])),
    )[: max(0, int(max_neighbors))]
    return [(int(anchor_atom), (0, 0, 0))] + [
        (int(rec["dst_atom"]), _tuple3(rec["shift"])) for rec in neighbors
    ]


def _sum_pbc_closest_selection(
    records: list[dict[str, Any]],
    *,
    anchor_atom: int,
    max_neighbors: int,
) -> list[int]:
    nearest_by_atom: dict[int, float] = {}
    for record in records:
        if int(record["src_atom"]) != int(anchor_atom):
            continue
        dst_atom = int(record["dst_atom"])
        if dst_atom == int(anchor_atom):
            continue
        dist = float(record["edge_length"])
        prev = nearest_by_atom.get(dst_atom)
        if prev is None or dist < prev:
            nearest_by_atom[dst_atom] = dist
    ordered = sorted(nearest_by_atom.items(), key=lambda item: (item[1], item[0]))
    return [int(anchor_atom)] + [
        atom for atom, _dist in ordered[: max(0, int(max_neighbors))]
    ]


def _shift_resolved_ranked_selection(
    sorted_records: list[dict[str, Any]],
    *,
    max_nodes: int,
) -> list[tuple[int, tuple[int, int, int]]]:
    nodes: list[tuple[int, tuple[int, int, int]]] = []
    seen = set()
    for record in sorted_records:
        for node in (
            (int(record["src_atom"]), (0, 0, 0)),
            (int(record["dst_atom"]), _tuple3(record["shift"])),
        ):
            if node in seen:
                continue
            nodes.append(node)
            seen.add(node)
            if len(nodes) >= int(max_nodes):
                return nodes
    return nodes


def _sum_pair_error_records(
    pred_dense: torch.Tensor,
    target_dense: torch.Tensor,
    atoms: tuple[str, ...],
    orbital_cfg: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    eps = 1.0e-12
    atom_count = len(atoms)
    for src_atom in range(atom_count):
        for dst_atom in range(atom_count):
            pred_block = _pair_dense_block(
                pred_dense, atoms, orbital_cfg, src_atom, dst_atom
            )
            target_block = _pair_dense_block(
                target_dense, atoms, orbital_cfg, src_atom, dst_atom
            )
            mae = _block_mean_abs(pred_block - target_block)
            magnitude = _block_mean_abs(target_block)
            records.append(
                {
                    "src_atom": src_atom,
                    "dst_atom": dst_atom,
                    "abs_mae": mae,
                    "rel_mae": mae / max(magnitude, eps),
                }
            )
    return records


def _sum_pbc_ranked_selection(
    sorted_pair_records: list[dict[str, Any]],
    *,
    max_nodes: int,
) -> list[int]:
    atoms: list[int] = []
    seen = set()
    for record in sorted_pair_records:
        for atom_idx in (int(record["src_atom"]), int(record["dst_atom"])):
            if atom_idx in seen:
                continue
            atoms.append(atom_idx)
            seen.add(atom_idx)
            if len(atoms) >= int(max_nodes):
                return atoms
    return atoms


def _build_shift_resolved_dense_cutout(
    mat,
    node_images: list[tuple[int, tuple[int, int, int]]],
) -> tuple[torch.Tensor, list[int]]:
    dims = _orbital_dims_for_atoms(mat.atoms, mat.orbital_cfg)
    device = next(iter(mat.pair_blocks.values())).device
    dtype = next(iter(mat.pair_blocks.values())).dtype
    row_chunks: list[torch.Tensor] = []
    dims_selected = [dims[atom_idx] for atom_idx, _shift in node_images]
    for row_atom, row_shift in node_images:
        row_dim = dims[row_atom]
        col_chunks: list[torch.Tensor] = []
        for col_atom, col_shift in node_images:
            col_dim = dims[col_atom]
            shift = (
                int(col_shift[0] - row_shift[0]),
                int(col_shift[1] - row_shift[1]),
                int(col_shift[2] - row_shift[2]),
            )
            block = torch.zeros((row_dim, col_dim), dtype=dtype, device=device)
            entry = mat.lookup.get((shift[0], shift[1], shift[2], row_atom, col_atom))
            if entry is not None:
                key, idx = entry
                block = mat.pair_blocks[key][idx]
            col_chunks.append(block)
        row_chunks.append(torch.cat(col_chunks, dim=1))
    return torch.cat(row_chunks, dim=0), dims_selected


def _random_shift_resolved_selections(
    records: list[dict[str, Any]],
    *,
    max_nodes: int,
    count: int,
) -> list[list[tuple[int, tuple[int, int, int]]]]:
    selections = []
    base_records = list(records)
    for seed in range(int(count)):
        shuffled = list(base_records)
        random.Random(seed).shuffle(shuffled)
        selection = _shift_resolved_ranked_selection(
            shuffled,
            max_nodes=max_nodes,
        )
        if selection:
            selections.append(selection)
    return selections


def _random_sum_pbc_selections(
    pair_records: list[dict[str, Any]],
    *,
    max_nodes: int,
    count: int,
) -> list[list[int]]:
    selections = []
    base_records = list(pair_records)
    for seed in range(int(count)):
        shuffled = list(base_records)
        random.Random(seed).shuffle(shuffled)
        selection = _sum_pbc_ranked_selection(shuffled, max_nodes=max_nodes)
        if selection:
            selections.append(selection)
    return selections


def _sum_pbc_records_for_atoms(
    pair_records: list[dict[str, Any]],
    selected_atoms: list[int],
) -> list[dict[str, Any]]:
    selected = set(int(atom) for atom in selected_atoms)
    return [
        rec
        for rec in pair_records
        if int(rec["src_atom"]) in selected and int(rec["dst_atom"]) in selected
    ]


def _shift_resolved_records_for_nodes(
    records: list[dict[str, Any]],
    selected_nodes: list[tuple[int, tuple[int, int, int]]],
) -> list[dict[str, Any]]:
    selected_atoms = {int(atom_idx) for atom_idx, _shift in selected_nodes}
    return [
        rec
        for rec in records
        if int(rec["src_atom"]) in selected_atoms
        and int(rec["dst_atom"]) in selected_atoms
    ]


def build_hamiltonian_interactive_heatmap_payload(
    pred,
    target,
    *,
    positions: torch.Tensor,
    box: torch.Tensor,
    default_clim: float,
    max_nodes: int = 6,
    random_count: int = 12,
    closest_neighbor_count: int = 5,
) -> dict[str, Any]:
    pred_dense_full = pred.to_dense().detach().cpu().to(torch.float64)
    target_dense_full = target.to_dense().detach().cpu().to(torch.float64)
    records = _edge_error_records(pred, target, positions=positions, box=box)
    atom_count = len(target.atoms)
    pair_records = _sum_pair_error_records(
        pred_dense_full,
        target_dense_full,
        target.atoms,
        target.orbital_cfg,
    )

    sorted_abs_edges = sorted(
        records,
        key=lambda rec: (
            -float(rec["abs_mae"]),
            rec["src_atom"],
            rec["dst_atom"],
            rec["shift"],
        ),
    )
    sorted_rel_edges = sorted(
        records,
        key=lambda rec: (
            -float(rec["rel_mae"]),
            rec["src_atom"],
            rec["dst_atom"],
            rec["shift"],
        ),
    )
    sorted_abs_pairs = sorted(
        pair_records,
        key=lambda rec: (-float(rec["abs_mae"]), rec["src_atom"], rec["dst_atom"]),
    )
    sorted_rel_pairs = sorted(
        pair_records,
        key=lambda rec: (-float(rec["rel_mae"]), rec["src_atom"], rec["dst_atom"]),
    )

    payload: dict[str, Any] = {
        "default_clim": float(default_clim),
        "atom_count": int(atom_count),
        "max_nodes": int(max_nodes),
        "closest_neighbor_count": int(closest_neighbor_count),
        "sum_pbc": {
            "closest_neighbors": {},
            "random": [],
        },
        "shift_resolved": {
            "closest_neighbors": {},
            "random": [],
        },
    }

    max_abs_value = 0.0

    def register_cutout(
        container: dict[str, Any], key: str, cutout: dict[str, Any]
    ) -> None:
        nonlocal max_abs_value
        container[key] = cutout
        max_abs_value = max(max_abs_value, float(cutout["abs_max"]))

    def register_random(container: list[Any], cutout: dict[str, Any]) -> None:
        nonlocal max_abs_value
        container.append(cutout)
        max_abs_value = max(max_abs_value, float(cutout["abs_max"]))

    for anchor_atom in range(atom_count):
        sum_atoms = _sum_pbc_closest_selection(
            records,
            anchor_atom=anchor_atom,
            max_neighbors=closest_neighbor_count,
        )
        if sum_atoms:
            sum_records = _sum_pbc_records_for_atoms(pair_records, sum_atoms)
            gt_dense, dims = _dense_atom_subset(
                target_dense_full,
                target.atoms,
                target.orbital_cfg,
                sum_atoms,
            )
            pred_dense, _ = _dense_atom_subset(
                pred_dense_full,
                pred.atoms,
                pred.orbital_cfg,
                sum_atoms,
            )
            register_cutout(
                payload["sum_pbc"]["closest_neighbors"],
                str(anchor_atom),
                _make_cutout_payload(
                    gt_dense,
                    pred_dense,
                    [str(atom_idx) for atom_idx in sum_atoms],
                    dims,
                    selection_label=f"Closest neighbors around atom {anchor_atom}",
                    worst_edges=_top_edge_records(
                        sum_records,
                        max_items=10,
                        include_shift=False,
                    ),
                ),
            )

        shift_nodes = _shift_resolved_closest_selection(
            records,
            anchor_atom=anchor_atom,
            max_neighbors=closest_neighbor_count,
        )
        if shift_nodes:
            shift_records = _shift_resolved_records_for_nodes(records, shift_nodes)
            gt_dense, dims = _build_shift_resolved_dense_cutout(target, shift_nodes)
            pred_dense, _ = _build_shift_resolved_dense_cutout(pred, shift_nodes)
            register_cutout(
                payload["shift_resolved"]["closest_neighbors"],
                str(anchor_atom),
                _make_cutout_payload(
                    gt_dense.detach().cpu().to(torch.float64),
                    pred_dense.detach().cpu().to(torch.float64),
                    [
                        _node_image_label(atom_idx, shift)
                        for atom_idx, shift in shift_nodes
                    ],
                    dims,
                    selection_label=f"Closest neighbors around atom {anchor_atom}",
                    worst_edges=_top_edge_records(
                        shift_records,
                        max_items=10,
                        include_shift=True,
                    ),
                ),
            )

    worst_abs_atoms = _sum_pbc_ranked_selection(sorted_abs_pairs, max_nodes=max_nodes)
    worst_rel_atoms = _sum_pbc_ranked_selection(sorted_rel_pairs, max_nodes=max_nodes)
    for key, atoms_selected, label in (
        ("worst_abs", worst_abs_atoms, "Worst absolute pair errors"),
        ("worst_rel", worst_rel_atoms, "Worst relative pair errors"),
    ):
        if atoms_selected:
            sum_records = _sum_pbc_records_for_atoms(pair_records, atoms_selected)
            gt_dense, dims = _dense_atom_subset(
                target_dense_full,
                target.atoms,
                target.orbital_cfg,
                atoms_selected,
            )
            pred_dense, _ = _dense_atom_subset(
                pred_dense_full,
                pred.atoms,
                pred.orbital_cfg,
                atoms_selected,
            )
            register_cutout(
                payload["sum_pbc"],
                key,
                _make_cutout_payload(
                    gt_dense,
                    pred_dense,
                    [str(atom_idx) for atom_idx in atoms_selected],
                    dims,
                    selection_label=label,
                    worst_edges=_top_edge_records(
                        sum_records,
                        max_items=10,
                        include_shift=False,
                    ),
                ),
            )

    worst_abs_nodes = _shift_resolved_ranked_selection(
        sorted_abs_edges,
        max_nodes=max_nodes,
    )
    worst_rel_nodes = _shift_resolved_ranked_selection(
        sorted_rel_edges,
        max_nodes=max_nodes,
    )
    for key, nodes_selected, label in (
        ("worst_abs", worst_abs_nodes, "Worst absolute edge errors"),
        ("worst_rel", worst_rel_nodes, "Worst relative edge errors"),
    ):
        if nodes_selected:
            shift_records = _shift_resolved_records_for_nodes(records, nodes_selected)
            gt_dense, dims = _build_shift_resolved_dense_cutout(target, nodes_selected)
            pred_dense, _ = _build_shift_resolved_dense_cutout(pred, nodes_selected)
            register_cutout(
                payload["shift_resolved"],
                key,
                _make_cutout_payload(
                    gt_dense.detach().cpu().to(torch.float64),
                    pred_dense.detach().cpu().to(torch.float64),
                    [
                        _node_image_label(atom_idx, shift)
                        for atom_idx, shift in nodes_selected
                    ],
                    dims,
                    selection_label=label,
                    worst_edges=_top_edge_records(
                        shift_records,
                        max_items=10,
                        include_shift=True,
                    ),
                ),
            )

    for atom_selection in _random_sum_pbc_selections(
        pair_records,
        max_nodes=max_nodes,
        count=random_count,
    ):
        sum_records = _sum_pbc_records_for_atoms(pair_records, atom_selection)
        gt_dense, dims = _dense_atom_subset(
            target_dense_full,
            target.atoms,
            target.orbital_cfg,
            atom_selection,
        )
        pred_dense, _ = _dense_atom_subset(
            pred_dense_full,
            pred.atoms,
            pred.orbital_cfg,
            atom_selection,
        )
        register_random(
            payload["sum_pbc"]["random"],
            _make_cutout_payload(
                gt_dense,
                pred_dense,
                [str(atom_idx) for atom_idx in atom_selection],
                dims,
                selection_label="Random summed-PBC selection",
                worst_edges=_top_edge_records(
                    sum_records,
                    max_items=10,
                    include_shift=False,
                ),
            ),
        )

    for node_selection in _random_shift_resolved_selections(
        records,
        max_nodes=max_nodes,
        count=random_count,
    ):
        shift_records = _shift_resolved_records_for_nodes(records, node_selection)
        gt_dense, dims = _build_shift_resolved_dense_cutout(target, node_selection)
        pred_dense, _ = _build_shift_resolved_dense_cutout(pred, node_selection)
        register_random(
            payload["shift_resolved"]["random"],
            _make_cutout_payload(
                gt_dense.detach().cpu().to(torch.float64),
                pred_dense.detach().cpu().to(torch.float64),
                [
                    _node_image_label(atom_idx, shift)
                    for atom_idx, shift in node_selection
                ],
                dims,
                selection_label="Random shift-resolved selection",
                worst_edges=_top_edge_records(
                    shift_records,
                    max_items=10,
                    include_shift=True,
                ),
            ),
        )

    payload["max_clim"] = float(max(max_abs_value, float(default_clim)))
    return payload


def save_hamiltonian_interactive_heatmap_payload(
    pred,
    target,
    *,
    positions: torch.Tensor,
    box: torch.Tensor,
    output_path: Path,
    default_clim: float,
    max_nodes: int = 6,
    random_count: int = 12,
    closest_neighbor_count: int = 5,
) -> None:
    payload = build_hamiltonian_interactive_heatmap_payload(
        pred,
        target,
        positions=positions,
        box=box,
        default_clim=default_clim,
        max_nodes=max_nodes,
        random_count=random_count,
        closest_neighbor_count=closest_neighbor_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def save_correlation_plot(
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
    from net.artifacts import _as_dense

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
    r2 = float(corr * corr)

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
        f"R^2 = {r2:.4f}\nN = {pred_np.size}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    torch.save(
        {
            "kind": "correlation",
            "title": title,
            "pred": pred_dense.detach().cpu(),
            "target": target_dense.detach().cpu(),
            "corr": corr,
            "r2": r2,
            "bound": bound,
            "lo": lo,
            "hi": hi,
            "sampled_points": int(pred_dense.numel()),
        },
        output_path.with_suffix(".pt"),
    )


def compute_dos_data(
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


def save_dos_comparison_plot(
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
    eig_pred, _eig_pred_window, grid_pred, dos_pred = compute_dos_data(
        h_pred,
        s_pred,
        sigma=sigma,
        bin_width=bin_width,
        energy_min=energy_min,
        energy_max=energy_max,
        overlap_psd_cleanup=overlap_psd_cleanup,
        overlap_jitter=overlap_jitter,
    )
    eig_true, _eig_true_window, grid_true, dos_true = compute_dos_data(
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
    fermi_true = fermi_level_from_dos(grid_true, dos_true, num_electrons_true)
    fermi_pred = fermi_level_from_dos(grid_pred, dos_pred, num_electrons_pred)
    plot_grid_true = grid_true.detach().cpu()
    plot_grid_pred = grid_pred.detach().cpu()
    if fermi_true is not None:
        plot_grid_true = plot_grid_true - float(fermi_true)
        plot_grid_pred = plot_grid_pred - float(fermi_true)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(
        plot_grid_true.numpy(), dos_true.cpu().numpy(), label="Ground Truth", lw=1.8
    )
    ax.plot(plot_grid_pred.numpy(), dos_pred.cpu().numpy(), label="Prediction", lw=1.4)
    if fermi_true is not None:
        ax.axvline(
            0.0,
            color="black",
            ls="--",
            lw=1.2,
            alpha=0.85,
            label="GT $E_F$",
        )
    if fermi_pred is not None:
        ax.axvline(
            float(fermi_pred - fermi_true) if fermi_true is not None else fermi_pred,
            color="#1f5aa6",
            ls=":",
            lw=1.4,
            alpha=0.9,
            label=(
                f"Pred $E_F - E_F^{{GT}}$ = {fermi_pred - fermi_true:.3f} eV"
                if fermi_true is not None
                else f"Pred $E_F$ = {fermi_pred:.3f} eV"
            ),
        )
    ax.set_title(title)
    ax.set_xlabel(
        "Energy - $E_F^{GT}$ (eV)" if fermi_true is not None else "Energy (eV)"
    )
    ax.set_ylabel("DOS")
    ax.grid(True, alpha=0.25)
    if fermi_true is not None:
        # DOS is plotted relative to the GT Fermi level.  The scientifically
        # useful window is symmetric around E_F; using the absolute DOS grid
        # bounds here can hide the entire unoccupied (+E) side.
        ax.set_xlim(-10.0, 10.0)
    else:
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
    torch.save(
        {
            "kind": "dos_comparison",
            "title": title,
            "grid_true": plot_grid_true.detach().cpu(),
            "dos_true": dos_true.detach().cpu(),
            "grid_pred": plot_grid_pred.detach().cpu(),
            "dos_pred": dos_pred.detach().cpu(),
            "dos_error": (dos_pred - dos_true).detach().cpu(),
            "num_electrons_true": (
                None if num_electrons_true is None else float(num_electrons_true)
            ),
            "num_electrons_pred": (
                None if num_electrons_pred is None else float(num_electrons_pred)
            ),
            "fermi_true_ev": None if fermi_true is None else float(fermi_true),
            "fermi_pred_ev": None if fermi_pred is None else float(fermi_pred),
            "energy_min_ev": float(energy_min),
            "energy_max_ev": float(energy_max),
            "method": "gaussian",
        },
        output_path.with_suffix(".pt"),
    )
    if error_output_path is not None:
        error_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
        dos_error = dos_pred - dos_true
        ax.plot(
            plot_grid_true.numpy(),
            dos_error.cpu().numpy(),
            color="#b23a48",
            lw=1.4,
            label="Prediction - Ground Truth",
        )
        ax.axhline(0.0, color="black", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(title.replace("comparison", "error"))
        ax.set_xlabel(
            "Energy - $E_F^{GT}$ (eV)" if fermi_true is not None else "Energy (eV)"
        )
        ax.set_ylabel("DOS Error")
        ax.grid(True, alpha=0.25)
        if fermi_true is not None:
            ax.set_xlim(-10.0, 10.0)
        else:
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


def save_dos_prediction_plot(
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
    _, _, grid, dos = compute_dos_data(
        h_pred,
        s_pred,
        sigma=sigma,
        bin_width=bin_width,
        energy_min=energy_min,
        energy_max=energy_max,
        overlap_psd_cleanup=overlap_psd_cleanup,
        overlap_jitter=overlap_jitter,
    )
    fermi = fermi_level_from_dos(grid, dos, num_electrons)
    grid, dos = _clip_energy_curve(
        grid.detach().cpu(),
        dos.detach().cpu(),
        energy_min=energy_min,
        energy_max=energy_max,
    )
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(grid.numpy(), dos.numpy(), lw=1.6, color="#1f5aa6")
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
    torch.save(
        {
            "kind": "dos_prediction",
            "title": title,
            "grid": grid.detach().cpu(),
            "dos": dos.detach().cpu(),
            "num_electrons": None if num_electrons is None else float(num_electrons),
            "fermi_ev": None if fermi is None else float(fermi),
            "energy_min_ev": float(energy_min),
            "energy_max_ev": float(energy_max),
            "method": "gaussian",
        },
        output_path.with_suffix(".pt"),
    )


def save_tetrahedron_dos_comparison_plot(
    gt_snapshot: Snapshot,
    pred_snapshot: Snapshot,
    output_path: Path,
    *,
    kmesh_spec: str,
    chunk_size: int,
    num_workers: int,
    energy_min: float,
    energy_max: float,
    title: str,
    error_output_path: Path | None = None,
    overlap_psd_cleanup: bool = False,
    overlap_jitter: bool = False,
    bin_width: float = 0.05,
    tetra_batch_size: int = 256,
    show_progress: bool | None = None,
    cache_path_true: Path | None = None,
    cache_path_pred: Path | None = None,
    progress_label: str = "Tetrahedron DOS k-mesh eigensolve",
) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if show_progress is None:
        show_progress = sys.stderr.isatty()
    grid_ev = torch.linspace(
        float(energy_min),
        float(energy_max),
        int(np.ceil((float(energy_max) - float(energy_min)) / float(bin_width))) + 1,
        dtype=torch.float64,
    )
    grid_true, dos_true, num_electrons_true, dos_target_true, fermi_true = (
        compute_tetrahedron_dos_and_fermi(
            gt_snapshot,
            kmesh_spec=kmesh_spec,
            chunk_size=chunk_size,
            num_workers=num_workers,
            psd_cleanup=overlap_psd_cleanup,
            allow_jitter=overlap_jitter,
            bin_width=bin_width,
            tetra_batch_size=tetra_batch_size,
            grid_ev=grid_ev,
            e_min=energy_min,
            e_max=energy_max,
            show_progress=show_progress,
            cache_path=cache_path_true,
            progress_label=f"{progress_label} (GT)",
        )
    )
    grid_pred, dos_pred, num_electrons_pred, dos_target_pred, fermi_pred = (
        compute_tetrahedron_dos_and_fermi(
            pred_snapshot,
            kmesh_spec=kmesh_spec,
            chunk_size=chunk_size,
            num_workers=num_workers,
            psd_cleanup=overlap_psd_cleanup,
            allow_jitter=overlap_jitter,
            bin_width=bin_width,
            tetra_batch_size=tetra_batch_size,
            grid_ev=grid_ev,
            e_min=energy_min,
            e_max=energy_max,
            show_progress=show_progress,
            cache_path=cache_path_pred,
            progress_label=f"{progress_label} (prediction)",
        )
    )
    grid_true, dos_true = _clip_energy_curve(
        grid_true.detach().cpu(),
        dos_true.detach().cpu(),
        energy_min=energy_min,
        energy_max=energy_max,
    )
    grid_pred, dos_pred = _clip_energy_curve(
        grid_pred.detach().cpu(),
        dos_pred.detach().cpu(),
        energy_min=energy_min,
        energy_max=energy_max,
    )
    plot_grid_true = grid_true
    plot_grid_pred = grid_pred
    if fermi_true is not None:
        plot_grid_true = plot_grid_true - float(fermi_true)
        plot_grid_pred = plot_grid_pred - float(fermi_true)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(plot_grid_true.numpy(), dos_true.numpy(), label="Ground Truth", lw=1.8)
    ax.plot(plot_grid_pred.numpy(), dos_pred.numpy(), label="Prediction", lw=1.4)
    if fermi_true is not None:
        ax.axvline(
            0.0,
            color="black",
            ls="--",
            lw=1.2,
            alpha=0.85,
            label="GT $E_F$",
        )
    if fermi_pred is not None:
        ax.axvline(
            float(fermi_pred - fermi_true) if fermi_true is not None else fermi_pred,
            color="#1f5aa6",
            ls=":",
            lw=1.4,
            alpha=0.9,
            label=(
                f"Pred $E_F - E_F^{{GT}}$ = {fermi_pred - fermi_true:.3f} eV"
                if fermi_true is not None
                else f"Pred $E_F$ = {fermi_pred:.3f} eV"
            ),
        )
    ax.set_title(title)
    ax.set_xlabel(
        "Energy - $E_F^{GT}$ (eV)" if fermi_true is not None else "Energy (eV)"
    )
    ax.set_ylabel("DOS")
    ax.grid(True, alpha=0.25)
    if fermi_true is not None:
        ax.set_xlim(-10.0, 10.0)
    else:
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
        ax.plot(
            plot_grid_true.numpy(),
            (dos_pred - dos_true).numpy(),
            color="#b23a48",
            lw=1.4,
            label="Prediction - Ground Truth",
        )
        ax.axhline(0.0, color="black", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(title.replace("comparison", "error"))
        ax.set_xlabel(
            "Energy - $E_F^{GT}$ (eV)" if fermi_true is not None else "Energy (eV)"
        )
        ax.set_ylabel("DOS Error")
        ax.grid(True, alpha=0.25)
        if fermi_true is not None:
            ax.set_xlim(-10.0, 10.0)
        else:
            ax.set_xlim(left=energy_min, right=energy_max)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(error_output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    torch.save(
        {
            "kind": "dos_comparison",
            "title": title,
            "grid_true": plot_grid_true.detach().cpu(),
            "dos_true": dos_true.detach().cpu(),
            "grid_pred": plot_grid_pred.detach().cpu(),
            "dos_pred": dos_pred.detach().cpu(),
            "dos_error": (dos_pred - dos_true).detach().cpu(),
            "num_electrons_true": (
                None if num_electrons_true is None else float(num_electrons_true)
            ),
            "num_electrons_pred": (
                None if num_electrons_pred is None else float(num_electrons_pred)
            ),
            "fermi_true_ev": None if fermi_true is None else float(fermi_true),
            "fermi_pred_ev": None if fermi_pred is None else float(fermi_pred),
            "energy_min_ev": float(energy_min),
            "energy_max_ev": float(energy_max),
            "method": "tetrahedron",
        },
        output_path.with_suffix(".pt"),
    )
    return {
        "dos_abs_mean": float(torch.mean(torch.abs(dos_pred - dos_true)).item()),
        "dos_abs_max": float(torch.max(torch.abs(dos_pred - dos_true)).item()),
        "fermi_gt_ev": float(fermi_true) if fermi_true is not None else float("nan"),
        "fermi_pred_ev": float(fermi_pred) if fermi_pred is not None else float("nan"),
        "dos_target_gt": (
            float(dos_target_true) if dos_target_true is not None else float("nan")
        ),
        "dos_target_pred": (
            float(dos_target_pred) if dos_target_pred is not None else float("nan")
        ),
    }


def save_tetrahedron_dos_prediction_plot(
    snapshot: Snapshot,
    output_path: Path,
    *,
    kmesh_spec: str,
    chunk_size: int,
    num_workers: int,
    energy_min: float,
    energy_max: float,
    num_electrons: float | None,
    title: str,
    overlap_psd_cleanup: bool = False,
    overlap_jitter: bool = False,
    bin_width: float = 0.05,
    tetra_batch_size: int = 256,
    show_progress: bool | None = None,
    cache_path: Path | None = None,
    progress_label: str = "Tetrahedron DOS k-mesh eigensolve",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if show_progress is None:
        show_progress = sys.stderr.isatty()
    grid_ev = torch.linspace(
        float(energy_min),
        float(energy_max),
        int(np.ceil((float(energy_max) - float(energy_min)) / float(bin_width))) + 1,
        dtype=torch.float64,
    )
    grid, dos, _num_electrons, _dos_target, fermi = compute_tetrahedron_dos_and_fermi(
        snapshot,
        kmesh_spec=kmesh_spec,
        chunk_size=chunk_size,
        num_workers=num_workers,
        psd_cleanup=overlap_psd_cleanup,
        allow_jitter=overlap_jitter,
        bin_width=bin_width,
        tetra_batch_size=tetra_batch_size,
        grid_ev=grid_ev,
        e_min=energy_min,
        e_max=energy_max,
        show_progress=show_progress,
        cache_path=cache_path,
        progress_label=progress_label,
    )
    grid, dos = _clip_energy_curve(
        grid.detach().cpu(),
        dos.detach().cpu(),
        energy_min=energy_min,
        energy_max=energy_max,
    )
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(grid.numpy(), dos.numpy(), lw=1.6, color="#1f5aa6")
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
    torch.save(
        {
            "kind": "dos_prediction",
            "title": title,
            "grid": grid.detach().cpu(),
            "dos": dos.detach().cpu(),
            "num_electrons": None if num_electrons is None else float(num_electrons),
            "fermi_ev": None if fermi is None else float(fermi),
            "energy_min_ev": float(energy_min),
            "energy_max_ev": float(energy_max),
            "method": "tetrahedron",
        },
        output_path.with_suffix(".pt"),
    )


def save_band_structure_comparison_plot(
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
    tick_labels = [display_k_label(label) for label in gt_band.tick_labels]

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


def save_band_structure_prediction_plot(
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
    tick_labels = [display_k_label(label) for label in band.tick_labels]
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


def _band_worker_init() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _band_chunk_worker(task: tuple[int, int]) -> torch.Tensor:
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


def compute_band_chunks_parallel(
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
    progress_label: str = "Band-path eigensolve",
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
    except ValueError as exc:  # pragma: no cover - platform specific
        raise RuntimeError(
            "multiprocessing fork context is required for parallel band chunks"
        ) from exc

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
        from tqdm.auto import tqdm

        with ctx.Pool(worker_count, initializer=_band_worker_init) as pool:
            result_iter = pool.imap(_band_chunk_worker, tasks, chunksize=1)
            if len(tasks) > 1:
                result_iter = tqdm(result_iter, total=len(tasks), desc=progress_label)
            return list(result_iter)
    finally:
        _BAND_MP_STATE = None


def build_band_structure_from_chunks(
    snapshot: Snapshot,
    *,
    path_string: str,
    special_points: dict[str, list[float]],
    num_points: int,
    chunk_size: int,
    num_workers: int,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
    progress_label: str = "Band-path eigensolve",
) -> BandStructure:
    (
        _resolved_path_string,
        fractional_kpoints,
        kpoints_abs,
        linear_k,
        tick_positions,
        tick_labels,
    ) = build_band_path(
        snapshot.box,
        path=path_string,
        special_points=special_points,
        npoints=num_points,
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
        from tqdm.auto import tqdm

        task_iter = tasks
        if len(tasks) > 1:
            task_iter = tqdm(tasks, total=len(tasks), desc=progress_label)
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
        eig_chunks = compute_band_chunks_parallel(
            ham_shift=ham_shift.detach().cpu(),
            ovl_shift=ovl_shift.detach().cpu(),
            shift_t=shift_t.detach().cpu(),
            box=snapshot.box.detach().cpu(),
            kpoints_abs=kpoints_abs.detach().cpu(),
            tasks=tasks,
            num_workers=num_workers,
            overlap_psd_cleanup=overlap_psd_cleanup,
            overlap_jitter=overlap_jitter,
            progress_label=progress_label,
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


def band_cache_matches_options(
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


def band_cache_path(
    output_dir: Path, *, kind: str, use_gt_overlap_for_eigs: bool
) -> Path:
    suffix = "_gt_overlap" if use_gt_overlap_for_eigs else ""
    return output_dir / f"band_structure_{kind}{suffix}.pt"


def band_structure_to_payload(
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


def band_structure_from_payload(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(
        eigenvalues=payload["eigenvalues"],
        kpoints_abs=payload["kpoints_abs"],
        linear_k=payload["linear_k"],
        tick_positions=payload["tick_positions"],
        tick_labels=list(payload["tick_labels"]),
        fractional_kpoints=payload.get("fractional_kpoints"),
        fermi_level=payload.get("fermi_level"),
    )


def compute_or_load_band_structure(
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
    progress_label: str = "Band-path eigensolve",
) -> Any:
    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if band_cache_matches_options(
            payload,
            path_string=path_string,
            overlap_psd_cleanup=overlap_psd_cleanup,
            overlap_jitter=overlap_jitter,
        ):
            return band_structure_from_payload(payload)
    band_structure = build_band_structure_from_chunks(
        snapshot,
        path_string=path_string,
        special_points=special_points,
        num_points=num_points,
        chunk_size=chunk_size,
        num_workers=num_workers,
        overlap_psd_cleanup=overlap_psd_cleanup,
        overlap_jitter=overlap_jitter,
        progress_label=progress_label,
    )
    band_structure.overlap_psd_cleanup = bool(overlap_psd_cleanup)
    band_structure.overlap_jitter = bool(overlap_jitter)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        band_structure_to_payload(band_structure, path_string=path_string), cache_path
    )
    return band_structure
