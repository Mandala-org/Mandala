from __future__ import annotations

import itertools
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
    info_path: Path, requested_path_string: str
) -> tuple[str, dict[str, list[float]]]:
    resolved = band_path_from_openmx_info(info_path)
    if resolved is None:
        raise ValueError(
            f"No OpenMX Band.kpath found in {info_path}; the hardcoded fallback was removed."
        )
    openmx_path_string, special_points = resolved
    if requested_path_string == DEFAULT_PATH_STRING:
        return openmx_path_string, special_points
    return requested_path_string, special_points


def display_k_label(label: str) -> str:
    label_str = str(label).strip()
    if label_str in {"G", "Gamma", r"$\Gamma$", "$\\Gamma$"}:
        return r"$\Gamma$"
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
            try:
                from tqdm.auto import tqdm
            except ImportError:  # pragma: no cover - optional dependency
                tqdm = None
            if tqdm is not None:
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
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - optional dependency
            tqdm = None
        with ctx.Pool(worker_count, initializer=_tetra_worker_init) as pool:
            result_iter = pool.imap(_tetra_batch_worker, tasks, chunksize=1)
            if show_progress and tqdm is not None and len(tasks) > 1:
                result_iter = tqdm(
                    result_iter, total=len(tasks), desc="Tetrahedron batches"
                )
            for cdf_part, pdf_part in result_iter:
                cumulative = cumulative + cdf_part
                dos = dos + pdf_part
        return grid_ev, dos, cumulative
    finally:
        _TETRA_MP_STATE = None


def _kmesh_eigenvalues(
    snapshot: Snapshot,
    *,
    kmesh_spec: str,
    chunk_size: int,
    num_workers: int,
    psd_cleanup: bool,
    allow_jitter: bool,
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
) -> tuple[torch.Tensor, torch.Tensor, float, float | None, float]:
    if show_progress is None:
        show_progress = sys.stderr.isatty()
    eigenvalues_ev, _fractional_kpoints = _kmesh_eigenvalues(
        snapshot,
        kmesh_spec=kmesh_spec,
        chunk_size=chunk_size,
        num_workers=num_workers,
        psd_cleanup=psd_cleanup,
        allow_jitter=allow_jitter,
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
    )
    num_electrons = float(snapshot.get_number_of_electrons().detach().cpu().item())
    dos_electron_target = effective_dos_electron_target(snapshot)
    fermi_level_ev = fermi_level_from_dos(grid_ev, dos, dos_electron_target)
    if (
        fermi_level_ev is None
        and getattr(snapshot.info, "fermi_level", None) is not None
    ):
        fermi_level_ev = float(snapshot.info.fermi_level.item() * HARTREE_TO_EV)
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
        if band_fermi_ev is not None:
            ref_energy = ref_energy - band_fermi_ev
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

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(grid_true.numpy(), dos_true.numpy(), label="Ground Truth", lw=1.8)
    ax.plot(grid_pred.numpy(), dos_pred.numpy(), label="Prediction", lw=1.4)
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
        ax.plot(
            grid_true.numpy(),
            (dos_pred - dos_true).numpy(),
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
) -> BandStructure:
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
    )
    band_structure.overlap_psd_cleanup = bool(overlap_psd_cleanup)
    band_structure.overlap_jitter = bool(overlap_jitter)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        band_structure_to_payload(band_structure, path_string=path_string), cache_path
    )
    return band_structure
