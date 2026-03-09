#!/usr/bin/env python3
"""Compute a cheap periodic PySCF baseline from an OpenMX ``.info.out`` snapshot.

Defaults are intentionally light-weight:
- periodic (PBC) calculation
- heuristic (no-SCF) method
- shell-count-matched custom basis derived from OpenMX orbital_set
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.openmx_info_parser import parse_info_out

HARTREE_TO_EV = 27.211386245988
DEFAULT_INFO_PATH = Path("data/small/H2O/original/H2O.info.out")
DEFAULT_OUTPUT_DIR = Path("data/pyscf_baseline/results")
DEFAULT_RUN_NAME = "h2o_original_rhf_openmx_like"

ORBITAL_TO_L = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
    "h": 5,
    "i": 6,
    "k": 7,
    "l": 8,
    "m": 9,
}
L_TO_ORBITAL = {value: key for key, value in ORBITAL_TO_L.items()}
# Conservative shell templates for a cheap, stable baseline.
# These are not OpenMX basis coefficients; only shell counts are matched.
BASE_EXPONENT_BY_L = {
    0: 1.80,  # s
    1: 1.20,  # p
    2: 0.80,  # d
    3: 0.60,  # f
    4: 0.45,  # g
    5: 0.35,  # h
}
EXPONENT_RATIO = 2.6
ELEMENTS_ORDER = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
)
ATOMIC_NUMBERS = {sym: (idx + 1) for idx, sym in enumerate(ELEMENTS_ORDER)}
HEURISTIC_ONSITE_BY_L = {
    0: -0.45,  # s
    1: -0.28,  # p
    2: -0.18,  # d
    3: -0.12,  # f
    4: -0.09,  # g
    5: -0.07,  # h
}


def _resolve_path(path: Path) -> Path:
    """Resolve relative paths from repository root."""
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _to_scalar(value: Any) -> float | None:
    """Convert scalar tensor-like values to a Python float."""
    if value is None:
        return None
    try:
        if hasattr(value, "numel") and int(value.numel()) == 0:
            return None
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return None


def _to_list(value: Any) -> Any:
    """Convert tensor/ndarray values to JSON-serializable nested lists."""
    if value is None:
        return None
    try:
        if hasattr(value, "numel") and int(value.numel()) == 0:
            return None
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value
    except Exception:
        return None


def _homo_lumo(
    mo_energy: np.ndarray, mo_occ: np.ndarray
) -> tuple[float | None, float | None]:
    """Return HOMO/LUMO in Hartree from orbital energies/occupations."""
    occ_mask = mo_occ > 1e-8
    vir_mask = ~occ_mask

    occupied = mo_energy[occ_mask]
    virtual = mo_energy[vir_mask]

    homo = float(np.max(occupied)) if occupied.size else None
    lumo = float(np.min(virtual)) if virtual.size else None
    return homo, lumo


def _parse_compact_orbital_spec(spec: str) -> dict[int, int]:
    """Parse compact orbital specs like ``3s2p2d`` into ``{l: count}``."""
    cleaned = spec.strip().lower().replace(" ", "").replace("+", "").replace(",", "")
    if not cleaned:
        raise ValueError("Empty orbital spec")

    counts: dict[int, int] = {}
    pos = 0
    for match in re.finditer(r"(\d+)([spdfghiklm])", cleaned):
        if match.start() != pos:
            raise ValueError(f"Invalid orbital spec: '{spec}'")
        n = int(match.group(1))
        l = ORBITAL_TO_L[match.group(2)]
        counts[l] = counts.get(l, 0) + n
        pos = match.end()

    if pos != len(cleaned):
        raise ValueError(f"Invalid orbital spec: '{spec}'")
    return counts


def _shell_counts_by_element(orbital_set: dict[str, str]) -> dict[str, dict[str, int]]:
    """Return element -> orbital letter -> radial shell count."""
    ret: dict[str, dict[str, int]] = {}
    for element, spec in sorted(orbital_set.items()):
        l_counts = _parse_compact_orbital_spec(spec)
        ret[element] = {L_TO_ORBITAL[l]: int(n) for l, n in sorted(l_counts.items())}
    return ret


def _ao_dims_from_orbital_set(
    orbital_set: dict[str, str], atoms: list[str]
) -> tuple[int, dict[str, int]]:
    """Return total AO count and per-element AO dimensions."""
    per_element_dim: dict[str, int] = {}
    for element, spec in orbital_set.items():
        l_counts = _parse_compact_orbital_spec(spec)
        dim = 0
        for l, n in l_counts.items():
            dim += int(n) * (2 * int(l) + 1)
        per_element_dim[element] = dim

    total = 0
    for atom in atoms:
        if atom not in per_element_dim:
            raise KeyError(f"Atom '{atom}' missing in orbital_set")
        total += per_element_dim[atom]

    return total, per_element_dim


def _even_tempered_exponents(l: int, n: int) -> list[float]:
    """Generate n exponents (tight -> diffuse) for angular momentum l."""
    if n <= 0:
        return []
    base = BASE_EXPONENT_BY_L.get(l, max(0.20, 1.80 / (l + 1.0)))
    return [float(base * (EXPONENT_RATIO ** (n - 1 - i))) for i in range(n)]


def _build_openmx_like_basis(orbital_set: dict[str, str]) -> dict[str, list[list[Any]]]:
    """Build a custom PySCF basis matching OpenMX shell counts."""
    basis: dict[str, list[list[Any]]] = {}
    for element, spec in sorted(orbital_set.items()):
        l_counts = _parse_compact_orbital_spec(spec)
        shells: list[list[Any]] = []
        for l, n in sorted(l_counts.items()):
            for exponent in _even_tempered_exponents(l, n):
                # One primitive per radial shell (cheap baseline).
                shells.append([l, [exponent, 1.0]])
        basis[element] = shells
    return basis


def _translation_shifts_for_kmesh(
    kmesh: Sequence[int], wrap_around: bool = True
) -> np.ndarray:
    """Return integer translation vectors in the same order as PySCF k2gamma."""
    if len(kmesh) != 3:
        raise ValueError(f"kmesh must have 3 entries, got: {kmesh}")
    kx, ky, kz = (int(kmesh[0]), int(kmesh[1]), int(kmesh[2]))
    if min(kx, ky, kz) <= 0:
        raise ValueError(f"kmesh entries must be positive, got: {kmesh}")

    ax = np.arange(kx, dtype=np.int64)
    ay = np.arange(ky, dtype=np.int64)
    az = np.arange(kz, dtype=np.int64)
    if wrap_around:
        ax[(kx + 1) // 2 :] -= kx
        ay[(ky + 1) // 2 :] -= ky
        az[(kz + 1) // 2 :] -= kz

    mx, my, mz = np.meshgrid(ax, ay, az, indexing="ij")
    return np.stack([mx.ravel(), my.ravel(), mz.ravel()], axis=1)


def _k_to_shifted_realspace(
    cell: Any,
    kpts: np.ndarray,
    kmesh: Sequence[int],
    mats_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform k-point AO matrices to shift-resolved real-space matrices.

    Returns
    -------
    shifts
        Integer shifts ``(sx, sy, sz)`` with shape ``(Nshift, 3)``.
    mats_shift
        Real-space matrices with shape ``(Nshift, nao, nao)``.
    """
    from pyscf.pbc.tools.k2gamma import get_phase

    mats_k = np.asarray(mats_k)
    if mats_k.ndim != 3:
        raise ValueError(f"Expected k-space matrices (Nk,nao,nao), got {mats_k.shape}")

    shifts = _translation_shifts_for_kmesh(kmesh, wrap_around=True)
    _, phase = get_phase(cell, kpts, kmesh=kmesh, wrap_around=True)

    if phase.shape[0] != shifts.shape[0]:
        raise RuntimeError(
            f"Phase/shift mismatch: phase rows={phase.shape[0]}, shifts={shifts.shape[0]}"
        )
    if phase.shape[1] != mats_k.shape[0]:
        raise RuntimeError(
            f"Phase/kpoint mismatch: phase cols={phase.shape[1]}, Nk={mats_k.shape[0]}"
        )

    mats_rs = np.einsum("Rk,kij,Sk->RiSj", phase, mats_k, phase.conj())

    zero_mask = np.all(shifts == np.array([0, 0, 0], dtype=np.int64), axis=1)
    if not np.any(zero_mask):
        raise RuntimeError("Could not locate zero shift in translation vectors")
    origin_idx = int(np.flatnonzero(zero_mask)[0])

    # Extract couplings from home cell to destination shifts.
    mats_shift = np.swapaxes(mats_rs[origin_idx], 0, 1)  # (Nshift, nao, nao)

    imag_max = float(np.max(np.abs(np.imag(mats_shift))))
    if imag_max > 1e-8:
        raise RuntimeError(
            f"Shift-resolved matrices have significant imaginary part ({imag_max:.3e})."
        )
    mats_shift = np.real(mats_shift)
    return shifts, mats_shift


def _collapse_spin_resolved_k_mats(
    mats_k: np.ndarray,
    *,
    name: str,
    combine_mode: str,
) -> np.ndarray:
    """Normalize k-space matrix arrays to shape ``(Nk, nao, nao)``.

    For unrestricted references, combine alpha/beta channels:
    - ``combine_mode='sum'`` for total density
    - ``combine_mode='mean'`` for a spin-averaged one-body operator
    """
    arr = np.asarray(mats_k)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[0] == 2:
        if combine_mode == "sum":
            return np.asarray(arr[0] + arr[1])
        if combine_mode == "mean":
            return np.asarray(0.5 * (arr[0] + arr[1]))
        raise ValueError(f"Unsupported combine_mode={combine_mode!r} for {name}")
    raise ValueError(
        f"{name} has unsupported shape {arr.shape}; expected (Nk,nao,nao) or (2,Nk,nao,nao)"
    )


def _atomic_number(element: str) -> int:
    """Return atomic number for common elements, with a safe fallback."""
    return int(ATOMIC_NUMBERS.get(str(element), 8))


def _normalized_orbital_vector(dim: int) -> np.ndarray:
    """Deterministic per-atom AO profile vector with unit norm."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    vec = np.linspace(1.0, 2.0, dim, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.ones(dim, dtype=np.float32) / np.sqrt(float(dim))
    return vec / norm


def _heuristic_atom_diagonal(element: str, orbital_spec: str) -> np.ndarray:
    """Build diagonal onsite AO energies for one atom from shell counts."""
    z = float(_atomic_number(element))
    z_scale = 1.0 + 0.08 * max(0.0, z - 1.0)
    l_counts = _parse_compact_orbital_spec(orbital_spec)
    diag_values: list[float] = []
    for l, n in sorted(l_counts.items()):
        base = float(HEURISTIC_ONSITE_BY_L.get(l, -0.06 / (l + 1.0)))
        for shell_idx in range(int(n)):
            eps = base * z_scale * (1.0 - 0.04 * shell_idx)
            diag_values.extend([eps] * (2 * int(l) + 1))
    return np.asarray(diag_values, dtype=np.float32)


def _heuristic_nuclear_repulsion(
    elements: Sequence[str], positions: np.ndarray
) -> float:
    """Very cheap Coulomb-like pair repulsion estimate in Hartree units."""
    n = len(elements)
    enuc = 0.0
    for i in range(n):
        zi = float(_atomic_number(elements[i]))
        ri = positions[i]
        for j in range(i + 1, n):
            zj = float(_atomic_number(elements[j]))
            rj = positions[j]
            dist = float(np.linalg.norm(rj - ri))
            if dist < 1e-8:
                continue
            enuc += (zi * zj) / dist
    # Scale down to keep values in a similar ballpark to one-electron terms.
    return 0.05 * enuc


def _kpts_from_kmesh(kmesh: Sequence[int]) -> np.ndarray:
    """Return a cheap fractional k-point grid representation."""
    kx, ky, kz = (int(kmesh[0]), int(kmesh[1]), int(kmesh[2]))
    ax = np.arange(kx, dtype=np.float32) / float(kx)
    ay = np.arange(ky, dtype=np.float32) / float(ky)
    az = np.arange(kz, dtype=np.float32) / float(kz)
    mx, my, mz = np.meshgrid(ax, ay, az, indexing="ij")
    return np.stack([mx.ravel(), my.ravel(), mz.ravel()], axis=1)


def _build_heuristic_pbc_matrices(
    *,
    elements: Sequence[str],
    positions: np.ndarray,
    box: np.ndarray,
    orbital_set: dict[str, str],
    charge: int,
    spin: int,
    kmesh: Sequence[int],
) -> dict[str, Any]:
    """Build a fast geometry-based PBC baseline without SCF."""
    expected_dim, per_element_dim = _ao_dims_from_orbital_set(
        orbital_set, list(elements)
    )
    dims = [int(per_element_dim[e]) for e in elements]
    offsets = np.zeros(len(dims), dtype=np.int64)
    if len(dims) > 1:
        offsets[1:] = np.cumsum(np.asarray(dims[:-1], dtype=np.int64))

    shifts = _translation_shifts_for_kmesh(kmesh, wrap_around=True).astype(np.int64)
    nshift = int(shifts.shape[0])
    nao = int(expected_dim)

    h_shift = np.zeros((nshift, nao, nao), dtype=np.float32)
    s_shift = np.zeros((nshift, nao, nao), dtype=np.float32)
    d_shift = np.zeros((nshift, nao, nao), dtype=np.float32)

    zero_mask = np.all(shifts == np.array([0, 0, 0], dtype=np.int64), axis=1)
    if not np.any(zero_mask):
        raise RuntimeError("No zero shift found in heuristic shift grid.")
    zero_idx = int(np.flatnonzero(zero_mask)[0])

    ao_profiles = [_normalized_orbital_vector(dim) for dim in dims]
    onsite_diags = [_heuristic_atom_diagonal(el, orbital_set[el]) for el in elements]

    z_list = np.asarray([_atomic_number(el) for el in elements], dtype=np.float32)
    total_z = float(np.sum(z_list))
    total_electrons = max(0, int(round(total_z - float(charge))))

    # Onsite terms at zero shift.
    for i, (dim, off) in enumerate(zip(dims, offsets)):
        diag = onsite_diags[i]
        if diag.shape[0] != dim:
            raise RuntimeError(
                f"Heuristic onsite diag mismatch for atom {i}: expected {dim}, got {diag.shape[0]}"
            )
        sl = slice(int(off), int(off + dim))
        h_shift[zero_idx, sl, sl] += np.diag(diag)
        s_shift[zero_idx, sl, sl] += np.eye(dim, dtype=np.float32)

        if total_z > 0.0:
            atom_e = float(total_electrons) * float(z_list[i] / total_z)
        else:
            atom_e = float(total_electrons) / max(1, len(elements))
        occ_per_ao = atom_e / float(dim)
        d_shift[zero_idx, sl, sl] += np.eye(dim, dtype=np.float32) * np.float32(
            occ_per_ao
        )

    # Pair couplings for all shifts.
    for s_idx, shift in enumerate(shifts):
        shift_cart = np.asarray(shift, dtype=np.float32) @ box.astype(np.float32)
        for i, (di, oi) in enumerate(zip(dims, offsets)):
            vi = ao_profiles[i]
            ri = positions[i]
            sli = slice(int(oi), int(oi + di))
            for j, (dj, oj) in enumerate(zip(dims, offsets)):
                if s_idx == zero_idx and i == j:
                    continue

                vj = ao_profiles[j]
                rj = positions[j] + shift_cart
                slj = slice(int(oj), int(oj + dj))

                dist = float(np.linalg.norm(rj - ri))
                if dist < 1e-8:
                    continue
                shape = np.outer(vi, vj)

                if i == j:
                    h_amp = -0.015 * np.exp(-2.2 * dist)
                    s_amp = 0.006 * np.exp(-2.8 * dist)
                    d_amp = 0.0
                else:
                    h_amp = -0.09 * np.exp(-1.35 * dist)
                    s_amp = 0.05 * np.exp(-1.7 * dist)
                    d_amp = (
                        0.02 * np.exp(-1.9 * dist)
                        if s_idx == zero_idx
                        else 0.004 * np.exp(-1.9 * dist)
                    )

                if abs(h_amp) > 1e-9:
                    h_shift[s_idx, sli, slj] += np.float32(h_amp) * shape
                if abs(s_amp) > 1e-9:
                    s_shift[s_idx, sli, slj] += np.float32(s_amp) * shape
                if abs(d_amp) > 1e-9:
                    d_shift[s_idx, sli, slj] += np.float32(d_amp) * shape

    # Keep home-cell operators explicitly symmetric.
    h_shift[zero_idx] = 0.5 * (h_shift[zero_idx] + h_shift[zero_idx].T)
    s_shift[zero_idx] = 0.5 * (s_shift[zero_idx] + s_shift[zero_idx].T)
    d_shift[zero_idx] = 0.5 * (d_shift[zero_idx] + d_shift[zero_idx].T)
    s_shift[zero_idx] += np.eye(nao, dtype=np.float32) * np.float32(1e-6)

    h0 = np.asarray(h_shift[zero_idx], dtype=np.float32)
    s0 = np.asarray(s_shift[zero_idx], dtype=np.float32)
    d0 = np.asarray(d_shift[zero_idx], dtype=np.float32)

    eigvals, eigvecs = np.linalg.eigh(h0.astype(np.float64))
    mo_energy = eigvals.astype(np.float32)
    mo_coeff = eigvecs.astype(np.float32)

    if spin == 0:
        mo_occ = np.zeros(nao, dtype=np.float32)
        ndoubly = min(total_electrons // 2, nao)
        mo_occ[:ndoubly] = 2.0
        if (total_electrons % 2 == 1) and (ndoubly < nao):
            mo_occ[ndoubly] = 1.0
    else:
        nalpha = max(0, min(nao, (total_electrons + int(spin)) // 2))
        nbeta = max(0, min(nao, total_electrons - nalpha))
        mo_occ_a = np.zeros(nao, dtype=np.float32)
        mo_occ_b = np.zeros(nao, dtype=np.float32)
        mo_occ_a[:nalpha] = 1.0
        mo_occ_b[:nbeta] = 1.0
        mo_occ = np.stack([mo_occ_a, mo_occ_b], axis=0)

    enuc = _heuristic_nuclear_repulsion(elements, positions)
    e1 = float(np.sum(d0 * h0))
    total_energy_h = float(enuc + e1)

    return {
        "actual_ao_dim": nao,
        "shifts": shifts,
        "hamiltonian_shifted": h_shift,
        "hcore_shifted": h_shift.copy(),
        "overlap_shifted": s_shift,
        "density_shifted": d_shift,
        "hamiltonian_ao": h0,
        "hcore_ao": h0.copy(),
        "fock_ao": h0.copy(),
        "overlap_ao": s0,
        "dm_ao": d0,
        "mo_coeff": mo_coeff,
        "mo_occ": mo_occ,
        "mo_energy_hartree": mo_energy,
        "kpts_abs": _kpts_from_kmesh(kmesh),
        "total_energy_hartree": total_energy_h,
        "nuclear_repulsion_hartree": float(enuc),
        "num_electrons": int(total_electrons),
        "periodic_meta": {
            "kmesh": [int(v) for v in kmesh],
            "num_kpts": int(np.prod(np.asarray(kmesh, dtype=np.int64))),
            "num_shifts": int(nshift),
            "num_nonzero_shifts": int(np.sum(np.any(shifts != 0, axis=1))),
            "zero_shift_index": int(zero_idx),
        },
    }


def _write_xyz(
    path: Path, atom_spec: list[tuple[str, tuple[float, float, float]]], comment: str
) -> None:
    """Write geometry to XYZ for quick inspection/reuse."""
    lines = [str(len(atom_spec)), comment]
    for element, (x, y, z) in atom_spec:
        lines.append(f"{element:2s} {x: .10f} {y: .10f} {z: .10f}")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--info-path",
        type=Path,
        default=DEFAULT_INFO_PATH,
        help="OpenMX .info.out snapshot path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON/NPZ/XYZ outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--basis-mode",
        choices=("openmx_like", "library"),
        default="openmx_like",
        help="Basis construction mode. openmx_like matches OpenMX shell counts.",
    )
    parser.add_argument(
        "--method",
        choices=("heuristic", "rhf", "rks"),
        default="heuristic",
        help="Method family. heuristic is geometry-based and avoids SCF.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="sto-3g",
        help="PySCF basis name when --basis-mode=library.",
    )
    parser.add_argument(
        "--xc",
        type=str,
        default="lda,vwn",
        help="DFT functional for --method rks.",
    )
    parser.add_argument("--charge", type=int, default=0, help="Molecular charge.")
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="2S value expected by PySCF (0 for closed-shell singlet).",
    )
    parser.add_argument(
        "--max-cycle",
        type=int,
        default=100,
        help="Maximum SCF iterations.",
    )
    parser.add_argument(
        "--conv-tol",
        type=float,
        default=1e-8,
        help="SCF energy convergence threshold in Hartree.",
    )
    parser.add_argument(
        "--grid-level",
        type=int,
        default=0,
        help="Numerical integration grid level for RKS/UKS.",
    )
    parser.add_argument(
        "--pyscf-verbose",
        type=int,
        default=4,
        help="PySCF verbosity level.",
    )
    parser.add_argument(
        "--kmesh",
        type=int,
        nargs=3,
        default=(1, 1, 1),
        metavar=("KX", "KY", "KZ"),
        help="k-point mesh for periodic mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse snapshot and write metadata/XYZ without running PySCF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info_path = _resolve_path(args.info_path)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{args.run_name}.json"
    npz_path = output_dir / f"{args.run_name}.npz"
    xyz_path = output_dir / f"{args.run_name}.xyz"
    ham_npy_path = output_dir / f"{args.run_name}.hamiltonian_ao.npy"
    overlap_npy_path = output_dir / f"{args.run_name}.overlap_ao.npy"
    density_npy_path = output_dir / f"{args.run_name}.density_ao.npy"
    shifts_npy_path = output_dir / f"{args.run_name}.shifts.npy"
    ham_shift_npy_path = output_dir / f"{args.run_name}.hamiltonian_shifted.npy"
    overlap_shift_npy_path = output_dir / f"{args.run_name}.overlap_shifted.npy"
    density_shift_npy_path = output_dir / f"{args.run_name}.density_shifted.npy"

    output_targets = (
        [json_path, xyz_path]
        if args.dry_run
        else [
            json_path,
            npz_path,
            xyz_path,
            ham_npy_path,
            overlap_npy_path,
            density_npy_path,
            shifts_npy_path,
            ham_shift_npy_path,
            overlap_shift_npy_path,
            density_shift_npy_path,
        ]
    )
    if not args.overwrite:
        existing = [p for p in output_targets if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Output file(s) already exist: {names}. Use --overwrite to replace."
            )

    if not info_path.exists():
        raise FileNotFoundError(f"Missing info file: {info_path}")

    info = parse_info_out(info_path)
    if not info.elements:
        raise RuntimeError(
            f"No atoms parsed from {info_path}. Expected <coordinates.forces> block."
        )
    if info.positions.numel() == 0:
        raise RuntimeError(f"No Cartesian positions parsed from {info_path}.")
    if info.box.numel() == 0:
        raise RuntimeError(
            "PBC baseline requires a lattice (`box`) parsed from .info.out."
        )

    positions = info.positions.detach().cpu().numpy()
    if positions.shape[0] != len(info.elements):
        raise RuntimeError(
            f"Element/position mismatch: {len(info.elements)} symbols vs {positions.shape[0]} positions."
        )

    atom_spec: list[tuple[str, tuple[float, float, float]]] = []
    atom_rows: list[dict[str, Any]] = []
    for idx, (element, xyz) in enumerate(zip(info.elements, positions), start=1):
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        atom_spec.append((element, (x, y, z)))
        atom_rows.append(
            {
                "index": idx,
                "element": element,
                "x_angstrom": x,
                "y_angstrom": y,
                "z_angstrom": z,
            }
        )

    _write_xyz(
        xyz_path,
        atom_spec,
        comment=f"run_name={args.run_name}; source={info_path}",
    )

    energies_h = {
        key: float(value.item()) if hasattr(value, "item") else float(value)
        for key, value in sorted(info.energies.items())
    }
    utot_h = energies_h.get("Utot")
    fermi_h = _to_scalar(info.fermi_level)
    orbital_set = {key: value for key, value in sorted(info.orbital_set.items())}
    shell_counts = _shell_counts_by_element(orbital_set)
    expected_ao_dim, per_element_ao_dim = _ao_dims_from_orbital_set(
        orbital_set, info.elements
    )
    openmx_num_states = (
        int(info.eigenvalues.shape[0])
        if hasattr(info.eigenvalues, "shape") and info.eigenvalues.numel()
        else None
    )

    result: dict[str, Any] = {
        "run_name": args.run_name,
        "dry_run": bool(args.dry_run),
        "snapshot": {
            "info_path": str(info_path),
            "num_atoms": len(atom_spec),
            "atoms": atom_rows,
            "box_angstrom": _to_list(info.box),
        },
        "openmx_reference": {
            "energies_hartree": energies_h,
            "utot_hartree": utot_h,
            "fermi_level_hartree": fermi_h,
            "orbital_set": orbital_set,
            "openmx_num_states": openmx_num_states,
        },
        "settings": {
            "method": args.method,
            "basis_mode": args.basis_mode,
            "basis_library": args.basis if args.basis_mode == "library" else None,
            "periodic": True,
            "kmesh": [int(v) for v in args.kmesh],
            "orbital_set": orbital_set,
            "shell_counts": shell_counts,
            "expected_ao_dim": expected_ao_dim,
            "per_element_ao_dim": per_element_ao_dim,
            "charge": args.charge,
            "spin": args.spin,
            "xc": args.xc if args.method == "rks" else None,
            "max_cycle": args.max_cycle,
            "conv_tol": args.conv_tol,
            "grid_level": args.grid_level if args.method == "rks" else None,
            "pyscf_verbose": args.pyscf_verbose,
        },
        "artifacts": {
            "xyz_path": str(xyz_path),
            "hamiltonian_npy_path": None if args.dry_run else str(ham_npy_path),
            "overlap_npy_path": None if args.dry_run else str(overlap_npy_path),
            "density_npy_path": None if args.dry_run else str(density_npy_path),
            "shifts_npy_path": None if args.dry_run else str(shifts_npy_path),
            "hamiltonian_shifted_npy_path": (
                None if args.dry_run else str(ham_shift_npy_path)
            ),
            "overlap_shifted_npy_path": (
                None if args.dry_run else str(overlap_shift_npy_path)
            ),
            "density_shifted_npy_path": (
                None if args.dry_run else str(density_shift_npy_path)
            ),
            "npz_path": None if args.dry_run else str(npz_path),
            "contains_shift_resolved": False if args.dry_run else True,
        },
    }

    if args.dry_run:
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[pyscf-baseline] dry run complete; wrote {json_path}")
        return

    kmesh = tuple(int(v) for v in args.kmesh)
    if args.method == "heuristic":
        heur = _build_heuristic_pbc_matrices(
            elements=info.elements,
            positions=positions.astype(np.float32),
            box=info.box.detach().cpu().numpy().astype(np.float32),
            orbital_set=orbital_set,
            charge=args.charge,
            spin=args.spin,
            kmesh=kmesh,
        )
        actual_ao_dim = int(heur["actual_ao_dim"])
        if actual_ao_dim != expected_ao_dim:
            raise RuntimeError(
                "AO dimension mismatch in heuristic path. "
                f"Expected {expected_ao_dim} from orbital_set={orbital_set}, got {actual_ao_dim}."
            )

        shifts = np.asarray(heur["shifts"], dtype=np.int64)
        hamiltonian_shifted = np.asarray(heur["hamiltonian_shifted"], dtype=np.float32)
        hcore_shifted = np.asarray(heur["hcore_shifted"], dtype=np.float32)
        overlap_shifted = np.asarray(heur["overlap_shifted"], dtype=np.float32)
        density_shifted = np.asarray(heur["density_shifted"], dtype=np.float32)

        hamiltonian_ao = np.asarray(heur["hamiltonian_ao"], dtype=np.float32)
        hcore = np.asarray(heur["hcore_ao"], dtype=np.float32)
        overlap = np.asarray(heur["overlap_ao"], dtype=np.float32)
        dm = np.asarray(heur["dm_ao"], dtype=np.float32)
        fock = np.asarray(heur["fock_ao"], dtype=np.float32)

        mo_coeff = np.asarray(heur["mo_coeff"], dtype=np.float32)
        mo_occ = np.asarray(heur["mo_occ"], dtype=np.float32)
        mo_energy = np.asarray(heur["mo_energy_hartree"], dtype=np.float32)
        periodic_meta = dict(heur["periodic_meta"])
        energy_h = float(heur["total_energy_hartree"])
        nuclear_repulsion_h = float(heur["nuclear_repulsion_hartree"])
        num_electrons = int(heur["num_electrons"])
        kpts_abs = np.asarray(heur["kpts_abs"], dtype=np.float32)
        converged = True
        method_label = "HeuristicPBC"
        basis_label = "heuristic_shell_count_matched"
    else:
        try:
            from pyscf.pbc import dft as pbc_dft
            from pyscf.pbc import gto as pbc_gto
            from pyscf.pbc import scf as pbc_scf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PySCF is not installed. Install it (for example `pip install pyscf`) "
                "or run with --method heuristic."
            ) from exc

        if args.basis_mode == "openmx_like":
            basis_spec = _build_openmx_like_basis(orbital_set)
            basis_label = "openmx_like_shell_count_matched"
        else:
            basis_spec = args.basis
            basis_label = args.basis

        cell = pbc_gto.Cell()
        cell.atom = atom_spec
        cell.a = info.box.detach().cpu().numpy().tolist()
        cell.unit = "Angstrom"
        cell.basis = basis_spec
        cell.charge = args.charge
        cell.spin = args.spin
        cell.verbose = args.pyscf_verbose
        cell.build()

        actual_ao_dim = int(cell.nao_nr())
        if actual_ao_dim != expected_ao_dim:
            raise RuntimeError(
                "AO dimension mismatch. "
                f"Expected {expected_ao_dim} from orbital_set={orbital_set}, "
                f"but PySCF built {actual_ao_dim} AOs with basis_mode={args.basis_mode}."
            )

        kpts = cell.make_kpts(kmesh, wrap_around=True)
        if args.method == "rhf":
            mf = (
                pbc_scf.KRHF(cell, kpts=kpts)
                if args.spin == 0
                else pbc_scf.KUHF(cell, kpts=kpts)
            )
        elif args.method == "rks":
            mf = (
                pbc_dft.KRKS(cell, kpts=kpts)
                if args.spin == 0
                else pbc_dft.KUKS(cell, kpts=kpts)
            )
            mf.xc = args.xc
            mf.grids.level = args.grid_level
        else:
            raise ValueError(f"Unsupported method: {args.method}")

        mf.max_cycle = args.max_cycle
        mf.conv_tol = args.conv_tol

        energy_h = float(mf.kernel())
        mo_coeff = np.asarray(mf.mo_coeff)
        mo_occ = np.asarray(mf.mo_occ)
        mo_energy = np.asarray(mf.mo_energy)

        hcore_k = np.asarray(mf.get_hcore(cell, kpts=kpts))
        overlap_k = np.asarray(mf.get_ovlp(cell, kpts=kpts))
        dm_k_raw = np.asarray(mf.make_rdm1())
        dm_k = _collapse_spin_resolved_k_mats(dm_k_raw, name="dm_k", combine_mode="sum")
        fock_k_raw = np.asarray(mf.get_fock(h1e=hcore_k, s1e=overlap_k, dm=dm_k_raw))
        fock_k = _collapse_spin_resolved_k_mats(
            fock_k_raw, name="fock_k", combine_mode="mean"
        )

        shifts, hamiltonian_shifted = _k_to_shifted_realspace(cell, kpts, kmesh, fock_k)
        _, hcore_shifted = _k_to_shifted_realspace(cell, kpts, kmesh, hcore_k)
        _, overlap_shifted = _k_to_shifted_realspace(cell, kpts, kmesh, overlap_k)
        _, density_shifted = _k_to_shifted_realspace(cell, kpts, kmesh, dm_k)

        zero_mask = np.all(shifts == np.array([0, 0, 0], dtype=np.int64), axis=1)
        if not np.any(zero_mask):
            raise RuntimeError("No zero shift found in periodic shift grid.")
        zero_idx = int(np.flatnonzero(zero_mask)[0])

        hamiltonian_ao = np.asarray(hamiltonian_shifted[zero_idx])
        hcore = np.asarray(hcore_shifted[zero_idx])
        overlap = np.asarray(overlap_shifted[zero_idx])
        dm = np.asarray(density_shifted[zero_idx])
        fock = np.asarray(hamiltonian_ao)

        periodic_meta = {
            "kmesh": [int(v) for v in kmesh],
            "num_kpts": int(len(kpts)),
            "num_shifts": int(shifts.shape[0]),
            "num_nonzero_shifts": int(np.sum(np.any(shifts != 0, axis=1))),
            "zero_shift_index": zero_idx,
        }

        nuclear_repulsion_h = float(cell.energy_nuc())
        num_electrons = int(cell.nelectron)
        kpts_abs = np.asarray(kpts)
        converged = bool(mf.converged)
        method_label = mf.__class__.__name__

    npz_payload = {
        "hamiltonian_ao": hamiltonian_ao,
        "fock_ao": fock,
        "hcore_ao": hcore,
        "dm_ao": dm,
        "overlap_ao": overlap,
        "hamiltonian_shifted": np.asarray(hamiltonian_shifted),
        "hcore_shifted": np.asarray(hcore_shifted),
        "overlap_shifted": np.asarray(overlap_shifted),
        "density_shifted": np.asarray(density_shifted),
        "shifts": shifts.astype(np.int64),
        "kmesh": np.asarray(kmesh, dtype=np.int64),
        "kpts_abs": np.asarray(kpts_abs),
        "mo_coeff": mo_coeff,
        "mo_occ": mo_occ,
        "mo_energy_hartree": mo_energy,
        "positions_angstrom": positions,
    }

    mo_occ_flat = np.asarray(mo_occ).reshape(-1)
    mo_energy_flat = np.asarray(mo_energy).reshape(-1)
    homo_h, lumo_h = _homo_lumo(mo_energy_flat, mo_occ_flat)

    np.save(ham_npy_path, hamiltonian_ao)
    np.save(overlap_npy_path, overlap)
    np.save(density_npy_path, dm)
    np.save(shifts_npy_path, shifts.astype(np.int64))
    np.save(ham_shift_npy_path, hamiltonian_shifted)
    np.save(overlap_shift_npy_path, overlap_shifted)
    np.save(density_shift_npy_path, density_shifted)
    np.savez_compressed(npz_path, **npz_payload)

    delta_h = None if utot_h is None else float(energy_h - utot_h)
    result["pyscf"] = {
        "converged": bool(converged),
        "method_label": method_label,
        "basis_label": basis_label,
        "ao_dim": actual_ao_dim,
        "total_energy_hartree": energy_h,
        "total_energy_ev": float(energy_h * HARTREE_TO_EV),
        "nuclear_repulsion_hartree": float(nuclear_repulsion_h),
        "num_electrons": int(num_electrons),
        "homo_hartree": homo_h,
        "lumo_hartree": lumo_h,
        "homo_lumo_gap_hartree": (
            None if (homo_h is None or lumo_h is None) else float(lumo_h - homo_h)
        ),
        "delta_vs_openmx_utot_hartree": delta_h,
        "delta_vs_openmx_utot_ev": (
            None if delta_h is None else float(delta_h * HARTREE_TO_EV)
        ),
    }
    result["pyscf"]["periodic"] = periodic_meta

    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"[pyscf-baseline] wrote {json_path}")
    print(f"[pyscf-baseline] wrote {ham_npy_path}")
    print(f"[pyscf-baseline] wrote {overlap_npy_path}")
    print(f"[pyscf-baseline] wrote {density_npy_path}")
    print(f"[pyscf-baseline] wrote {shifts_npy_path}")
    print(f"[pyscf-baseline] wrote {ham_shift_npy_path}")
    print(f"[pyscf-baseline] wrote {overlap_shift_npy_path}")
    print(f"[pyscf-baseline] wrote {density_shift_npy_path}")
    print(f"[pyscf-baseline] wrote {npz_path}")
    print(
        f"[pyscf-baseline] converged={converged} method={method_label} E={energy_h:.12f} Ha"
    )


if __name__ == "__main__":
    main()
