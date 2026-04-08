#!/usr/bin/env python3
"""Production periodic PySCF KRKS calculation with shift-resolved matrices.

This script is intentionally strict:
- only periodic KRKS (no RHF/heuristic branches)
- no fallback/workaround paths
- aborts immediately on non-convergence or unavailable production properties
- exports energy-like quantities in eV and geometric quantities in Angstrom units
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
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
STRESS_HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3 = HARTREE_TO_EV / (BOHR_TO_ANGSTROM**3)
DEFAULT_INFO_PATH = Path("data/small/H2O/original/H2O.info.out")
DEFAULT_OUTPUT_DIR = Path("data/pyscf_baseline/results")
DEFAULT_RUN_NAME = "h2o_original_rks_openmx_like_prod_k444"

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
BASE_EXPONENT_BY_L = {
    0: 1.80,  # s
    1: 1.20,  # p
    2: 0.80,  # d
    3: 0.60,  # f
    4: 0.45,  # g
    5: 0.35,  # h
}
EXPONENT_RATIO = 2.6


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _parse_compact_orbital_spec(spec: str) -> dict[int, int]:
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


def _ao_dims_from_orbital_set(
    orbital_set: dict[str, str], atoms: Sequence[str]
) -> tuple[int, dict[str, int]]:
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
    if n <= 0:
        return []
    base = BASE_EXPONENT_BY_L.get(l, max(0.20, 1.80 / (l + 1.0)))
    return [float(base * (EXPONENT_RATIO ** (n - 1 - i))) for i in range(n)]


def _build_openmx_like_basis(orbital_set: dict[str, str]) -> dict[str, list[list[Any]]]:
    basis: dict[str, list[list[Any]]] = {}
    for element, spec in sorted(orbital_set.items()):
        l_counts = _parse_compact_orbital_spec(spec)
        shells: list[list[Any]] = []
        for l, n in sorted(l_counts.items()):
            for exponent in _even_tempered_exponents(l, n):
                shells.append([l, [exponent, 1.0]])
        basis[element] = shells
    return basis


def _translation_shifts_for_kmesh(
    kmesh: Sequence[int], wrap_around: bool = True
) -> np.ndarray:
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
    mats_shift = np.swapaxes(mats_rs[origin_idx], 0, 1)  # (Nshift, nao, nao)

    imag_max = float(np.max(np.abs(np.imag(mats_shift))))
    if imag_max > 1e-8:
        raise RuntimeError(
            f"Shift-resolved matrices have significant imaginary part ({imag_max:.3e})."
        )
    return shifts, np.real(mats_shift)


def _safe_trace_real(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.einsum("ij,ji->", np.asarray(a), np.asarray(b)).real)


def _write_xyz(
    path: Path, atom_spec: list[tuple[str, tuple[float, float, float]]], comment: str
) -> None:
    lines = [str(len(atom_spec)), comment]
    for element, (x, y, z) in atom_spec:
        lines.append(f"{element:2s} {x: .10f} {y: .10f} {z: .10f}")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info-path", type=Path, default=DEFAULT_INFO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--xc",
        type=str,
        default="pbe,pbe",
        help="Periodic KRKS functional.",
    )
    parser.add_argument(
        "--grid-level",
        type=int,
        default=3,
        help="PySCF numerical integration grid level.",
    )
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="Only spin=0 is supported by this strict production script.",
    )
    parser.add_argument("--max-cycle", type=int, default=200)
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--pyscf-verbose", type=int, default=4)
    parser.add_argument(
        "--kmesh",
        type=int,
        nargs=3,
        default=(4, 4, 4),
        metavar=("KX", "KY", "KZ"),
    )
    parser.add_argument(
        "--df-backend",
        choices=("gdf", "aftdf", "fft"),
        default="gdf",
        help="Density-fitting backend for production KRKS.",
    )
    parser.add_argument(
        "--df-auxbasis",
        type=str,
        default=None,
        help="Optional auxiliary basis used when --df-backend gdf.",
    )
    parser.add_argument("--pbc-precision", type=float, default=1e-8)
    parser.add_argument(
        "--ke-cutoff",
        type=float,
        default=None,
        help="Optional explicit PW kinetic cutoff for Cell (Ha).",
    )
    parser.add_argument("--max-memory-mb", type=int, default=1200000)
    parser.add_argument(
        "--require-forces-stress",
        action="store_true",
        help=(
            "Require KRKS forces and stress. "
            "If unavailable in this PySCF path, abort immediately."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.spin) != 0:
        raise ValueError("This production script supports only spin=0 KRKS.")

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

    output_targets = [
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
    if not args.overwrite:
        existing = [p for p in output_targets if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Output file(s) already exist: {names}. Use --overwrite to replace."
            )

    info = parse_info_out(info_path)
    if not info.elements:
        raise RuntimeError(f"No atoms parsed from {info_path}")
    if info.positions.numel() == 0:
        raise RuntimeError(f"No Cartesian positions parsed from {info_path}")
    if info.box.numel() == 0:
        raise RuntimeError("Periodic KRKS requires lattice vectors (box) in .info.out")

    positions_angstrom = info.positions.detach().cpu().numpy().astype(np.float64)
    box_angstrom = info.box.detach().cpu().numpy().astype(np.float64)

    atom_spec: list[tuple[str, tuple[float, float, float]]] = []
    atom_rows: list[dict[str, Any]] = []
    for idx, (element, xyz) in enumerate(
        zip(info.elements, positions_angstrom), start=1
    ):
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

    orbital_set = {key: value for key, value in sorted(info.orbital_set.items())}
    expected_ao_dim, _ = _ao_dims_from_orbital_set(orbital_set, info.elements)
    basis_spec = _build_openmx_like_basis(orbital_set)
    kmesh = tuple(int(v) for v in args.kmesh)

    try:
        from pyscf.pbc import dft as pbc_dft
        from pyscf.pbc import df as pbc_df
        from pyscf.pbc import gto as pbc_gto
    except ModuleNotFoundError as exc:
        raise RuntimeError("PySCF is required for this production script.") from exc

    cell = pbc_gto.Cell()
    cell.atom = atom_spec
    cell.a = box_angstrom.tolist()
    cell.unit = "Angstrom"
    cell.basis = basis_spec
    cell.charge = int(args.charge)
    cell.spin = int(args.spin)
    cell.verbose = int(args.pyscf_verbose)
    cell.precision = float(args.pbc_precision)
    if args.ke_cutoff is not None:
        cell.ke_cutoff = float(args.ke_cutoff)
    cell.build()

    actual_ao_dim = int(cell.nao_nr())
    if actual_ao_dim != expected_ao_dim:
        raise RuntimeError(
            "AO dimension mismatch. "
            f"Expected {expected_ao_dim} from orbital_set={orbital_set}, "
            f"but PySCF built {actual_ao_dim}."
        )

    kpts = cell.make_kpts(kmesh, wrap_around=True)
    mf = pbc_dft.KRKS(cell, kpts=kpts)
    mf.xc = args.xc
    mf.grids.level = int(args.grid_level)

    if args.df_backend == "gdf":
        with_df = pbc_df.GDF(cell, kpts=kpts)
        if args.df_auxbasis:
            with_df.auxbasis = args.df_auxbasis
        mf.with_df = with_df
    elif args.df_backend == "aftdf":
        mf.with_df = pbc_df.AFTDF(cell, kpts=kpts)
    elif args.df_backend == "fft":
        # Allowed for completeness, but may be memory intensive at dense k-meshes.
        pass
    else:
        raise ValueError(f"Unsupported df backend: {args.df_backend}")

    mf.max_memory = int(args.max_memory_mb)
    mf.max_cycle = int(args.max_cycle)
    mf.conv_tol = float(args.conv_tol)
    mf.verbose = int(args.pyscf_verbose)

    energy_hartree = float(mf.kernel())
    if not bool(mf.converged):
        raise RuntimeError("KRKS did not converge; aborting.")

    hcore_k = np.asarray(mf.get_hcore(cell, kpts=kpts))
    overlap_k = np.asarray(mf.get_ovlp(cell, kpts=kpts))
    dm_k = np.asarray(mf.make_rdm1())
    if dm_k.ndim != 3:
        raise RuntimeError(
            f"Unexpected dm_k shape for KRKS spin=0: got {dm_k.shape}, expected (Nk,nao,nao)."
        )
    veff_k = np.asarray(mf.get_veff(cell, dm_k, kpts=kpts))
    if veff_k.ndim != 3:
        raise RuntimeError(
            f"Unexpected veff_k shape for KRKS spin=0: got {veff_k.shape}, expected (Nk,nao,nao)."
        )
    hks_k = hcore_k + veff_k

    shifts, hamiltonian_shifted_h = _k_to_shifted_realspace(cell, kpts, kmesh, hks_k)
    _, hcore_shifted_h = _k_to_shifted_realspace(cell, kpts, kmesh, hcore_k)
    _, overlap_shifted = _k_to_shifted_realspace(cell, kpts, kmesh, overlap_k)
    _, density_shifted = _k_to_shifted_realspace(cell, kpts, kmesh, dm_k)

    zero_mask = np.all(shifts == np.array([0, 0, 0], dtype=np.int64), axis=1)
    if not np.any(zero_mask):
        raise RuntimeError("No zero shift found in periodic shift grid.")
    zero_idx = int(np.flatnonzero(zero_mask)[0])

    hamiltonian_ao_h = np.asarray(hamiltonian_shifted_h[zero_idx], dtype=np.float64)
    hcore_ao_h = np.asarray(hcore_shifted_h[zero_idx], dtype=np.float64)
    overlap_ao = np.asarray(overlap_shifted[zero_idx], dtype=np.float64)
    dm_ao = np.asarray(density_shifted[zero_idx], dtype=np.float64)

    forces_ev_per_angstrom: np.ndarray | None = None
    stress_ev_per_angstrom3: np.ndarray | None = None
    forces_stress_status = "not_requested"
    if args.require_forces_stress:
        grad_obj = mf.nuc_grad_method()
        if grad_obj is None:
            raise RuntimeError(
                "KRKS nuclear gradient object is not available; aborting."
            )

        gradients_h_per_bohr = np.asarray(grad_obj.kernel(), dtype=np.float64)
        if gradients_h_per_bohr.shape != (len(info.elements), 3):
            raise RuntimeError(
                "Unexpected gradient shape: "
                f"{gradients_h_per_bohr.shape} (expected {(len(info.elements), 3)})"
            )
        forces_ev_per_angstrom = (
            -gradients_h_per_bohr * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        )

        if not hasattr(grad_obj, "get_stress"):
            raise RuntimeError(
                "KRKS stress is unavailable in this production path; aborting."
            )
        stress_h_per_bohr3 = np.asarray(grad_obj.get_stress(), dtype=np.float64)
        if stress_h_per_bohr3.shape != (3, 3):
            raise RuntimeError(
                f"Unexpected stress shape: {stress_h_per_bohr3.shape} (expected (3,3))"
            )
        stress_ev_per_angstrom3 = (
            stress_h_per_bohr3 * STRESS_HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3
        )
        forces_stress_status = "computed"

    hamiltonian_shifted_ev = hamiltonian_shifted_h * HARTREE_TO_EV
    hcore_shifted_ev = hcore_shifted_h * HARTREE_TO_EV
    hamiltonian_ao_ev = hamiltonian_ao_h * HARTREE_TO_EV
    hcore_ao_ev = hcore_ao_h * HARTREE_TO_EV
    energy_ev = energy_hartree * HARTREE_TO_EV
    mo_energy_ev = np.asarray(mf.mo_energy, dtype=np.float64) * HARTREE_TO_EV

    density_trace_electrons = _safe_trace_real(dm_ao, overlap_ao)
    one_body_trace_energy_ev = _safe_trace_real(dm_ao, hamiltonian_ao_ev)
    num_electrons = int(cell.nelectron)

    np.save(ham_npy_path, hamiltonian_ao_ev.astype(np.float32))
    np.save(overlap_npy_path, overlap_ao.astype(np.float32))
    np.save(density_npy_path, dm_ao.astype(np.float32))
    np.save(shifts_npy_path, shifts.astype(np.int64))
    np.save(ham_shift_npy_path, hamiltonian_shifted_ev.astype(np.float32))
    np.save(overlap_shift_npy_path, overlap_shifted.astype(np.float32))
    np.save(density_shift_npy_path, density_shifted.astype(np.float32))

    npz_payload: dict[str, np.ndarray] = {
        "hamiltonian_ao": hamiltonian_ao_ev.astype(np.float32),
        "hcore_ao": hcore_ao_ev.astype(np.float32),
        "overlap_ao": overlap_ao.astype(np.float32),
        "dm_ao": dm_ao.astype(np.float32),
        "hamiltonian_shifted": hamiltonian_shifted_ev.astype(np.float32),
        "hcore_shifted": hcore_shifted_ev.astype(np.float32),
        "overlap_shifted": overlap_shifted.astype(np.float32),
        "density_shifted": density_shifted.astype(np.float32),
        "shifts": shifts.astype(np.int64),
        "kmesh": np.asarray(kmesh, dtype=np.int64),
        "kpts_abs": np.asarray(kpts, dtype=np.float64),
        "mo_coeff": np.asarray(mf.mo_coeff, dtype=np.float32),
        "mo_occ": np.asarray(mf.mo_occ, dtype=np.float32),
        "mo_energy_ev": mo_energy_ev.astype(np.float32),
        "positions_angstrom": positions_angstrom.astype(np.float32),
        "box_angstrom": box_angstrom.astype(np.float32),
        "total_energy_ev": np.asarray(energy_ev, dtype=np.float64),
        "num_electrons": np.asarray(num_electrons, dtype=np.int64),
        "density_trace_num_electrons": np.asarray(
            density_trace_electrons, dtype=np.float64
        ),
        "one_body_trace_energy_ev": np.asarray(
            one_body_trace_energy_ev, dtype=np.float64
        ),
    }
    if forces_ev_per_angstrom is not None:
        npz_payload["forces_ev_per_angstrom"] = forces_ev_per_angstrom.astype(
            np.float32
        )
    if stress_ev_per_angstrom3 is not None:
        npz_payload["stress_ev_per_angstrom3"] = stress_ev_per_angstrom3.astype(
            np.float32
        )

    np.savez_compressed(npz_path, **npz_payload)

    openmx_energies_ev = {
        key: float((value.item() if hasattr(value, "item") else value) * HARTREE_TO_EV)
        for key, value in sorted(info.energies.items())
    }
    openmx_forces_ev_per_angstrom = None
    if info.forces.numel():
        openmx_forces_ev_per_angstrom = (
            info.forces.detach().cpu().numpy().astype(np.float64)
            * FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        ).tolist()
    openmx_stress_ev_per_angstrom3 = None
    if info.stress.numel():
        openmx_stress_ev_per_angstrom3 = (
            info.stress.detach().cpu().numpy().astype(np.float64) * HARTREE_TO_EV
        ).tolist()

    result: dict[str, Any] = {
        "run_name": args.run_name,
        "units": {
            "energy": "eV",
            "length": "Angstrom",
            "force": "eV/Angstrom",
            "stress": "eV/Angstrom^3",
            "hamiltonian": "eV",
            "overlap": "dimensionless",
            "density": "electron",
        },
        "snapshot": {
            "info_path": str(info_path),
            "num_atoms": len(atom_spec),
            "atoms": atom_rows,
            "positions_angstrom": positions_angstrom.tolist(),
            "box_angstrom": box_angstrom.tolist(),
        },
        "settings": {
            "method": "KRKS",
            "xc": args.xc,
            "kmesh": [int(v) for v in kmesh],
            "charge": int(args.charge),
            "spin": int(args.spin),
            "grid_level": int(args.grid_level),
            "max_cycle": int(args.max_cycle),
            "conv_tol": float(args.conv_tol),
            "pyscf_verbose": int(args.pyscf_verbose),
            "basis_mode": "openmx_like_shell_count_matched",
            "orbital_set": orbital_set,
            "df_backend": args.df_backend,
            "df_auxbasis": args.df_auxbasis,
            "pbc_precision": float(args.pbc_precision),
            "ke_cutoff": args.ke_cutoff,
            "max_memory_mb": int(args.max_memory_mb),
            "require_forces_stress": bool(args.require_forces_stress),
            "hamiltonian_definition": "Kohn-Sham matrix hcore + veff (not PySCF get_fock)",
        },
        "openmx_reference": {
            "energies_ev": openmx_energies_ev,
            "forces_ev_per_angstrom": openmx_forces_ev_per_angstrom,
            "stress_ev_per_angstrom3": openmx_stress_ev_per_angstrom3,
        },
        "pyscf": {
            "converged": True,
            "total_energy_ev": float(energy_ev),
            "num_electrons": int(num_electrons),
            "density_trace_num_electrons": float(density_trace_electrons),
            "one_body_trace_energy_ev": float(one_body_trace_energy_ev),
            "forces_stress_status": forces_stress_status,
            "forces_ev_per_angstrom": (
                None
                if forces_ev_per_angstrom is None
                else forces_ev_per_angstrom.tolist()
            ),
            "stress_ev_per_angstrom3": (
                None
                if stress_ev_per_angstrom3 is None
                else stress_ev_per_angstrom3.tolist()
            ),
            "periodic": {
                "num_kpts": int(len(kpts)),
                "num_shifts": int(shifts.shape[0]),
                "num_nonzero_shifts": int(np.sum(np.any(shifts != 0, axis=1))),
                "zero_shift_index": int(zero_idx),
            },
        },
        "artifacts": {
            "xyz_path": str(xyz_path),
            "hamiltonian_npy_path": str(ham_npy_path),
            "overlap_npy_path": str(overlap_npy_path),
            "density_npy_path": str(density_npy_path),
            "shifts_npy_path": str(shifts_npy_path),
            "hamiltonian_shifted_npy_path": str(ham_shift_npy_path),
            "overlap_shifted_npy_path": str(overlap_shift_npy_path),
            "density_shifted_npy_path": str(density_shift_npy_path),
            "npz_path": str(npz_path),
        },
    }

    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"[pyscf-production] wrote {json_path}")
    print(f"[pyscf-production] wrote {npz_path}")
    print(f"[pyscf-production] converged=True method=KRKS E={energy_ev:.8f} eV")


if __name__ == "__main__":
    main()
