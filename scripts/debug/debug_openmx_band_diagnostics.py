#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.snapshot import Snapshot  # noqa: E402
from data.kspace_snapshot import block_matrix_to_shiftspace_dense  # noqa: E402
from utils.units import HARTREE_TO_EV  # noqa: E402


HARTREE_PER_EV = 1.0 / HARTREE_TO_EV


@dataclass
class OpenMXBandReference:
    labels: list[str]
    tick_positions: list[float]
    x_values: torch.Tensor
    spectra_by_position: dict[float, torch.Tensor]


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostics for OpenMX real-space -> k-space band reconstruction."
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=Path("data/small/Si/HS.out"),
        help="OpenMX HS.out path.",
    )
    parser.add_argument(
        "--info-path",
        type=Path,
        default=Path("data/small/Si/Si.out"),
        help="OpenMX Si.out path.",
    )
    parser.add_argument(
        "--banddat-path",
        type=Path,
        default=Path("data/small/Si/Si.BANDDAT1"),
        help="OpenMX BANDDAT file.",
    )
    parser.add_argument(
        "--gnuband-path",
        type=Path,
        default=Path("data/small/Si/Si.GNUBAND"),
        help="OpenMX GNUBAND helper file.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=[
            "all",
            "reference",
            "units",
            "gamma",
            "path_compare",
            "overlap",
            "hermiticity",
            "phase_sign",
            "phase_units",
            "atom_phase",
            "shift_pairs",
        ],
        help="Diagnostic to run.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="openmx",
        choices=["openmx", "e3nn"],
        help="Basis convention to load for diagnostics.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="How many eigenvalues to print in comparisons.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/debug_openmx_band_diagnostics"),
        help="Directory for cached parsed snapshot/reference payloads.",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Ignore cached parsed inputs and rebuild them from source files.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(message, flush=True)


def _cache_key(*parts: Any) -> str:
    text = "||".join(str(p) for p in parts)
    return sha1(text.encode("utf-8")).hexdigest()[:16]


def _file_sig(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))


def _parse_band_segments(
    info_path: Path,
) -> list[tuple[int, torch.Tensor, torch.Tensor, str, str]]:
    text = info_path.read_text()
    m = re.search(r"<Band\.kpath(.*?)Band\.kpath>", text, re.S)
    if m is None:
        raise ValueError(f"Could not find <Band.kpath> block in {info_path}")
    segments = []
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
        start_label = parts[7]
        end_label = parts[8]
        segments.append((npts, start, end, start_label, end_label))
    return segments


def _parse_gnuband_ticks(gnuband_path: Path) -> tuple[list[str], list[float]]:
    text = gnuband_path.read_text()
    m = re.search(r"set xtics \((.*)\)", text)
    if m is None:
        raise ValueError(f"Could not find xtics in {gnuband_path}")
    labels: list[str] = []
    positions: list[float] = []
    for part in m.group(1).split(","):
        label, pos = part.rsplit(" ", 1)
        labels.append(label.strip().strip('"'))
        positions.append(float(pos))
    return labels, positions


def _parse_banddat_raw(banddat_path: Path) -> dict[float, list[float]]:
    grouped: dict[float, list[float]] = {}
    for raw in banddat_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        x = round(float(parts[0]), 12)
        ev = float(parts[1])
        grouped.setdefault(x, []).append(ev)
    return grouped


def _parse_special_point_blocks(banddat_path: Path) -> dict[float, list[float]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw in banddat_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)

    specials: dict[float, list[float]] = {}
    for block in blocks:
        if len(block) >= 20:
            break
        xs = {round(float(line.split()[0]), 12) for line in block}
        if len(xs) != 1:
            continue
        x = next(iter(xs))
        specials[x] = [float(line.split()[1]) for line in block]
    return specials


def _remove_one_match(values: list[float], target: float, tol: float = 1.0e-4) -> None:
    best_idx = None
    best_err = None
    for idx, val in enumerate(values):
        err = abs(val - target)
        if best_err is None or err < best_err:
            best_err = err
            best_idx = idx
    if best_idx is None or best_err is None or best_err > tol:
        raise ValueError(
            f"Could not match special-point eigenvalue {target:.6f} within tolerance {tol}"
        )
    values.pop(best_idx)


def _spectrum_from_group(
    raw_values: list[float],
    special_values: list[float] | None,
) -> torch.Tensor:
    vals = sorted(float(v) for v in raw_values)
    if len(vals) == 104:
        return torch.tensor(vals, dtype=torch.float64)
    if len(vals) == 208:
        paired = [0.5 * (vals[2 * i] + vals[2 * i + 1]) for i in range(104)]
        return torch.tensor(paired, dtype=torch.float64)
    if len(vals) == 220 and special_values is not None and len(special_values) == 12:
        remaining = vals.copy()
        for val in special_values:
            _remove_one_match(remaining, val)
        if len(remaining) != 208:
            raise ValueError(
                f"Expected 208 values after removing special-point dump, got {len(remaining)}"
            )
        paired = [0.5 * (remaining[2 * i] + remaining[2 * i + 1]) for i in range(104)]
        return torch.tensor(paired, dtype=torch.float64)
    raise ValueError(
        f"Unsupported raw spectrum size {len(vals)} for OpenMX BANDDAT reference"
    )


def load_openmx_reference(
    banddat_path: Path, gnuband_path: Path
) -> OpenMXBandReference:
    labels, tick_positions = _parse_gnuband_ticks(gnuband_path)
    grouped = _parse_banddat_raw(banddat_path)
    specials = _parse_special_point_blocks(banddat_path)
    spectra_by_position: dict[float, torch.Tensor] = {}
    x_values = torch.tensor(sorted(grouped.keys()), dtype=torch.float64)
    for key_t in x_values.tolist():
        key = round(float(key_t), 12)
        spectra_by_position[key] = _spectrum_from_group(grouped[key], specials.get(key))
    return OpenMXBandReference(
        labels=labels,
        tick_positions=tick_positions,
        x_values=x_values,
        spectra_by_position=spectra_by_position,
    )


def _reference_to_payload(reference: OpenMXBandReference) -> dict[str, Any]:
    return {
        "labels": list(reference.labels),
        "tick_positions": list(reference.tick_positions),
        "x_values": reference.x_values.detach().cpu(),
        "spectra_by_position": {
            str(k): v.detach().cpu() for k, v in reference.spectra_by_position.items()
        },
    }


def _reference_from_payload(payload: dict[str, Any]) -> OpenMXBandReference:
    return OpenMXBandReference(
        labels=list(payload["labels"]),
        tick_positions=[float(x) for x in payload["tick_positions"]],
        x_values=payload["x_values"],
        spectra_by_position={
            float(k): v for k, v in payload["spectra_by_position"].items()
        },
    )


def _snapshot_to_payload(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "snapshot": snapshot._payload(),
        "info_fermi_level": (
            None
            if getattr(snapshot.info, "fermi_level", None) is None
            else snapshot.info.fermi_level.detach().cpu()
        ),
    }


def _snapshot_from_payload(payload: dict[str, Any]) -> Snapshot:
    top = payload["snapshot"]
    mats = {
        name: Snapshot._matrix_from_payload(pld, device="cpu")
        for name, pld in top["mats"].items()
    }
    positions = top.get("positions")
    forces = top.get("forces")
    box = top.get("box")
    stress = top.get("stress")
    info = None
    fermi = payload.get("info_fermi_level")
    if fermi is not None:
        info = SimpleNamespace(fermi_level=fermi)
    return Snapshot(
        mats["hamiltonian"],
        mats["overlap"],
        mats["density"],
        positions=positions,
        forces=forces,
        box=box,
        stress=stress,
        matrix_path=top.get("matrix_path"),
        info_path=top.get("info_path"),
        cutoff_radius=top.get("cutoff_radius"),
        info=info,
    )


def _reference_eigs_at_position(
    reference: OpenMXBandReference,
    position: float,
) -> torch.Tensor:
    key = round(float(position), 12)
    if key not in reference.spectra_by_position:
        raise KeyError(f"No OpenMX reference spectrum found at x={position:.6f}")
    return reference.spectra_by_position[key]


def _format_label(label: str) -> str:
    return "Gamma" if label in {"G", r"$\Gamma$"} else label


def _fractional_to_cartesian_k(
    k_frac: torch.Tensor,
    box: torch.Tensor,
) -> torch.Tensor:
    reciprocal = 2.0 * torch.pi * torch.linalg.inv(box).T
    return k_frac.to(dtype=box.dtype, device=box.device) @ reciprocal


def _global_offsets(snapshot: Snapshot) -> torch.Tensor:
    dims = [
        snapshot.hamiltonian.orbital_cfg.block_dims(f"{el}-{el}")[0]
        for el in snapshot.hamiltonian.atoms
    ]
    offsets = [0]
    for dim in dims[:-1]:
        offsets.append(offsets[-1] + int(dim))
    return torch.tensor(offsets, dtype=torch.long)


def _assemble_kspace_dense(
    mat,
    *,
    k_frac: torch.Tensor,
    box: torch.Tensor,
    positions: torch.Tensor | None,
    sign: int,
    use_cartesian: bool,
    include_atom_positions: bool,
) -> torch.Tensor:
    total_dim = sum(mat.orbital_cfg.block_dims(f"{el}-{el}")[0] for el in mat.atoms)
    any_block = next(iter(mat.pair_blocks.values()))
    out = torch.zeros(
        (total_dim, total_dim), dtype=torch.complex128, device=any_block.device
    )
    offsets = _global_offsets_from_matrix(mat, device=any_block.device)
    box_t = box.to(device=any_block.device, dtype=torch.float64)
    k_frac_t = k_frac.to(device=any_block.device, dtype=torch.float64)
    k_cart = _fractional_to_cartesian_k(k_frac_t.unsqueeze(0), box_t).squeeze(0)
    pos_t = None
    if positions is not None:
        pos_t = positions.to(device=any_block.device, dtype=torch.float64)
    for key, edges in mat.pair_edges.items():
        blocks = mat.pair_blocks[key].to(torch.complex128)
        for idx, edge in enumerate(edges.t().tolist()):
            sx, sy, sz, i, j = edge
            shift = torch.tensor(
                [sx, sy, sz], dtype=torch.float64, device=any_block.device
            )
            if use_cartesian:
                vec = shift @ box_t
                if include_atom_positions:
                    if pos_t is None:
                        raise ValueError(
                            "Atom positions are required for include_atom_positions=True"
                        )
                    vec = vec + (pos_t[j] - pos_t[i])
                phase_arg = torch.dot(k_cart, vec)
            else:
                phase_arg = 2.0 * math.pi * torch.dot(k_frac_t, shift)
                if include_atom_positions:
                    if pos_t is None:
                        raise ValueError(
                            "Atom positions are required for include_atom_positions=True"
                        )
                    frac_pos = torch.linalg.solve(box_t.T, pos_t.T).T
                    phase_arg = phase_arg + 2.0 * math.pi * torch.dot(
                        k_frac_t, frac_pos[j] - frac_pos[i]
                    )
            phase = torch.exp(
                torch.tensor(sign * 1j, dtype=torch.complex128, device=any_block.device)
                * phase_arg.to(torch.complex128)
            )
            di = mat.orbital_cfg.block_dims(f"{mat.atoms[i]}-{mat.atoms[i]}")[0]
            dj = mat.orbital_cfg.block_dims(f"{mat.atoms[j]}-{mat.atoms[j]}")[0]
            r0 = int(offsets[i].item())
            c0 = int(offsets[j].item())
            out[r0 : r0 + di, c0 : c0 + dj] += phase * blocks[idx]
    return out


def _global_offsets_from_matrix(mat, *, device: torch.device) -> torch.Tensor:
    dims = [mat.orbital_cfg.block_dims(f"{el}-{el}")[0] for el in mat.atoms]
    offsets = [0]
    for dim in dims[:-1]:
        offsets.append(offsets[-1] + int(dim))
    return torch.tensor(offsets, dtype=torch.long, device=device)


def _generalized_eigs_ev(H: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    Hh = 0.5 * (H + H.transpose(-1, -2).conj())
    Sh = 0.5 * (S + S.transpose(-1, -2).conj())
    L = torch.linalg.cholesky(Sh)
    tmp = torch.linalg.solve(L, Hh)
    A = torch.linalg.solve(L, tmp.transpose(-1, -2).conj()).transpose(-1, -2).conj()
    A = 0.5 * (A + A.transpose(-1, -2).conj())
    return torch.linalg.eigvalsh(A).real * HARTREE_TO_EV


def _compare_eigenvalues(
    observed_ev: torch.Tensor,
    reference_ev: torch.Tensor,
    *,
    top_n: int,
) -> dict[str, float]:
    n = min(observed_ev.numel(), reference_ev.numel())
    obs = observed_ev[:n]
    ref = reference_ev[:n]
    abs_err = torch.abs(obs - ref)
    return {
        "n": float(n),
        "mae_ev": float(abs_err.mean().item()),
        "max_abs_ev": float(abs_err.max().item()),
        "first_values": obs[:top_n].detach().cpu().tolist(),
        "first_reference": ref[:top_n].detach().cpu().tolist(),
    }


def _log_comparison(
    prefix: str,
    observed_ev: torch.Tensor,
    reference_ev: torch.Tensor,
    *,
    top_n: int,
) -> None:
    stats = _compare_eigenvalues(observed_ev, reference_ev, top_n=top_n)
    _log(f"{prefix} MAE (eV): {stats['mae_ev']:.6f}")
    _log(f"{prefix} Max abs err (eV): {stats['max_abs_ev']:.6f}")
    _log(f"{prefix} First observed eigs (eV): {stats['first_values']}")
    _log(f"{prefix} First OpenMX eigs (eV): {stats['first_reference']}")


def _snapshot_cache_path(args: argparse.Namespace) -> Path:
    key = _cache_key(
        "snapshot",
        _file_sig(args.matrix_path),
        _file_sig(args.info_path),
        args.convention,
    )
    return args.cache_dir / f"snapshot_{key}.pt"


def _reference_cache_path(args: argparse.Namespace) -> Path:
    key = _cache_key(
        "reference",
        _file_sig(args.banddat_path),
        _file_sig(args.gnuband_path),
    )
    return args.cache_dir / f"reference_{key}.pt"


def _load_snapshot(args: argparse.Namespace) -> Snapshot:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _snapshot_cache_path(args)
    if cache_path.exists() and not args.force_reparse:
        try:
            _log(f"[cache] Loading parsed snapshot from {cache_path}")
            return _snapshot_from_payload(
                torch.load(cache_path, map_location="cpu", weights_only=False)
            )
        except Exception as exc:
            _log(f"[cache] Snapshot cache load failed ({exc}); rebuilding.")
    snapshot = Snapshot.from_openmx(
        args.matrix_path,
        args.info_path,
        convention=args.convention,
    )
    torch.save(_snapshot_to_payload(snapshot), cache_path)
    _log(f"[cache] Saved parsed snapshot to {cache_path}")
    return snapshot


def _load_reference(args: argparse.Namespace) -> OpenMXBandReference:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _reference_cache_path(args)
    if cache_path.exists() and not args.force_reparse:
        try:
            _log(f"[cache] Loading parsed OpenMX band reference from {cache_path}")
            return _reference_from_payload(
                torch.load(cache_path, map_location="cpu", weights_only=False)
            )
        except Exception as exc:
            _log(f"[cache] Reference cache load failed ({exc}); rebuilding.")
    reference = load_openmx_reference(args.banddat_path, args.gnuband_path)
    torch.save(_reference_to_payload(reference), cache_path)
    _log(f"[cache] Saved parsed OpenMX band reference to {cache_path}")
    return reference


def _print_header(title: str) -> None:
    _log("")
    _log(f"=== {title} ===")


def run_reference(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Reference")
    _log(f"snapshot basis: {snapshot.hamiltonian.basis}")
    _log(f"atom count: {len(snapshot.hamiltonian.atoms)}")
    _log(
        f"box shape: {tuple(snapshot.box.shape) if snapshot.box is not None else None}"
    )
    _log(f"OpenMX labels: {reference.labels}")
    _log(f"OpenMX tick positions: {reference.tick_positions}")
    counts = {
        pos: int(reference.spectra_by_position[round(float(pos), 12)].numel())
        for pos in reference.tick_positions
    }
    _log(f"OpenMX reference band counts at ticks: {counts}")
    _log(f"OpenMX unique x positions: {int(reference.x_values.numel())}")


def run_units(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Units")
    fermi_h = float(snapshot.info.fermi_level.item())
    _log(f"snapshot info fermi (Hartree): {fermi_h:.6f}")
    _log(f"snapshot info fermi (eV): {fermi_h * HARTREE_TO_EV:.6f}")
    all_ref = torch.cat(list(reference.spectra_by_position.values()))
    ref_min = float(all_ref.min().item())
    ref_max = float(all_ref.max().item())
    _log(f"OpenMX reference eigenvalue range (eV): [{ref_min:.6f}, {ref_max:.6f}]")
    shifts = snapshot.get_translation_shifts()
    H_shift = block_matrix_to_shiftspace_dense(snapshot.hamiltonian, shifts=shifts)
    S_shift = block_matrix_to_shiftspace_dense(snapshot.overlap, shifts=shifts)
    gamma = torch.zeros((1, 3), dtype=snapshot.box.dtype, device=snapshot.box.device)
    from core.periodic_fourier import shiftspace_to_kspace_dense

    H_gamma = shiftspace_to_kspace_dense(
        H_shift, kpoints_abs=gamma, shifts=shifts, box=snapshot.box
    )[0]
    S_gamma = shiftspace_to_kspace_dense(
        S_shift, kpoints_abs=gamma, shifts=shifts, box=snapshot.box
    )[0]
    eig_gamma_h = _generalized_eigs_ev(H_gamma, S_gamma) * HARTREE_PER_EV
    _log(
        "Gamma eigenvalue range from our reconstruction: "
        f"[{float(eig_gamma_h.min().item()):.6f}, {float(eig_gamma_h.max().item()):.6f}] Hartree"
    )
    _log(
        "Gamma eigenvalue range from our reconstruction: "
        f"[{float((eig_gamma_h * HARTREE_TO_EV).min().item()):.6f}, "
        f"{float((eig_gamma_h * HARTREE_TO_EV).max().item()):.6f}] eV"
    )


def run_gamma(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Gamma")
    gamma_pos = None
    for label, pos in zip(reference.labels, reference.tick_positions):
        if label == "G":
            gamma_pos = pos
            break
    if gamma_pos is None:
        raise ValueError("Could not find Gamma in GNUBAND ticks")
    ref_gamma = _reference_eigs_at_position(reference, gamma_pos)
    k_gamma_frac = torch.zeros(3, dtype=torch.float64)
    H_gamma = _assemble_kspace_dense(
        snapshot.hamiltonian,
        k_frac=k_gamma_frac,
        box=snapshot.box,
        positions=snapshot.positions,
        sign=+1,
        use_cartesian=True,
        include_atom_positions=False,
    )
    S_gamma = _assemble_kspace_dense(
        snapshot.overlap,
        k_frac=k_gamma_frac,
        box=snapshot.box,
        positions=snapshot.positions,
        sign=+1,
        use_cartesian=True,
        include_atom_positions=False,
    )
    eig_gamma = _generalized_eigs_ev(H_gamma, S_gamma)
    fermi_ev = float(snapshot.info.fermi_level.item() * HARTREE_TO_EV)
    _log(f"Gamma comparison against OpenMX at x={gamma_pos:.6f}")
    _log_comparison("Raw", eig_gamma, ref_gamma, top_n=args.top_n)
    _log_comparison(
        "Shifted by snapshot E_F",
        eig_gamma - fermi_ev,
        ref_gamma,
        top_n=args.top_n,
    )


def run_overlap(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Overlap")
    gamma = torch.zeros(3, dtype=torch.float64)
    S_gamma = _assemble_kspace_dense(
        snapshot.overlap,
        k_frac=gamma,
        box=snapshot.box,
        positions=snapshot.positions,
        sign=+1,
        use_cartesian=True,
        include_atom_positions=False,
    )
    diag = torch.diagonal(S_gamma).real
    eigs = torch.linalg.eigvalsh(0.5 * (S_gamma + S_gamma.T.conj())).real
    _log(
        "diag(S(Gamma)) stats: "
        f"min={float(diag.min().item()):.6f} "
        f"max={float(diag.max().item()):.6f} "
        f"mean={float(diag.mean().item()):.6f}"
    )
    _log(
        "eig(S(Gamma)) stats: "
        f"min={float(eigs.min().item()):.6e} "
        f"max={float(eigs.max().item()):.6e}"
    )
    _log(f"Number of negative eigs below -1e-8: {int((eigs < -1e-8).sum().item())}")


def run_hermiticity(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Hermiticity")
    shifts = snapshot.get_translation_shifts()
    H_shift = block_matrix_to_shiftspace_dense(snapshot.hamiltonian, shifts=shifts)
    S_shift = block_matrix_to_shiftspace_dense(snapshot.overlap, shifts=shifts)
    shift_to_idx = {tuple(map(int, s.tolist())): i for i, s in enumerate(shifts)}
    h_errs: list[float] = []
    s_errs: list[float] = []
    missing = 0
    for idx, shift in enumerate(shifts.tolist()):
        neg = tuple(-int(x) for x in shift)
        j = shift_to_idx.get(neg)
        if j is None:
            missing += 1
            continue
        h_errs.append(
            float(torch.max(torch.abs(H_shift[idx] - H_shift[j].T.conj())).item())
        )
        s_errs.append(
            float(torch.max(torch.abs(S_shift[idx] - S_shift[j].T.conj())).item())
        )
    _log(f"shift count: {len(shifts)}")
    _log(f"missing opposite shifts: {missing}")
    _log(f"max |H(R) - H(-R)^H|: {max(h_errs) if h_errs else float('nan'):.6e}")
    _log(f"max |S(R) - S(-R)^H|: {max(s_errs) if s_errs else float('nan'):.6e}")


def _special_point_map(info_path: Path) -> dict[str, torch.Tensor]:
    points: dict[str, torch.Tensor] = {}
    for _, start, end, start_label, end_label in _parse_band_segments(info_path):
        points.setdefault(start_label, start)
        points.setdefault(end_label, end)
    return points


def _openmx_path_kpoints(info_path: Path) -> torch.Tensor:
    segments = _parse_band_segments(info_path)
    if not segments:
        raise ValueError(f"No OpenMX band path found in {info_path}")
    chunks: list[torch.Tensor] = []
    for seg_idx, (npts, start, end, _start_label, _end_label) in enumerate(segments):
        t = torch.linspace(0.0, 1.0, npts, dtype=torch.float64).unsqueeze(1)
        pts = (1.0 - t) * start.unsqueeze(0) + t * end.unsqueeze(0)
        if seg_idx > 0:
            pts = pts[1:]
        chunks.append(pts)
    return torch.cat(chunks, dim=0)


def run_phase_sign(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Phase Sign")
    points = _special_point_map(args.info_path)
    fermi_ev = float(snapshot.info.fermi_level.item() * HARTREE_TO_EV)
    for label in ["G", "X", "U", "L"]:
        if label not in points:
            continue
        pos = None
        for lab, tick in zip(reference.labels, reference.tick_positions):
            if lab == label:
                pos = tick
                break
        if pos is None:
            continue
        ref_eigs = _reference_eigs_at_position(reference, pos)
        k_frac = points[label]
        for sign in (+1, -1):
            Hk = _assemble_kspace_dense(
                snapshot.hamiltonian,
                k_frac=k_frac,
                box=snapshot.box,
                positions=snapshot.positions,
                sign=sign,
                use_cartesian=True,
                include_atom_positions=False,
            )
            Sk = _assemble_kspace_dense(
                snapshot.overlap,
                k_frac=k_frac,
                box=snapshot.box,
                positions=snapshot.positions,
                sign=sign,
                use_cartesian=True,
                include_atom_positions=False,
            )
            eigs = _generalized_eigs_ev(Hk, Sk)
            raw = _compare_eigenvalues(eigs, ref_eigs, top_n=4)
            shifted = _compare_eigenvalues(eigs - fermi_ev, ref_eigs, top_n=4)
            _log(
                f"{_format_label(label)} sign={'+' if sign > 0 else '-'} "
                f"raw_MAE={raw['mae_ev']:.6f} eV raw_max={raw['max_abs_ev']:.6f} eV "
                f"shifted_MAE={shifted['mae_ev']:.6f} eV shifted_max={shifted['max_abs_ev']:.6f} eV"
            )


def run_path_compare(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Path Compare")
    fermi_ev = float(snapshot.info.fermi_level.item() * HARTREE_TO_EV)
    kpoints_frac = _openmx_path_kpoints(args.info_path)
    if kpoints_frac.shape[0] != reference.x_values.numel():
        raise ValueError(
            f"k-point count mismatch: built {kpoints_frac.shape[0]} points but "
            f"reference has {reference.x_values.numel()} x-values"
        )
    reciprocal = 2.0 * torch.pi * torch.linalg.inv(snapshot.box).T.to(torch.float64)
    kpoints_abs = (kpoints_frac @ reciprocal).to(snapshot.box.dtype)
    band = snapshot.get_band_structure(
        kpoints_abs=kpoints_abs,
        fractional_kpoints=kpoints_frac.to(snapshot.box.dtype),
        shifts=snapshot.get_translation_shifts().to(snapshot.box.device),
        chunk_size=32,
        show_progress=False,
        psd_cleanup=False,
        allow_jitter=False,
    )
    eigs_all = (
        band.eigenvalues.detach().cpu().to(torch.float64) * HARTREE_TO_EV - fermi_ev
    )
    ref_all = torch.stack(
        [
            _reference_eigs_at_position(reference, x)
            for x in reference.x_values.tolist()
        ],
        dim=0,
    )
    err = torch.abs(eigs_all[:, : ref_all.shape[1]] - ref_all)
    maes_t = err.mean(dim=1)
    maxes_t = err.max(dim=1).values
    worst_idx = int(torch.argmax(maxes_t).item())
    worst_x = float(reference.x_values[worst_idx].item())
    _log(f"path points compared: {int(reference.x_values.numel())}")
    _log(
        f"shifted path MAE stats (eV): min={float(maes_t.min().item()):.6f} "
        f"mean={float(maes_t.mean().item()):.6f} max={float(maes_t.max().item()):.6f}"
    )
    _log(
        f"shifted path max-abs stats (eV): min={float(maxes_t.min().item()):.6f} "
        f"mean={float(maxes_t.mean().item()):.6f} max={float(maxes_t.max().item()):.6f}"
    )
    _log(f"worst x position: {worst_x:.6f}")


def run_phase_units(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Phase Units")
    points = _special_point_map(args.info_path)
    for label in ["X", "U", "L"]:
        if label not in points:
            continue
        k_frac = points[label]
        k_cart = _fractional_to_cartesian_k(
            k_frac.unsqueeze(0), snapshot.box.to(torch.float64)
        ).squeeze(0)
        shifts = snapshot.get_translation_shifts().to(torch.float64)
        box = snapshot.box.to(torch.float64)
        phase_cart = torch.exp(1j * ((shifts @ box) @ k_cart))
        phase_frac = torch.exp(1j * (2.0 * math.pi * (shifts @ k_frac)))
        max_err = float(torch.max(torch.abs(phase_cart - phase_frac)).item())
        _log(
            f"{_format_label(label)} max |exp(i k_cart·R) - exp(2pi i k_frac·shift)| = {max_err:.6e}"
        )


def run_atom_phase(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Atom Phase")
    points = _special_point_map(args.info_path)
    fermi_ev = float(snapshot.info.fermi_level.item() * HARTREE_TO_EV)
    for label in ["X", "U", "L"]:
        if label not in points:
            continue
        pos = None
        for lab, tick in zip(reference.labels, reference.tick_positions):
            if lab == label:
                pos = tick
                break
        if pos is None:
            continue
        ref_eigs = _reference_eigs_at_position(reference, pos)
        k_frac = points[label]
        for include_atom_positions in (False, True):
            Hk = _assemble_kspace_dense(
                snapshot.hamiltonian,
                k_frac=k_frac,
                box=snapshot.box,
                positions=snapshot.positions,
                sign=+1,
                use_cartesian=True,
                include_atom_positions=include_atom_positions,
            )
            Sk = _assemble_kspace_dense(
                snapshot.overlap,
                k_frac=k_frac,
                box=snapshot.box,
                positions=snapshot.positions,
                sign=+1,
                use_cartesian=True,
                include_atom_positions=include_atom_positions,
            )
            eigs = _generalized_eigs_ev(Hk, Sk)
            raw = _compare_eigenvalues(eigs, ref_eigs, top_n=4)
            shifted = _compare_eigenvalues(eigs - fermi_ev, ref_eigs, top_n=4)
            mode = "R only" if not include_atom_positions else "R + r_j - r_i"
            _log(
                f"{_format_label(label)} phase={mode} "
                f"raw_MAE={raw['mae_ev']:.6f} eV raw_max={raw['max_abs_ev']:.6f} eV "
                f"shifted_MAE={shifted['mae_ev']:.6f} eV shifted_max={shifted['max_abs_ev']:.6f} eV"
            )


def run_shift_pairs(
    args: argparse.Namespace, snapshot: Snapshot, reference: OpenMXBandReference
) -> None:
    _print_header("Shift Pairs")
    shifts = snapshot.get_translation_shifts()
    shift_set = {tuple(map(int, s.tolist())) for s in shifts}
    missing = []
    for shift in shift_set:
        neg = tuple(-x for x in shift)
        if neg not in shift_set:
            missing.append((shift, neg))
    _log(f"total unique shifts: {len(shift_set)}")
    _log(f"missing opposite pairs: {len(missing)}")
    if missing:
        _log(f"first missing pairs: {missing[:10]}")


def main() -> None:
    args = setup_argparse()
    snapshot = _load_snapshot(args)
    reference = _load_reference(args)

    tests = {
        "reference": run_reference,
        "units": run_units,
        "gamma": run_gamma,
        "path_compare": run_path_compare,
        "overlap": run_overlap,
        "hermiticity": run_hermiticity,
        "phase_sign": run_phase_sign,
        "phase_units": run_phase_units,
        "atom_phase": run_atom_phase,
        "shift_pairs": run_shift_pairs,
    }
    if args.test == "all":
        order = [
            "reference",
            "units",
            "gamma",
            "path_compare",
            "overlap",
            "hermiticity",
            "shift_pairs",
            "phase_units",
            "phase_sign",
            "atom_phase",
        ]
        for name in order:
            tests[name](args, snapshot, reference)
    else:
        tests[args.test](args, snapshot, reference)


if __name__ == "__main__":
    main()
