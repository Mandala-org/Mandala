from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch  # real PyTorch – no stub

# ---------------------------------------------------------------------------
# helper: recover lattice given frac + cartesian
# ---------------------------------------------------------------------------


def recover_box(frac_coords: torch.Tensor, abs_coords: torch.Tensor) -> torch.Tensor:
    """Recover the lattice matrix given fractional and absolute coordinates."""
    F = torch.as_tensor(frac_coords, dtype=torch.float64)
    A = torch.as_tensor(abs_coords, dtype=torch.float64)

    if F.shape != A.shape or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("Inputs must both be (N,3) arrays")

    cell, *_ = torch.linalg.lstsq(F, A)
    cell._requires_grad = True
    return cell  # (3,3)


# ---------------------------------------------------------------------------
# returned data container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InfoOutData:
    energies: Dict[str, torch.Tensor]
    occupancies: torch.Tensor
    record_meta: List[Tuple[int, str, str, int, int]]
    orbital_set: Dict[str, str]

    eigenvalues: torch.Tensor
    dipole_abs: torch.Tensor
    dipole: torch.Tensor

    elements: List[str]
    xyz: torch.Tensor  # Cartesian (Å)
    frac: torch.Tensor  # fractional (unit-cell)
    box: torch.Tensor  # (3,3) lattice matrix
    forces: torch.Tensor

    # convenience
    def occupancy_by_element(self, el: str) -> torch.Tensor:
        """Return occupancy tensor for the given element symbol."""
        mask = [e == el for _idx, e, *_ in self.record_meta]
        return self.occupancies[mask]


# ---------------------------------------------------------------------------
# regexps
# ---------------------------------------------------------------------------
_RE_ENERGY = re.compile(r"^\s*([A-Za-z0-9]+)\.\s+([-+0-9Ee\.]+)")

_RE_ATOM_HDR = re.compile(r"^\s*\d+\s+([A-Z][a-z]?)\s+Up spin", re.I)
_RE_ORB_PARSE = re.compile(
    r"^\s*([A-Za-z][^\s]*)\s+(\d+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)"
)

_RE_EIG_ROW = re.compile(r"^\s*\d+\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)")

_RE_DIP_ABS = re.compile(r"^\s*Absolute D\s+([-+0-9Ee\.]+)")
_RE_DIP_VEC = re.compile(
    r"^\s*Total\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)"
)

# Cartesian + forces
_RE_XYZ_START = re.compile(r"<coordinates\.forces", re.I)
_RE_XYZ_END = re.compile(r"coordinates\.forces>", re.I)

# Fractional coordinates

_FRAC_HEADER = re.compile(r"^.*Fractional coordinates of the final structure", re.I)

_ORB_ORDER = "spdfghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# main parser
# ---------------------------------------------------------------------------


def parse_info_out(path: str | Path) -> InfoOutData:  # noqa: C901 (single large fn)
    lines = Path(path).read_text(errors="ignore").splitlines()

    # 1) energies -----------------------------------------------------------
    energies: Dict[str, torch.Tensor] = {}
    for ln in lines:
        if m := _RE_ENERGY.match(ln):
            energies[m.group(1)] = torch.tensor(float(m.group(2)), dtype=torch.float64)

    # 2) occupancies --------------------------------------------------------
    occup, meta = [], []
    mul_ctr: Dict[str, Dict[str, set[int]]] = {}
    atom_id: Dict[str, int] = {}
    cur_el: Optional[str] = None
    in_block = False

    for ln in lines:
        if h := _RE_ATOM_HDR.match(ln):
            cur_el = h.group(1)
            atom_id[cur_el] = atom_id.get(cur_el, 0)
            in_block = True
            continue

        if not in_block:
            continue
        if not ln.strip():
            in_block = False
            atom_id[cur_el] += 1
            continue
        if ln.lstrip().startswith(("sum", "multiple")):
            continue
        if m := _RE_ORB_PARSE.match(ln):
            sym, mul_s, up, dn = m.groups()
            l = sym[0].lower()
            mul = int(mul_s)
            occup.append(torch.tensor([float(up), float(dn)], dtype=torch.float64))
            meta.append((atom_id[cur_el], cur_el, l, mul, 0))
            mul_ctr.setdefault(cur_el, {}).setdefault(l, set()).add(mul)

    occupancies = torch.stack(occup, dim=0) if occup else torch.tensor([])

    orbital_set = {
        el: "".join(f"{len(ld[l])}{l}" for l in sorted(ld, key=_ORB_ORDER.index))
        for el, ld in mul_ctr.items()
    }

    # 3) eigenvalues --------------------------------------------------------
    eig_rows = []
    grab = False
    for ln in lines:
        if "Eigenvalues" in ln and "SCF" in ln:
            grab = True
            continue
        if grab:
            if not ln.strip():
                if eig_rows:
                    break
                continue
            if m := _RE_EIG_ROW.match(ln):
                eig_rows.append(
                    torch.tensor(
                        [float(m.group(1)), float(m.group(2))], dtype=torch.float64
                    )
                )

    eigenvalues = torch.stack(eig_rows, dim=0) if eig_rows else torch.tensor([])

    # 4) dipole -------------------------------------------------------------
    dip_abs = torch.tensor(float("nan"), dtype=torch.float64)
    dip_vec = torch.tensor([float("nan")] * 3, dtype=torch.float64)
    for ln in lines:
        if m := _RE_DIP_ABS.match(ln):
            dip_abs = torch.tensor(float(m.group(1)), dtype=torch.float64)
        elif m := _RE_DIP_VEC.match(ln):
            dip_vec = torch.tensor([float(x) for x in m.groups()], dtype=torch.float64)
            break

    # 5a) Cartesian coords/forces ------------------------------------------
    elements, xyz_list, f_list = [], [], []
    in_xyz = False
    for ln in lines:
        if _RE_XYZ_START.search(ln):
            in_xyz = True
            continue
        if in_xyz and _RE_XYZ_END.search(ln):
            in_xyz = False
            break
        if not in_xyz:
            continue
        parts = ln.split()
        if len(parts) >= 8:
            elements.append(parts[1])
            xyz_list.append([float(v) for v in parts[2:5]])
            f_list.append([float(v) for v in parts[5:8]])

    xyz = torch.tensor(xyz_list, dtype=torch.float64, requires_grad=True)
    forces = torch.tensor(f_list, dtype=torch.float64)

    # 5b) Fractional coords -------------------------------------------------
    frac_list: list[list[float]] = []
    i = 0
    while i < len(lines):
        if _FRAC_HEADER.search(lines[i]):
            # Skip banner (4-5 lines of stars / blank)
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("*")):
                i += 1
            # Now numeric block
            while i < len(lines):
                ln = lines[i]
                if not ln.strip():  # blank line ends block
                    break
                # Expected: idx  Elem  f1 f2 f3   (≥5 tokens)
                parts = ln.split()
                if len(parts) >= 5 and parts[0].isdigit():
                    frac_list.append([float(v) for v in parts[2:5]])
                else:  # header-like row → stop
                    break
                i += 1
            break
        i += 1

    frac = (
        torch.tensor(frac_list, dtype=torch.float64) if frac_list else torch.tensor([])
    )

    # 5c) lattice matrix ----------------------------------------------------
    if xyz.numel() and frac.numel() and xyz.shape == frac.shape:
        box = recover_box(frac, xyz)  # (3,3)
    else:
        box = torch.tensor([])

    # 6) pack ---------------------------------------------------------------
    return InfoOutData(
        energies=energies,
        occupancies=occupancies,
        record_meta=meta,
        orbital_set=orbital_set,
        eigenvalues=eigenvalues,
        dipole_abs=dip_abs,
        dipole=dip_vec,
        elements=elements,
        xyz=xyz,
        frac=frac,
        box=box,
        forces=forces,
    )
