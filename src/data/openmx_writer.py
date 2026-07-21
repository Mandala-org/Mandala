"""Write predicted H/S/D blocks into an OpenMX ``HS.out`` layout.

OpenMX's text dump contains information that is not part of Mandala's matrix
objects, notably local neighbour indices, integer ``Rn`` identifiers, and
position/momentum overlap sections.  The writer therefore uses an existing
``HS.out`` as a template and changes only the numeric H/S/D block rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import torch

from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix

_HAMILTONIAN_HEADER = "Kohn-Sham Hamiltonian spin=0"
_OVERLAP_HEADER = "Overlap matrix"
_DENSITY_HEADER = "Density matrix spin=0"
_POSITION_X_HEADER = "Overlap matrix with position operator x"

_BLOCK_HEADER_RE = re.compile(
    r"^global index=(\d+)\s+local index=(\d+)\s+"
    r"\(global=(\d+),\s*Rn=(-?\d+)"
    r"(?:\s+(-?\d+)\s+(-?\d+)\s+(-?\d+))?\)$"
)


@dataclass(frozen=True)
class OpenMXWriteStats:
    """Block counts recorded while filling an OpenMX template."""

    blocks_written: Mapping[str, int]
    zero_filled_blocks: Mapping[str, int]
    density_sections: int


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.readlines()


def _rn_shift_map(lines: list[str]) -> dict[int, tuple[int, int, int]]:
    shifts: dict[int, tuple[int, int, int]] = {}
    in_position_x = False
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == _POSITION_X_HEADER:
            in_position_x = True
            continue
        if line in {_HAMILTONIAN_HEADER, _OVERLAP_HEADER, _DENSITY_HEADER} or (
            line.startswith("Overlap matrix with ") and line != _POSITION_X_HEADER
        ):
            in_position_x = False
        if not in_position_x:
            continue
        match = _BLOCK_HEADER_RE.fullmatch(line)
        if match is None:
            continue
        if match.group(5) is None:
            raise ValueError(
                f"Missing lattice shift in position-overlap header: {line}"
            )
        rn = int(match.group(4))
        shift = tuple(int(match.group(index)) for index in (5, 6, 7))
        previous = shifts.setdefault(rn, shift)
        if previous != shift:
            raise ValueError(f"Rn={rn} maps to both {previous} and {shift}")
    if not shifts:
        raise ValueError("Template has no position-overlap Rn-to-shift mapping")
    return shifts


def _format_row(row: torch.Tensor, newline: str) -> str:
    return "".join(f"{float(value):20.16f} " for value in row) + newline


def write_openmx_hsout_prediction(
    template_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    predictions: Mapping[str, BlockMatrix],
    *,
    zero_fill_missing: bool = True,
) -> OpenMXWriteStats:
    """Fill the H/S/D sections of an OpenMX text dump from block matrices.

    The template's complete block support is retained.  If a model does not
    predict a template edge (normally because it lies outside the model
    cutoff), the corresponding block is written as zeros when
    ``zero_fill_missing`` is true.

    OpenMX stores one half of the non-spin-polarized density matrix in its
    first ``Density matrix spin=0`` section; Mandala's parser reconstructs the
    full density as ``D + D.T``.  Consequently, this writer stores half of the
    supplied, symmetrized density prediction in that first section.  The
    second identically named section is OpenMX's zero-valued imaginary-density
    slot and is retained as zeros.
    """

    required = {"hamiltonian", "overlap", "density"}
    if set(predictions) < required:
        missing = sorted(required - set(predictions))
        raise ValueError(f"Missing predicted matrices: {missing}")

    matrices = {name: predictions[name] for name in sorted(required)}
    first = matrices["hamiltonian"]
    for name, matrix in matrices.items():
        if matrix.basis != "openmx":
            raise ValueError(f"{name} must use the OpenMX basis, got {matrix.basis!r}")
        if matrix.atoms != first.atoms:
            raise ValueError("Predicted matrices must use the same atom ordering")
        if matrix.orbital_cfg.to_dict() != first.orbital_cfg.to_dict():
            raise ValueError("Predicted matrices must use the same orbital basis")

    template = Path(template_path)
    output = Path(output_path)
    lines = _read_lines(template)
    shifts = _rn_shift_map(lines)
    newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
    dtype = next(iter(first.pair_blocks.values())).dtype
    mapper = BlockIrrepMapper(first.orbital_cfg, dtype=dtype)

    written = {name: 0 for name in required}
    zeroed = {name: 0 for name in required}
    density_sections = 0
    current: str | None = None
    output_lines: list[str] = []
    index = 0

    while index < len(lines):
        raw = lines[index]
        stripped = raw.rstrip("\r\n")
        if stripped == _HAMILTONIAN_HEADER:
            current = "hamiltonian"
        elif stripped == _OVERLAP_HEADER:
            current = "overlap"
        elif stripped == _DENSITY_HEADER:
            density_sections += 1
            current = "density" if density_sections == 1 else "density_zero"
        elif stripped.startswith("Overlap matrix with "):
            current = None

        match = _BLOCK_HEADER_RE.fullmatch(stripped) if current is not None else None
        if match is None:
            output_lines.append(raw)
            index += 1
            continue

        src = int(match.group(1)) - 1
        dst = int(match.group(3)) - 1
        rn = int(match.group(4))
        if rn not in shifts:
            raise ValueError(f"No lattice shift found for Rn={rn}: {stripped}")
        if not (0 <= src < len(first.atoms) and 0 <= dst < len(first.atoms)):
            raise ValueError(f"Atom index outside prediction: {stripped}")

        key = f"{first.atoms[src]}-{first.atoms[dst]}"
        rows, columns = mapper.block_dims(key)
        output_lines.append(raw)  # Preserve the complete block header verbatim.

        for offset in range(1, rows + 1):
            if index + offset >= len(lines):
                raise ValueError(f"Unexpected EOF inside block: {stripped}")
            numeric = lines[index + offset].split()
            if len(numeric) != columns:
                raise ValueError(
                    f"Template row has {len(numeric)} values; expected {columns}: "
                    f"{stripped}"
                )
            try:
                [float(value) for value in numeric]
            except ValueError as exc:
                raise ValueError(f"Non-numeric template row below: {stripped}") from exc

        matrix_name = "density" if current == "density_zero" else current
        assert matrix_name is not None
        if current == "density_zero":
            block = torch.zeros((rows, columns), dtype=dtype)
        else:
            matrix = matrices[matrix_name]
            edge = (*shifts[rn], src, dst)
            location = matrix.lookup.get(edge)
            if location is None:
                if not zero_fill_missing:
                    raise KeyError(
                        f"Prediction has no {matrix_name} block for edge {edge}"
                    )
                block = torch.zeros((rows, columns), dtype=dtype)
                zeroed[matrix_name] += 1
            else:
                location_key, block_index = location
                if location_key != key:
                    raise ValueError(
                        f"Prediction lookup for {edge} has key {location_key}, expected {key}"
                    )
                block = matrix.pair_blocks[key][block_index].detach().cpu()
                if block.shape != (rows, columns):
                    raise ValueError(
                        f"Prediction block {edge} has shape {tuple(block.shape)}, "
                        f"expected {(rows, columns)}"
                    )
                if current == "density":
                    block = 0.5 * block
            written[matrix_name] += 1

        output_lines.extend(_format_row(block[row], newline) for row in range(rows))
        index += rows + 1

    if density_sections != 2:
        raise ValueError(
            f"Expected exactly two '{_DENSITY_HEADER}' sections, found {density_sections}"
        )
    if any(count == 0 for count in written.values()):
        raise ValueError(f"One or more H/S/D sections were not written: {written}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f"{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.writelines(output_lines)
    try:
        os.replace(temporary, output)
        os.chmod(output, template.stat().st_mode & 0o777)
    finally:
        if temporary.exists():
            temporary.unlink()

    return OpenMXWriteStats(
        blocks_written=written,
        zero_filled_blocks=zeroed,
        density_sections=density_sections,
    )


__all__ = ["OpenMXWriteStats", "write_openmx_hsout_prediction"]
