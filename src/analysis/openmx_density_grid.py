"""Real-space electron-density reconstruction from OpenMX block matrices."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import torch
from ase.data import atomic_numbers
from tqdm.auto import tqdm

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.sparse_math import trace_matmul_sparse_block_matrix
from data.block_matrix import BlockMatrix
from data.openmx_info_parser import parse_info_out
from data.openmx_parser import parse_openmx_scfout
from data.snapshot import Snapshot


ANGSTROM_TO_BOHR = 1.8897261254578281


@dataclass(frozen=True)
class OpenMXGrid:
    """Regular OpenMX real-space grid, with all vectors expressed in bohr."""

    shape: tuple[int, int, int]
    origin: torch.Tensor
    steps: torch.Tensor

    @property
    def num_points(self) -> int:
        return math.prod(self.shape)

    @property
    def voxel_volume(self) -> float:
        return float(torch.abs(torch.linalg.det(self.steps)).item())


@dataclass(frozen=True)
class PAORadialTable:
    """Radial pseudo-atomic orbitals tabulated on an increasing bohr grid."""

    path: Path
    cutoff_bohr: float
    radii: torch.Tensor
    values_by_l: Mapping[int, torch.Tensor]


@dataclass(frozen=True)
class DensityGridResult:
    """Reconstructed densities and scalar validation metrics."""

    densities: Mapping[str, torch.Tensor]
    metrics: Mapping[str, object]


_GRID_SHAPE_RE = re.compile(
    r"Num\.\s+of\s+grids\s+of\s+a-,\s*b-,\s*and\s*c-axes\s*=\s*"
    r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
    re.I,
)
_GRID_INDEX_RE = re.compile(r"Num\.Grid([123])\.\s+(\d+)", re.I)
_GRID_ORIGIN_RE = re.compile(
    r"Grid_Origin\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
    re.I,
)
_GRID_STEP_RE = re.compile(
    r"gtv_([abc])\s*=\s*([-+0-9.eE]+)\s*,\s*" r"([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)",
    re.I,
)
_PAO_CUTOFF_RE = re.compile(r"^\s*radial\.cutoff\.pao\s+([-+0-9.eE]+)", re.I)
_PAO_SECTION_START_RE = re.compile(r"^\s*<pseudo\.atomic\.orbitals\.L=(\d+)\s*$", re.I)
_PAO_SECTION_END_RE = re.compile(r"^\s*pseudo\.atomic\.orbitals\.L=(\d+)>\s*$", re.I)


def parse_openmx_grid(paths: Sequence[str | Path], *, dtype: torch.dtype) -> OpenMXGrid:
    """Parse shape, origin, and voxel vectors from one or more OpenMX outputs."""

    shape: tuple[int, int, int] | None = None
    indexed_shape: dict[int, int] = {}
    origin: tuple[float, float, float] | None = None
    steps: dict[str, tuple[float, float, float]] = {}
    inspected: list[str] = []

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Grid metadata file not found: {path}")
        inspected.append(str(path))
        for line in path.read_text(errors="ignore").splitlines():
            if match := _GRID_SHAPE_RE.search(line):
                candidate = tuple(int(value) for value in match.groups())
                if shape is not None and shape != candidate:
                    raise ValueError(
                        f"Conflicting OpenMX grid shapes: {shape} and {candidate}"
                    )
                shape = candidate
            if match := _GRID_INDEX_RE.search(line):
                indexed_shape[int(match.group(1)) - 1] = int(match.group(2))
            if match := _GRID_ORIGIN_RE.search(line):
                candidate = tuple(float(value) for value in match.groups())
                if origin is not None and any(
                    abs(left - right) > 1.0e-10
                    for left, right in zip(origin, candidate)
                ):
                    raise ValueError(
                        f"Conflicting OpenMX grid origins: {origin} and {candidate}"
                    )
                origin = candidate
            if match := _GRID_STEP_RE.search(line):
                label = match.group(1).lower()
                candidate = tuple(float(value) for value in match.groups()[1:])
                if label in steps and any(
                    abs(left - right) > 1.0e-10
                    for left, right in zip(steps[label], candidate)
                ):
                    raise ValueError(f"Conflicting OpenMX gtv_{label} vectors")
                steps[label] = candidate

    if shape is None and len(indexed_shape) == 3:
        shape = tuple(indexed_shape[index] for index in range(3))
    missing = []
    if shape is None:
        missing.append("shape")
    if origin is None:
        missing.append("Grid_Origin")
    if set(steps) != {"a", "b", "c"}:
        missing.append("gtv_a/gtv_b/gtv_c")
    if missing:
        raise ValueError(
            f"Missing OpenMX grid metadata ({', '.join(missing)}) in {inspected}"
        )
    assert shape is not None and origin is not None
    return OpenMXGrid(
        shape=shape,
        origin=torch.tensor(origin, dtype=dtype),
        steps=torch.tensor([steps[label] for label in "abc"], dtype=dtype),
    )


def parse_openmx_pao(path: str | Path, *, dtype: torch.dtype) -> PAORadialTable:
    """Parse numerical radial functions from an OpenMX ``*.pao`` file."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OpenMX PAO file not found: {path}")
    cutoff: float | None = None
    rows_by_l: dict[int, list[list[float]]] = {}
    active_l: int | None = None
    for line_number, line in enumerate(
        path.read_text(errors="ignore").splitlines(), start=1
    ):
        if match := _PAO_CUTOFF_RE.match(line):
            cutoff = float(match.group(1))
        if match := _PAO_SECTION_START_RE.match(line):
            active_l = int(match.group(1))
            rows_by_l[active_l] = []
            continue
        if match := _PAO_SECTION_END_RE.match(line):
            if active_l != int(match.group(1)):
                raise ValueError(f"Mismatched PAO section end at {path}:{line_number}")
            active_l = None
            continue
        if active_l is None:
            continue
        try:
            row = [float(value) for value in line.split()]
        except ValueError as exc:
            raise ValueError(f"Non-numeric PAO row at {path}:{line_number}") from exc
        if len(row) < 3:
            raise ValueError(f"Short PAO row at {path}:{line_number}")
        rows_by_l[active_l].append(row)

    if cutoff is None:
        raise ValueError(f"No radial.cutoff.pao found in {path}")
    if not rows_by_l:
        raise ValueError(f"No pseudo.atomic.orbitals sections found in {path}")

    radii: torch.Tensor | None = None
    values_by_l: dict[int, torch.Tensor] = {}
    for angular_momentum, rows in sorted(rows_by_l.items()):
        table = torch.tensor(rows, dtype=dtype)
        # OpenMX columns are log(r), r in bohr, followed by radial PAOs.
        section_radii = table[:, 1].contiguous()
        if not torch.all(section_radii[1:] > section_radii[:-1]):
            raise ValueError(
                f"PAO radial grid is not strictly increasing for L={angular_momentum}"
            )
        if radii is None:
            radii = section_radii
        elif not torch.allclose(radii, section_radii, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"PAO radial grids differ between angular momenta in {path}"
            )
        values_by_l[angular_momentum] = table[:, 2:].contiguous()
    assert radii is not None
    return PAORadialTable(path, cutoff, radii, values_by_l)


def load_openmx_snapshot(
    matrix_path: str | Path,
    info_path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
) -> Snapshot:
    """Load H/S/D while retaining OpenMX coordinates and basis conventions."""

    matrix_path = Path(matrix_path).expanduser().resolve()
    info_path = Path(info_path).expanduser().resolve()
    info = parse_info_out(info_path, dtype=dtype)
    if not info.elements or info.positions.shape != (len(info.elements), 3):
        raise ValueError(f"Could not parse atoms and positions from {info_path}")
    if info.box.shape != (3, 3):
        raise ValueError(f"Could not parse a 3x3 lattice from {info_path}")
    if not info.orbital_set:
        raise ValueError(f"Could not infer the orbital basis from {info_path}")
    snapshot = parse_openmx_scfout(
        matrix_path,
        info.elements,
        OrbitalIrrepConfig.from_dict(info.orbital_set),
        convention="openmx",
        # Retain native spin=0 normalization; averaging is not a spin sum.
        symmetrize_density=True,
    )
    snapshot.positions = info.positions
    snapshot.box = info.box
    snapshot.matrix_path = matrix_path
    snapshot.info_path = info_path
    snapshot.info = info
    return snapshot


def _associated_legendre(l: int, m: int, z: torch.Tensor) -> torch.Tensor:
    p_mm = torch.ones_like(z)
    if m:
        root = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        for index in range(1, m + 1):
            p_mm = -(2 * index - 1) * root * p_mm
    if l == m:
        return p_mm
    p_lm_previous = p_mm
    p_lm = (2 * m + 1) * z * p_mm
    if l == m + 1:
        return p_lm
    for degree in range(m + 2, l + 1):
        next_value = (
            (2 * degree - 1) * z * p_lm - (degree + m - 1) * p_lm_previous
        ) / (degree - m)
        p_lm_previous, p_lm = p_lm, next_value
    return p_lm


_OPENMX_FROM_STANDARD_REAL = {
    0: (0,),
    1: (2, 0, 1),
    2: (2, 4, 0, 3, 1),
    3: (3, 4, 2, 5, 1, 6, 0),
}


def real_spherical_harmonics_openmx(l: int, displacement: torch.Tensor) -> torch.Tensor:
    """Evaluate normalized real harmonics in native OpenMX orbital order."""

    if l not in _OPENMX_FROM_STANDARD_REAL:
        raise ValueError(
            f"OpenMX real harmonics are implemented only for l=0..3, got l={l}"
        )
    radius = torch.linalg.vector_norm(displacement, dim=-1)
    safe_radius = torch.where(radius > 0.0, radius, torch.ones_like(radius))
    x, y, z = (displacement[:, index] / safe_radius for index in range(3))
    azimuth = torch.atan2(y, x)
    standard: list[torch.Tensor] = []
    for m in range(-l, l + 1):
        abs_m = abs(m)
        normalization = math.sqrt(
            (2 * l + 1)
            / (4.0 * math.pi)
            * math.factorial(l - abs_m)
            / math.factorial(l + abs_m)
        )
        legendre = _associated_legendre(l, abs_m, z)
        if m < 0:
            value = (
                math.sqrt(2.0)
                * ((-1.0) ** abs_m)
                * normalization
                * legendre
                * torch.sin(abs_m * azimuth)
            )
        elif m == 0:
            value = normalization * legendre
        else:
            value = (
                math.sqrt(2.0)
                * ((-1.0) ** m)
                * normalization
                * legendre
                * torch.cos(m * azimuth)
            )
        standard.append(value)
    result = torch.stack(standard, dim=-1)[:, _OPENMX_FROM_STANDARD_REAL[l]]
    if l > 0:
        result = torch.where(
            (radius > 0.0).unsqueeze(-1), result, torch.zeros_like(result)
        )
    return result


def _interpolate_radial(
    radius: torch.Tensor,
    table_radii: torch.Tensor,
    table_values: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    table_radii = table_radii.to(device=radius.device, dtype=radius.dtype)
    table_values = table_values.to(device=radius.device, dtype=radius.dtype)
    upper = torch.searchsorted(table_radii, radius).clamp(1, table_radii.numel() - 1)
    lower = upper - 1
    lower_radius = table_radii.index_select(0, lower)
    upper_radius = table_radii.index_select(0, upper)
    fraction = (radius - lower_radius) / (upper_radius - lower_radius)
    lower_value = table_values.index_select(0, lower)
    upper_value = table_values.index_select(0, upper)
    interpolated = lower_value + fraction.unsqueeze(-1) * (upper_value - lower_value)
    interpolated = torch.where(
        (radius <= table_radii[0]).unsqueeze(-1), table_values[0], interpolated
    )
    return torch.where(
        (radius <= float(cutoff)).unsqueeze(-1),
        interpolated,
        torch.zeros_like(interpolated),
    )


def evaluate_atomic_basis(
    element: str,
    displacement: torch.Tensor,
    orbital_cfg: OrbitalIrrepConfig,
    pao: PAORadialTable,
) -> torch.Tensor:
    """Evaluate all PAOs for one atom/image at a set of grid points."""

    radius = torch.linalg.vector_norm(displacement, dim=-1)
    columns: list[torch.Tensor] = []
    for multiplicity, irrep in orbital_cfg.element_to_irreps[element]:
        l = irrep.l
        if l not in pao.values_by_l:
            raise ValueError(
                f"{pao.path} has no radial functions for l={l} ({element})"
            )
        radial_table = pao.values_by_l[l]
        if radial_table.shape[1] < multiplicity:
            raise ValueError(
                f"{element} requires {multiplicity} radial functions for l={l}, "
                f"but {pao.path} contains {radial_table.shape[1]}"
            )
        radial = _interpolate_radial(
            radius,
            pao.radii,
            radial_table[:, :multiplicity],
            pao.cutoff_bohr,
        )
        angular = real_spherical_harmonics_openmx(l, displacement)
        for zeta in range(multiplicity):
            columns.append(radial[:, zeta : zeta + 1] * angular)
    result = torch.cat(columns, dim=-1)
    expected = orbital_cfg.element_to_irreps[element].dim
    if result.shape[1] != expected:
        raise RuntimeError(
            f"Evaluated {result.shape[1]} orbitals for {element}; expected {expected}"
        )
    return result


def _grid_fractional_bounds(
    grid: OpenMXGrid, box_bohr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    extents = torch.tensor([size - 1 for size in grid.shape], dtype=grid.origin.dtype)
    corners = []
    for bits in product((0.0, 1.0), repeat=3):
        coefficients = extents * torch.tensor(bits, dtype=grid.origin.dtype)
        corners.append(grid.origin + coefficients @ grid.steps)
    fractional = torch.stack(corners) @ torch.linalg.inv(box_bohr)
    return fractional.amin(dim=0), fractional.amax(dim=0)


def _relevant_atom_images(
    positions_bohr: torch.Tensor,
    box_bohr: torch.Tensor,
    grid: OpenMXGrid,
    paos: Mapping[str, PAORadialTable],
    atoms: Sequence[str],
) -> list[set[tuple[int, int, int]]]:
    fractional_min, fractional_max = _grid_fractional_bounds(grid, box_bohr)
    inverse_box = torch.linalg.inv(box_bohr)
    atom_fractional = positions_bohr @ inverse_box
    image_sets: list[set[tuple[int, int, int]]] = []
    for atom_index, element in enumerate(atoms):
        # If |delta_cart| <= cutoff, each fractional component is bounded by
        # cutoff * ||inverse_box[:, component]||. This is conservative for a
        # triclinic cell and cannot discard a contributing image.
        margin = paos[element].cutoff_bohr * torch.linalg.vector_norm(
            inverse_box, dim=0
        )
        lower = torch.ceil(fractional_min - margin - atom_fractional[atom_index]).to(
            torch.long
        )
        upper = torch.floor(fractional_max + margin - atom_fractional[atom_index]).to(
            torch.long
        )
        ranges = [range(int(lower[i]), int(upper[i]) + 1) for i in range(3)]
        image_sets.append({tuple(values) for values in product(*ranges)})
    return image_sets


@dataclass(frozen=True)
class _ExpandedTerms:
    key: str
    block_indices: torch.Tensor
    source_centers: tuple[tuple[int, int, int, int], ...]
    target_centers: tuple[tuple[int, int, int, int], ...]


def _expand_periodic_terms(
    matrix: BlockMatrix,
    image_sets: Sequence[set[tuple[int, int, int]]],
) -> tuple[list[_ExpandedTerms], set[tuple[int, int, int, int]]]:
    groups: list[_ExpandedTerms] = []
    centers: set[tuple[int, int, int, int]] = set()
    for key, edges in matrix.pair_edges.items():
        block_indices: list[int] = []
        sources: list[tuple[int, int, int, int]] = []
        targets: list[tuple[int, int, int, int]] = []
        for block_index, edge in enumerate(edges.t().tolist()):
            sx, sy, sz, source, target = (int(value) for value in edge)
            relative_shift = (sx, sy, sz)
            target_images = image_sets[target]
            for common_shift in image_sets[source]:
                target_shift = tuple(
                    common_shift[index] + relative_shift[index] for index in range(3)
                )
                if target_shift not in target_images:
                    continue
                source_center = (source, *common_shift)
                target_center = (target, *target_shift)
                block_indices.append(block_index)
                sources.append(source_center)
                targets.append(target_center)
                centers.add(source_center)
                centers.add(target_center)
        if block_indices:
            groups.append(
                _ExpandedTerms(
                    key,
                    torch.tensor(block_indices, dtype=torch.long),
                    tuple(sources),
                    tuple(targets),
                )
            )
    return groups, centers


def _grid_points(
    grid: OpenMXGrid, start: int, stop: int, device: torch.device
) -> torch.Tensor:
    ny, nz = grid.shape[1:]
    flat = torch.arange(start, stop, device=device, dtype=torch.long)
    i = torch.div(flat, ny * nz, rounding_mode="floor")
    remainder = flat - i * ny * nz
    j = torch.div(remainder, nz, rounding_mode="floor")
    k = remainder - j * nz
    coefficients = torch.stack((i, j, k), dim=-1).to(dtype=grid.origin.dtype)
    return grid.origin.to(device) + coefficients @ grid.steps.to(device)


def _validate_matrix_alignment(
    reference: BlockMatrix, other: BlockMatrix, name: str
) -> None:
    if reference.atoms != other.atoms:
        raise ValueError(f"{name} atom order differs from the reference matrix")
    if reference.basis != "openmx" or other.basis != "openmx":
        raise ValueError("Density reconstruction requires matrices in the OpenMX basis")
    if reference.orbital_cfg.to_dict() != other.orbital_cfg.to_dict():
        raise ValueError(f"{name} orbital basis differs from the reference matrix")
    missing = set(reference.lookup) - set(other.lookup)
    extra = set(other.lookup) - set(reference.lookup)
    if missing or extra:
        raise ValueError(
            f"{name} periodic edge support differs: missing={len(missing)} extra={len(extra)}"
        )


def density_hermiticity_metrics(matrix: BlockMatrix) -> dict[str, float]:
    maximum = 0.0
    total = 0.0
    count = 0
    for edge, (key, index) in matrix.lookup.items():
        sx, sy, sz, source, target = edge
        reverse = (-sx, -sy, -sz, target, source)
        reverse_key, reverse_index = matrix.lookup[reverse]
        difference = (
            matrix.pair_blocks[key][index]
            - matrix.pair_blocks[reverse_key][reverse_index].T
        )
        maximum = max(maximum, float(difference.abs().max().item()))
        total += float(difference.abs().sum().item())
        count += difference.numel()
    return {"max_abs": maximum, "mae": total / count if count else 0.0}


def evaluate_density_grids(
    snapshots: Mapping[str, Snapshot],
    grid: OpenMXGrid,
    paos: Mapping[str, PAORadialTable],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    point_chunk_size: int = 4096,
    edge_batch_size: int = 32,
    expected_electrons: float | None = None,
) -> DensityGridResult:
    """Evaluate matrices in their supplied normalization on one OpenMX grid.

    No spin sum is implicit. Native OpenMX spin=0 inputs produce a per-spin
    grid; expected_electrons must use that same normalization.
    """

    if not snapshots:
        raise ValueError("At least one snapshot is required")
    names = list(snapshots)
    reference = snapshots[names[0]]
    if reference.positions is None or reference.box is None:
        raise ValueError("Reference snapshot has no positions or lattice")
    for name in names[1:]:
        _validate_matrix_alignment(reference.density, snapshots[name].density, name)
    missing_paos = sorted(set(reference.density.atoms) - set(paos))
    if missing_paos:
        raise ValueError(f"Missing PAO files for elements: {missing_paos}")
    if point_chunk_size <= 0 or edge_batch_size <= 0:
        raise ValueError("Chunk sizes must be positive")

    device = torch.device(device)
    positions_bohr = reference.positions.to(dtype=dtype) * ANGSTROM_TO_BOHR
    box_bohr = reference.box.to(dtype=dtype) * ANGSTROM_TO_BOHR
    grid = OpenMXGrid(
        grid.shape,
        grid.origin.to(dtype=dtype),
        grid.steps.to(dtype=dtype),
    )
    image_sets = _relevant_atom_images(
        positions_bohr,
        box_bohr,
        grid,
        paos,
        reference.density.atoms,
    )
    terms, centers = _expand_periodic_terms(reference.density, image_sets)
    if not terms:
        raise ValueError("No density-matrix terms overlap the requested grid")
    expanded_count = sum(len(group.block_indices) for group in terms)
    print(
        f"--- Density grid: shape={grid.shape}, points={grid.num_points:,}, "
        f"voxel={grid.voxel_volume:.9g} bohr^3 ---"
    )
    print(
        f"--- Periodic expansion: matrix_edges={len(reference.density.lookup):,}, "
        f"image_terms={expanded_count:,}, unique_atom_images={len(centers):,} ---"
    )

    matrices_by_key: dict[str, torch.Tensor] = {}
    for group in terms:
        aligned = []
        reference_edges = reference.density.pair_edges[group.key]
        for name in names:
            density = snapshots[name].density
            blocks = []
            for edge in reference_edges.t().tolist():
                location_key, location_index = density.lookup[
                    tuple(int(v) for v in edge)
                ]
                blocks.append(density.pair_blocks[location_key][location_index])
            aligned.append(torch.stack(blocks))
        matrices_by_key[group.key] = torch.stack(aligned).to(device=device, dtype=dtype)

    atom_centers: dict[tuple[int, int, int, int], torch.Tensor] = {}
    for center in centers:
        atom_index, sx, sy, sz = center
        shift = torch.tensor((sx, sy, sz), dtype=dtype)
        atom_centers[center] = positions_bohr[atom_index] + shift @ box_bohr

    centers_by_element: dict[
        str, list[tuple[tuple[int, int, int, int], torch.Tensor]]
    ] = {}
    for center, center_position in atom_centers.items():
        element = reference.density.atoms[center[0]]
        centers_by_element.setdefault(element, []).append((center, center_position))
    center_locations: dict[tuple[int, int, int, int], tuple[str, int]] = {}
    for element, element_centers in centers_by_element.items():
        for center_index, (center, _) in enumerate(element_centers):
            center_locations[center] = (element, center_index)
    term_center_indices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for group in terms:
        source_element, target_element = group.key.split("-")
        source_indices = []
        target_indices = []
        for center in group.source_centers:
            element, center_index = center_locations[center]
            if element != source_element:
                raise RuntimeError(f"Source-center element mismatch for {group.key}")
            source_indices.append(center_index)
        for center in group.target_centers:
            element, center_index = center_locations[center]
            if element != target_element:
                raise RuntimeError(f"Target-center element mismatch for {group.key}")
            target_indices.append(center_index)
        term_center_indices[group.key] = (
            torch.tensor(source_indices, dtype=torch.long, device=device),
            torch.tensor(target_indices, dtype=torch.long, device=device),
        )

    flat_densities = torch.empty(
        (len(names), grid.num_points), dtype=dtype, device="cpu"
    )
    ranges = range(0, grid.num_points, point_chunk_size)
    progress = tqdm(ranges, desc="Density grid chunks", unit="chunk")
    for start in progress:
        stop = min(start + point_chunk_size, grid.num_points)
        points = _grid_points(grid, start, stop, device)
        basis_by_element: dict[str, torch.Tensor] = {}
        for element, element_centers in centers_by_element.items():
            center_positions = torch.stack(
                [center_position for _, center_position in element_centers]
            ).to(device=device)
            displacement = points.unsqueeze(0) - center_positions.unsqueeze(1)
            evaluated = evaluate_atomic_basis(
                element,
                displacement.reshape(-1, 3),
                reference.density.orbital_cfg,
                paos[element],
            ).reshape(len(element_centers), stop - start, -1)
            basis_by_element[element] = evaluated

        chunk_density = torch.zeros(
            (len(names), stop - start), dtype=dtype, device=device
        )
        for group in terms:
            all_blocks = matrices_by_key[group.key]
            source_element, target_element = group.key.split("-")
            all_source_indices, all_target_indices = term_center_indices[group.key]
            for edge_start in range(0, len(group.block_indices), edge_batch_size):
                edge_stop = min(edge_start + edge_batch_size, len(group.block_indices))
                block_indices = group.block_indices[edge_start:edge_stop].to(device)
                source_basis = basis_by_element[source_element].index_select(
                    0, all_source_indices[edge_start:edge_stop]
                )
                target_basis = basis_by_element[target_element].index_select(
                    0, all_target_indices[edge_start:edge_stop]
                )
                blocks = all_blocks.index_select(1, block_indices)
                # bmm is substantially faster than the equivalent three-input
                # einsum on CPU while preserving the exact contraction.
                for matrix_index in range(len(names)):
                    projected = torch.bmm(source_basis, blocks[matrix_index])
                    chunk_density[matrix_index] += (projected * target_basis).sum(
                        dim=(0, 2)
                    )
        flat_densities[:, start:stop] = chunk_density.detach().cpu()

    densities = {
        name: flat_densities[index].reshape(grid.shape)
        for index, name in enumerate(names)
    }
    metrics: dict[str, object] = {
        "grid": {
            "shape": list(grid.shape),
            "origin_bohr": grid.origin.tolist(),
            "steps_bohr": grid.steps.tolist(),
            "voxel_volume_bohr3": grid.voxel_volume,
        },
        "periodic_expansion": {
            "matrix_edges": len(reference.density.lookup),
            "image_terms": expanded_count,
            "unique_atom_images": len(centers),
        },
        "expected_electrons": expected_electrons,
        "matrices": {},
    }
    for name, snapshot in snapshots.items():
        density = densities[name]
        integral = float(density.sum().item() * grid.voxel_volume)
        trace_gt_overlap = float(
            trace_matmul_sparse_block_matrix(snapshot.density, reference.overlap)
            .detach()
            .item()
        )
        trace_own_overlap = float(
            trace_matmul_sparse_block_matrix(snapshot.density, snapshot.overlap)
            .detach()
            .item()
        )
        metrics["matrices"][name] = {
            "integrated_electrons": integral,
            "integrated_electron_error": (
                integral - expected_electrons
                if expected_electrons is not None
                else None
            ),
            "trace_density_ground_truth_overlap": trace_gt_overlap,
            "trace_density_own_overlap": trace_own_overlap,
            "density_min_e_per_bohr3": float(density.min().item()),
            "density_max_e_per_bohr3": float(density.max().item()),
            "hermiticity": density_hermiticity_metrics(snapshot.density),
        }
    if len(names) == 2:
        reference_density = densities[names[0]]
        predicted_density = densities[names[1]]
        error = predicted_density - reference_density
        metrics["comparison"] = {
            "reference": names[0],
            "prediction": names[1],
            "electron_count_error": float(error.sum().item() * grid.voxel_volume),
            "mae_e_per_bohr3": float(error.abs().mean().item()),
            "rmse_e_per_bohr3": float(torch.sqrt(torch.mean(error.square())).item()),
            "max_abs_error_e_per_bohr3": float(error.abs().max().item()),
            "integrated_abs_error_e": float(
                error.abs().sum().item() * grid.voxel_volume
            ),
            "l2_error_e_per_bohr3_sqrt_bohr3": float(
                torch.sqrt(error.square().sum() * grid.voxel_volume).item()
            ),
        }
    return DensityGridResult(densities=densities, metrics=metrics)


def write_cube(
    path: str | Path,
    values: torch.Tensor,
    grid: OpenMXGrid,
    atoms: Sequence[str],
    positions_angstrom: torch.Tensor,
    *,
    comment: str,
) -> None:
    """Write a scalar field in Gaussian cube format (bohr coordinates)."""

    path = Path(path)
    if tuple(values.shape) != grid.shape:
        raise ValueError(
            f"Cube values have shape {tuple(values.shape)}, expected {grid.shape}"
        )
    positions_bohr = (
        positions_angstrom.detach().cpu().to(torch.float64) * ANGSTROM_TO_BOHR
    )
    cube_origin = grid.origin.detach().cpu().to(torch.float64)
    cube_steps = grid.steps.detach().cpu().to(torch.float64)
    cube_cell = torch.tensor(grid.shape, dtype=torch.float64).unsqueeze(1) * cube_steps
    fractional = (positions_bohr - cube_origin) @ torch.linalg.inv(cube_cell)
    positions_bohr = cube_origin + (fractional - torch.floor(fractional)) @ cube_cell
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{comment}\n")
        handle.write("Generated by Mandala OpenMX density reconstruction\n")
        handle.write(
            f"{len(atoms):5d} {float(grid.origin[0]):13.6f} "
            f"{float(grid.origin[1]):13.6f} {float(grid.origin[2]):13.6f}\n"
        )
        for size, step in zip(grid.shape, grid.steps):
            handle.write(
                f"{size:5d} {float(step[0]):13.6f} "
                f"{float(step[1]):13.6f} {float(step[2]):13.6f}\n"
            )
        for element, position in zip(atoms, positions_bohr):
            number = atomic_numbers[element]
            handle.write(
                f"{number:5d} {float(number):13.6f} {float(position[0]):13.6f} "
                f"{float(position[1]):13.6f} {float(position[2]):13.6f}\n"
            )
        flat = values.detach().cpu().reshape(-1)
        for start in range(0, flat.numel(), 6):
            handle.write(
                " ".join(f"{float(value):13.5e}" for value in flat[start : start + 6])
            )
            handle.write("\n")


def write_density_outputs(
    output_dir: str | Path,
    result: DensityGridResult,
    grid: OpenMXGrid,
    snapshot: Snapshot,
) -> None:
    """Write cube files, metrics JSON, and orthogonal slice-triplet figures."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(result.densities)
    if len(names) == 1:
        write_cube(
            output_dir / "density.cube",
            result.densities[names[0]],
            grid,
            snapshot.density.atoms,
            snapshot.positions,
            comment=f"Electron density (input normalization): {names[0]} (e/bohr^3)",
        )
    elif len(names) == 2:
        ground_truth = result.densities[names[0]]
        prediction = result.densities[names[1]]
        error = prediction - ground_truth
        write_cube(
            output_dir / "ground_truth_density.cube",
            ground_truth,
            grid,
            snapshot.density.atoms,
            snapshot.positions,
            comment="Ground-truth electron density, input normalization (e/bohr^3)",
        )
        write_cube(
            output_dir / "predicted_density.cube",
            prediction,
            grid,
            snapshot.density.atoms,
            snapshot.positions,
            comment="Predicted electron density, input normalization (e/bohr^3)",
        )
        write_cube(
            output_dir / "signed_density_error.cube",
            error,
            grid,
            snapshot.density.atoms,
            snapshot.positions,
            comment="Predicted minus ground-truth electron density (e/bohr^3)",
        )
        write_cube(
            output_dir / "absolute_density_error.cube",
            error.abs(),
            grid,
            snapshot.density.atoms,
            snapshot.positions,
            comment="Absolute electron-density error (e/bohr^3)",
        )

        density_max = float(
            torch.quantile(
                torch.cat((ground_truth.flatten(), prediction.flatten())), 0.995
            )
        )
        error_limit = float(torch.quantile(error.abs().flatten(), 0.995))
        density_max = max(density_max, torch.finfo(ground_truth.dtype).eps)
        error_limit = max(error_limit, torch.finfo(error.dtype).eps)
        axis_names = "xyz"
        all_figures = []
        for axis, axis_name in enumerate(axis_names):
            index = grid.shape[axis] // 2
            slices = [
                ground_truth.select(axis, index).T,
                prediction.select(axis, index).T,
                error.select(axis, index).T,
            ]
            figure, axes = plt.subplots(
                1, 3, figsize=(15, 4.5), constrained_layout=True
            )
            for plot_axis, data, title in zip(
                axes, slices, ("Ground truth", "Prediction", "Prediction - GT")
            ):
                if title == "Prediction - GT":
                    image = plot_axis.imshow(
                        data.numpy(),
                        origin="lower",
                        cmap="bwr",
                        norm=TwoSlopeNorm(
                            vmin=-error_limit, vcenter=0.0, vmax=error_limit
                        ),
                        interpolation="nearest",
                        aspect="equal",
                    )
                else:
                    image = plot_axis.imshow(
                        data.numpy(),
                        origin="lower",
                        cmap="viridis",
                        vmin=0.0,
                        vmax=density_max,
                        interpolation="nearest",
                        aspect="equal",
                    )
                plot_axis.set_title(title)
                plot_axis.set_xlabel("grid index")
                plot_axis.set_ylabel("grid index")
                figure.colorbar(image, ax=plot_axis, shrink=0.82, label="e / bohr^3")
            figure.suptitle(f"Density slice: {axis_name} index {index}")
            figure.savefig(
                output_dir / f"density_slice_{axis_name}_{index:03d}.png", dpi=180
            )
            all_figures.append((axis_name, index, slices))
            plt.close(figure)

        overview, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
        for row, (axis_name, index, slices) in enumerate(all_figures):
            for column, (data, title) in enumerate(
                zip(slices, ("Ground truth", "Prediction", "Prediction - GT"))
            ):
                if column == 2:
                    image = axes[row, column].imshow(
                        data.numpy(),
                        origin="lower",
                        cmap="bwr",
                        norm=TwoSlopeNorm(
                            vmin=-error_limit, vcenter=0.0, vmax=error_limit
                        ),
                    )
                else:
                    image = axes[row, column].imshow(
                        data.numpy(),
                        origin="lower",
                        cmap="viridis",
                        vmin=0.0,
                        vmax=density_max,
                    )
                axes[row, column].set_title(f"{title}: {axis_name}={index}")
                overview.colorbar(
                    image, ax=axes[row, column], shrink=0.78, label="e / bohr^3"
                )
        overview.suptitle("Orthogonal electron-density slice triplets")
        overview.savefig(output_dir / "density_slice_triplets.png", dpi=180)
        plt.close(overview)

    (output_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ANGSTROM_TO_BOHR",
    "DensityGridResult",
    "OpenMXGrid",
    "PAORadialTable",
    "density_hermiticity_metrics",
    "evaluate_atomic_basis",
    "evaluate_density_grids",
    "load_openmx_snapshot",
    "parse_openmx_grid",
    "parse_openmx_pao",
    "real_spherical_harmonics_openmx",
    "write_cube",
    "write_density_outputs",
]
