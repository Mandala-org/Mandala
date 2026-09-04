"""Canonical sharded cache construction for the released HamGNN SiO2 graphs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from ase import Atoms
from ase.neighborlist import neighbor_list
import h5py
import numpy as np
import torch

from core.basis_converter import OpenMXE3NNConverter
from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_descriptors import AtomicNeighborDensity
from pair_hamiltonian.hamgnn_sio2 import ACTIVE_AO_14, BOHR_TO_ANGSTROM, SPECIES_BY_Z
from pair_hamiltonian.output_schema import FullBlockIrrepTransform

CANONICAL_PAIR_NAMES = ("O-O", "O-Si", "Si-Si")
PAIR_TO_INDEX = {name: index for index, name in enumerate(CANONICAL_PAIR_NAMES)}


@dataclass(frozen=True, slots=True)
class CachedStructureSummary:
    structure_index: int
    source_key: int
    split: str
    atom_count: int
    offsite_pair_count: int
    neighbor_count: int
    size_bytes: int
    sha256: str


def split_hash(split_indices: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("train", "validation", "test"):
        values = np.asarray(split_indices[name], dtype="<i8")
        digest.update(name.encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _geometry_neighbors(
    atomic_numbers: np.ndarray,
    positions_angstrom: np.ndarray,
    cell_angstrom: np.ndarray,
    cutoff_angstrom: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions_angstrom,
        cell=cell_angstrom,
        pbc=True,
    )
    center, neighbor, image = neighbor_list(
        "ijS", atoms, cutoff_angstrom, self_interaction=False
    )
    order = np.lexsort((image[:, 2], image[:, 1], image[:, 0], neighbor, center))
    center = center[order].astype(np.int64, copy=False)
    neighbor = neighbor[order].astype(np.int64, copy=False)
    image = image[order].astype(np.int16, copy=False)
    displacement = (
        positions_angstrom[neighbor]
        + image.astype(np.float64) @ cell_angstrom
        - positions_angstrom[center]
    )
    return center, neighbor, image, displacement


def _target_irreps(
    blocks_openmx: torch.Tensor,
    pair_indices: np.ndarray,
    transform: FullBlockIrrepTransform,
    converter: OpenMXE3NNConverter,
) -> np.ndarray:
    result = torch.empty(blocks_openmx.shape[0], 169, dtype=torch.float64, device="cpu")
    for pair_index, pair_name in enumerate(CANONICAL_PAIR_NAMES):
        selected = np.flatnonzero(pair_indices == pair_index)
        if selected.size == 0:
            continue
        index = torch.from_numpy(selected)
        blocks = blocks_openmx.index_select(0, index).to(torch.float64)
        e3nn_blocks = converter.block_openmx_to_e3nn(pair_name, blocks)
        result.index_copy_(0, index, transform.blocks_to_irreps(pair_name, e3nn_blocks))
    return result.to(torch.float32).numpy()


def build_structure_arrays(
    graph: object,
    *,
    descriptor_cutoff_angstrom: float,
    density_radial_count: int,
    density_l_max: int,
    hamiltonian_cutoff_angstrom: float,
    nao_max: int = 14,
) -> dict[str, np.ndarray]:
    """Convert one released graph into one canonical full-block cache shard."""
    atomic_numbers = graph.z.detach().cpu().numpy().astype(np.int16, copy=False)
    positions_bohr = graph.pos.detach().cpu().numpy().astype(np.float64, copy=False)
    raw_cell = graph.cell.detach().cpu().numpy().astype(np.float64, copy=False)
    cell_bohr = raw_cell[0] if raw_cell.shape == (1, 3, 3) else raw_cell
    if cell_bohr.shape != (3, 3):
        raise ValueError(f"Expected cell shape (3, 3), got {raw_cell.shape}")
    positions_angstrom = positions_bohr * BOHR_TO_ANGSTROM
    cell_angstrom = cell_bohr * BOHR_TO_ANGSTROM
    center, neighbor, neighbor_image, neighbor_displacement = _geometry_neighbors(
        atomic_numbers,
        positions_angstrom,
        cell_angstrom,
        descriptor_cutoff_angstrom,
    )
    density = AtomicNeighborDensity(
        (8, 14),
        n_radial=density_radial_count,
        l_max=density_l_max,
        cutoff=descriptor_cutoff_angstrom,
    ).to(dtype=torch.float64)
    descriptors = density(
        torch.from_numpy(neighbor_displacement),
        torch.from_numpy(atomic_numbers[neighbor].astype(np.int64)),
        torch.from_numpy(center),
        num_centers=atomic_numbers.size,
    ).to(torch.float32)

    edge_index = graph.edge_index.detach().cpu().to(torch.long)
    inverse = graph.inv_edge_idx.detach().cpu().to(torch.long)
    edge_count = edge_index.shape[1]
    if inverse.shape != (edge_count,) or not torch.equal(
        inverse[inverse], torch.arange(edge_count)
    ):
        raise ValueError("Invalid inverse-edge mapping")
    representatives = torch.arange(edge_count)[torch.arange(edge_count) < inverse]
    source_all, target_all = edge_index
    rank = {8: 0, 14: 1}
    canonical_edges: list[int] = []
    for edge in representatives.tolist():
        source_z = int(graph.z[source_all[edge]])
        target_z = int(graph.z[target_all[edge]])
        canonical_edges.append(
            edge if rank[source_z] <= rank[target_z] else int(inverse[edge])
        )
    selected = torch.tensor(canonical_edges, dtype=torch.long)
    source = source_all.index_select(0, selected)
    target = target_all.index_select(0, selected)
    displacement = (
        graph.pos[target]
        - graph.pos[source]
        + graph.nbr_shift.index_select(0, selected)
    ).to(torch.float64) * BOHR_TO_ANGSTROM
    keep = torch.linalg.vector_norm(displacement, dim=1) <= (
        hamiltonian_cutoff_angstrom + 1.0e-7
    )
    selected = selected[keep]
    source = source[keep]
    target = target[keep]
    displacement = displacement[keep]
    image = graph.cell_shift.detach().cpu().index_select(0, selected).to(torch.int16)
    source_z = graph.z.detach().cpu().index_select(0, source)
    target_z = graph.z.detach().cpu().index_select(0, target)
    pair_names = [
        f"{SPECIES_BY_Z[int(a)]}-{SPECIES_BY_Z[int(b)]}"
        for a, b in zip(source_z, target_z)
    ]
    pair_indices = np.asarray(
        [PAIR_TO_INDEX[name] for name in pair_names], dtype=np.uint8
    )

    active = torch.as_tensor(ACTIVE_AO_14[8], dtype=torch.long)
    packed = graph.hamiltonian.detach().cpu()
    atom_count = atomic_numbers.size
    matrices = packed.reshape(-1, nao_max, nao_max)
    onsite_blocks = (
        matrices[:atom_count].index_select(1, active).index_select(2, active)
    )
    offsite_blocks = (
        matrices[atom_count:]
        .index_select(0, selected)
        .index_select(1, active)
        .index_select(2, active)
    )
    orbital_config = OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"})
    transform = FullBlockIrrepTransform(orbital_config, dtype=torch.float64)
    converter = OpenMXE3NNConverter(orbital_config)
    onsite_pair_indices = np.asarray(
        [
            PAIR_TO_INDEX[f"{SPECIES_BY_Z[int(z)]}-{SPECIES_BY_Z[int(z)]}"]
            for z in atomic_numbers
        ],
        dtype=np.uint8,
    )
    onsite_targets = _target_irreps(
        onsite_blocks, onsite_pair_indices, transform, converter
    )
    offsite_targets = _target_irreps(offsite_blocks, pair_indices, transform, converter)
    return {
        "atomic_numbers": atomic_numbers,
        "positions_angstrom": positions_angstrom.astype(np.float32),
        "cell_angstrom": cell_angstrom.astype(np.float32),
        "descriptor": descriptors.numpy(),
        "onsite_target_irreps_hartree": onsite_targets,
        "offsite_source": source.numpy().astype(np.int32),
        "offsite_target": target.numpy().astype(np.int32),
        "offsite_image": image.numpy(),
        "offsite_displacement_angstrom": displacement.numpy().astype(np.float32),
        "offsite_pair_type": pair_indices,
        "offsite_target_irreps_hartree": offsite_targets,
        "neighbor_center": center.astype(np.int32),
        "neighbor_atom": neighbor.astype(np.int32),
        "neighbor_image": neighbor_image,
        "neighbor_displacement_angstrom": neighbor_displacement.astype(np.float32),
    }


def write_structure_shard(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    structure_index: int,
    source_key: int,
    split: str,
    metadata: Mapping[str, object],
) -> CachedStructureSummary:
    """Atomically write and checksum one independently reusable HDF5 shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["complete"] = False
            handle.attrs["structure_index"] = structure_index
            handle.attrs["source_key"] = source_key
            handle.attrs["split"] = split
            handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            for name, values in arrays.items():
                handle.create_dataset(
                    name, data=values, compression="gzip", compression_opts=1
                )
            handle.attrs["complete"] = True
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return CachedStructureSummary(
        structure_index=structure_index,
        source_key=source_key,
        split=split,
        atom_count=int(arrays["atomic_numbers"].size),
        offsite_pair_count=int(arrays["offsite_source"].size),
        neighbor_count=int(arrays["neighbor_center"].size),
        size_bytes=path.stat().st_size,
        sha256=digest,
    )


def validate_structure_shard(path: Path, metadata: Mapping[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return (
                bool(handle.attrs.get("complete", False))
                and json.loads(handle.attrs["metadata_json"]) == metadata
            )
    except (OSError, KeyError, json.JSONDecodeError):
        return False
