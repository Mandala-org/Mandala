"""Validation and inventory helpers for the released HamGNN SiO2 graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from core.basis_converter import OpenMXE3NNConverter
from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_hamiltonian.output_schema import FullBlockIrrepTransform

BOHR_TO_ANGSTROM = 0.5291772083
HARTREE_TO_MEV = 27_211.386245988
SPECIES_BY_Z = {8: "O", 14: "Si"}
ACTIVE_AO_14 = {
    8: np.asarray([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    14: np.asarray([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
}


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def membership(self, size: int) -> np.ndarray:
        result = np.full(size, -1, dtype=np.int8)
        result[self.train] = 0
        result[self.validation] = 1
        result[self.test] = 2
        if np.any(result < 0):
            raise ValueError("Split indices do not cover every structure exactly once")
        return result


def published_protocol_split(
    size: int,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> SplitIndices:
    """Mirror HamGNN's RandomState shuffle and rounded 80/10/10 slicing."""
    if size <= 0:
        raise ValueError("Dataset must contain at least one structure")
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must be positive and sum to less than one")
    indices = list(range(size))
    np.random.RandomState(seed=seed).shuffle(indices)
    n_train = round(train_ratio * size)
    n_validation = round(validation_ratio * size)
    return SplitIndices(
        train=np.asarray(indices[:n_train], dtype=np.int64),
        validation=np.asarray(
            indices[n_train : n_train + n_validation], dtype=np.int64
        ),
        test=np.asarray(indices[n_train + n_validation :], dtype=np.int64),
    )


def unpack_active_blocks(
    packed: torch.Tensor,
    row_z: torch.Tensor,
    column_z: torch.Tensor,
    *,
    nao_max: int = 14,
) -> torch.Tensor:
    """Remove HamGNN padding from a homogeneous SiO2 block collection."""
    if packed.ndim != 2 or packed.shape[1] != nao_max * nao_max:
        raise ValueError(
            f"Expected packed blocks (*, {nao_max**2}), got {tuple(packed.shape)}"
        )
    if row_z.shape != column_z.shape or row_z.numel() != packed.shape[0]:
        raise ValueError("Species arrays must have one entry per packed block")
    if not torch.equal(row_z, row_z[:1].expand_as(row_z)) or not torch.equal(
        column_z, column_z[:1].expand_as(column_z)
    ):
        raise ValueError("unpack_active_blocks expects one homogeneous species pair")
    row_indices = torch.as_tensor(ACTIVE_AO_14[int(row_z[0])], device=packed.device)
    column_indices = torch.as_tensor(
        ACTIVE_AO_14[int(column_z[0])], device=packed.device
    )
    matrices = packed.reshape(-1, nao_max, nao_max)
    return matrices.index_select(1, row_indices).index_select(2, column_indices)


def _tail_cutoff(
    distance: np.ndarray,
    absolute_sum: np.ndarray,
    squared_sum: np.ndarray,
    *,
    elements_per_block: int,
    bin_width_angstrom: float,
    mae_limit_mev: float,
    retained_squared_fraction: float,
) -> dict[str, float | bool]:
    max_edge = float(distance.max(initial=0.0))
    upper_edges = np.arange(
        bin_width_angstrom, max_edge + bin_width_angstrom * 1.5, bin_width_angstrom
    )
    bin_index = np.minimum(
        np.floor(distance / bin_width_angstrom).astype(np.int64), len(upper_edges) - 1
    )
    bin_abs = np.bincount(bin_index, weights=absolute_sum, minlength=len(upper_edges))
    bin_sq = np.bincount(bin_index, weights=squared_sum, minlength=len(upper_edges))
    bin_blocks = np.bincount(bin_index, minlength=len(upper_edges))
    omitted_abs = np.cumsum(bin_abs[::-1])[::-1] - bin_abs
    omitted_sq = np.cumsum(bin_sq[::-1])[::-1] - bin_sq
    total_sq = float(bin_sq.sum())
    denominator = max(1, int(bin_blocks.sum()) * elements_per_block)
    omitted_mae = omitted_abs / denominator * HARTREE_TO_MEV
    retained = 1.0 - omitted_sq / total_sq if total_sq > 0 else np.ones_like(omitted_sq)
    valid = (omitted_mae <= mae_limit_mev) & (retained >= retained_squared_fraction)
    if not np.any(valid):
        return {
            "found": False,
            "cutoff_angstrom": math.nan,
            "omitted_mae_mev": float(omitted_mae[-1]),
            "retained_squared_fraction": float(retained[-1]),
            "maximum_observed_distance_angstrom": max_edge,
        }
    selected = int(np.flatnonzero(valid)[0])
    return {
        "found": True,
        "cutoff_angstrom": float(upper_edges[selected]),
        "omitted_mae_mev": float(omitted_mae[selected]),
        "retained_squared_fraction": float(retained[selected]),
        "maximum_observed_distance_angstrom": max_edge,
    }


def inspect_graphs(
    graphs: Mapping[int, object] | Sequence[object],
    split: SplitIndices,
    *,
    nao_max: int = 14,
    distance_bin_width_angstrom: float = 0.05,
    benchmark_mae_mev: float = 2.29,
    absolute_mae_limit_mev: float = 0.1,
    retained_squared_fraction: float = 0.999,
    descriptor_cutoffs_angstrom: Iterable[float] = (4.0, 6.0, 8.0, 10.0),
    show_progress: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Validate graphs and return a JSON-ready summary plus distance-bin rows."""
    # dict insertion order is scientifically relevant: HamGNN loads the release
    # with ``list(graph_data.values())`` before generating split indices.
    ordered = list(graphs.values()) if isinstance(graphs, Mapping) else list(graphs)
    membership = split.membership(len(ordered))
    orbital_config = OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"})
    converter = OpenMXE3NNConverter(orbital_config)
    transform = FullBlockIrrepTransform(orbital_config, dtype=torch.float64)

    atom_counts: list[int] = []
    edge_counts: list[int] = []
    species_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    bins: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {
            "blocks": 0.0,
            "abs_sum_hartree": 0.0,
            "sq_sum_hartree2": 0.0,
            "frobenius_sum_hartree": 0.0,
            "max_abs_hartree": 0.0,
        }
    )
    train_distance: list[np.ndarray] = []
    train_absolute_sum: list[np.ndarray] = []
    train_squared_sum: list[np.ndarray] = []
    periodic_shift_max_error = 0.0
    inverse_index_failures = 0
    transpose_max_error = 0.0
    conversion_max_error = 0.0
    conversion_validated_pairs: set[str] = set()
    descriptor_neighbor_counts = {
        float(cutoff): 0 for cutoff in descriptor_cutoffs_angstrom
    }

    iterator = tqdm(
        ordered, desc="structures", unit="structure", disable=not show_progress
    )
    for structure_index, graph in enumerate(iterator):
        required = (
            "z",
            "pos",
            "cell",
            "edge_index",
            "inv_edge_idx",
            "nbr_shift",
            "cell_shift",
            "hamiltonian",
        )
        missing = [name for name in required if not hasattr(graph, name)]
        if missing:
            raise ValueError(f"Structure {structure_index} lacks fields {missing}")
        z = graph.z.detach().cpu().to(torch.long)
        pos = graph.pos.detach().cpu().to(torch.float64)
        cell = graph.cell.detach().cpu().to(torch.float64).reshape(3, 3)
        edge_index = graph.edge_index.detach().cpu().to(torch.long)
        inverse = graph.inv_edge_idx.detach().cpu().to(torch.long)
        shifts = graph.nbr_shift.detach().cpu().to(torch.float64)
        cell_shift = graph.cell_shift.detach().cpu().to(torch.float64)
        packed = graph.hamiltonian.detach().cpu()
        n_atoms = z.numel()
        n_edges = edge_index.shape[1]
        if edge_index.shape != (2, n_edges) or inverse.shape != (n_edges,):
            raise ValueError(
                f"Structure {structure_index} has inconsistent edge arrays"
            )
        if packed.shape != (n_atoms + n_edges, nao_max * nao_max):
            raise ValueError(
                f"Structure {structure_index} has Hamiltonian shape {tuple(packed.shape)}"
            )
        unknown = sorted(set(z.tolist()) - set(SPECIES_BY_Z))
        if unknown:
            raise ValueError(
                f"Structure {structure_index} contains unsupported Z={unknown}"
            )
        atom_counts.append(n_atoms)
        edge_counts.append(n_edges)
        species_counts.update(SPECIES_BY_Z[int(value)] for value in z.tolist())

        reconstructed_shifts = cell_shift @ cell
        periodic_shift_max_error = max(
            periodic_shift_max_error,
            float((reconstructed_shifts - shifts).abs().max().item()),
        )
        expected = torch.arange(n_edges)
        inverse_index_failures += int((inverse[inverse] != expected).sum().item())
        source, target = edge_index
        reverse_source, reverse_target = edge_index[:, inverse]
        inverse_index_failures += int(
            ((reverse_source != target) | (reverse_target != source)).sum().item()
        )
        displacement_bohr = pos[target] - pos[source] + shifts
        distance_angstrom = (
            torch.linalg.vector_norm(displacement_bohr, dim=1) * BOHR_TO_ANGSTROM
        )
        offsite_packed = packed[n_atoms:]
        # O and Si share the same 13 active positions in this nao_max=14
        # release, so unpack and validate reversal once per structure.
        active = torch.as_tensor(ACTIVE_AO_14[8])
        active_blocks = (
            offsite_packed.reshape(-1, nao_max, nao_max)
            .index_select(1, active)
            .index_select(2, active)
            .to(torch.float64)
        )
        transpose_max_error = max(
            transpose_max_error,
            float(
                (active_blocks - active_blocks[inverse].transpose(-1, -2))
                .abs()
                .max()
                .item()
            ),
        )

        for row_z_value in SPECIES_BY_Z:
            for column_z_value in SPECIES_BY_Z:
                mask = (z[source] == row_z_value) & (z[target] == column_z_value)
                if not torch.any(mask):
                    continue
                pair = f"{SPECIES_BY_Z[row_z_value]}-{SPECIES_BY_Z[column_z_value]}"
                pair_counts[pair] += int(mask.sum().item())
                blocks = active_blocks[mask]
                if pair not in conversion_validated_pairs:
                    converted = converter.block_openmx_to_e3nn(pair, blocks[:1])
                    restored = transform.irreps_to_blocks(
                        pair, transform.blocks_to_irreps(pair, converted)
                    )
                    conversion_max_error = max(
                        conversion_max_error,
                        float((restored - converted).abs().max().item()),
                    )
                    conversion_validated_pairs.add(pair)
                distances = distance_angstrom[mask].numpy()
                abs_sum = blocks.abs().sum(dim=(-2, -1)).numpy()
                sq_sum = blocks.square().sum(dim=(-2, -1)).numpy()
                frobenius = blocks.square().sum(dim=(-2, -1)).sqrt().numpy()
                max_abs = blocks.abs().amax(dim=(-2, -1)).numpy()
                indices = np.floor(distances / distance_bin_width_angstrom).astype(
                    np.int64
                )
                for bin_index in np.unique(indices):
                    selected = indices == bin_index
                    record = bins[(pair, int(bin_index))]
                    record["blocks"] += int(selected.sum())
                    record["abs_sum_hartree"] += float(abs_sum[selected].sum())
                    record["sq_sum_hartree2"] += float(sq_sum[selected].sum())
                    record["frobenius_sum_hartree"] += float(frobenius[selected].sum())
                    record["max_abs_hartree"] = max(
                        record["max_abs_hartree"], float(max_abs[selected].max())
                    )
                if membership[structure_index] == 0:
                    train_distance.append(distances)
                    train_absolute_sum.append(abs_sum)
                    train_squared_sum.append(sq_sum)

        # Geometry-only descriptor-neighbor count, independent of Hamiltonian sparsity.
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        atoms = Atoms(
            numbers=z.numpy(),
            positions=pos.numpy() * BOHR_TO_ANGSTROM,
            cell=cell.numpy() * BOHR_TO_ANGSTROM,
            pbc=True,
        )
        max_descriptor_cutoff = max(descriptor_neighbor_counts, default=0.0)
        if max_descriptor_cutoff > 0:
            descriptor_distances = neighbor_list(
                "d", atoms, max_descriptor_cutoff, self_interaction=False
            )
            for cutoff in descriptor_neighbor_counts:
                descriptor_neighbor_counts[cutoff] += int(
                    np.count_nonzero(descriptor_distances < cutoff)
                )

    distances = np.concatenate(train_distance)
    absolute_sum = np.concatenate(train_absolute_sum)
    squared_sum = np.concatenate(train_squared_sum)
    mae_limit = min(absolute_mae_limit_mev, 0.05 * benchmark_mae_mev)
    cutoff = _tail_cutoff(
        distances,
        absolute_sum,
        squared_sum,
        elements_per_block=13 * 13,
        bin_width_angstrom=distance_bin_width_angstrom,
        mae_limit_mev=mae_limit,
        retained_squared_fraction=retained_squared_fraction,
    )
    bin_rows: list[dict[str, object]] = []
    for (pair, bin_index), record in sorted(bins.items()):
        count = int(record["blocks"])
        bin_rows.append(
            {
                "species_pair": pair,
                "distance_lower_angstrom": bin_index * distance_bin_width_angstrom,
                "distance_upper_angstrom": (bin_index + 1)
                * distance_bin_width_angstrom,
                "block_count": count,
                "mean_frobenius_hartree": record["frobenius_sum_hartree"] / count,
                "max_element_abs_hartree": record["max_abs_hartree"],
                "mean_element_abs_mev": record["abs_sum_hartree"]
                / (count * 169)
                * HARTREE_TO_MEV,
                "squared_frobenius_mass_hartree2": record["sq_sum_hartree2"],
            }
        )
    validation_passed = (
        periodic_shift_max_error <= 1.0e-5
        and inverse_index_failures == 0
        and transpose_max_error <= 1.0e-7
        and conversion_max_error <= 1.0e-10
        and conversion_validated_pairs == {"O-O", "O-Si", "Si-O", "Si-Si"}
    )
    summary: dict[str, object] = {
        "dataset": "HamGNN SiO2",
        "structure_count": len(ordered),
        "expected_structure_count_from_publication": 663,
        "publication_count_matches": len(ordered) == 663,
        "atom_count": {
            "total": sum(atom_counts),
            "min": min(atom_counts),
            "max": max(atom_counts),
        },
        "directed_offsite_block_count": sum(edge_counts),
        "species_atom_counts": dict(sorted(species_counts.items())),
        "species_pair_counts": dict(sorted(pair_counts.items())),
        "basis": {
            "source": "OpenMX",
            "packed_nao_max": nao_max,
            "active_indices": ACTIVE_AO_14[8].tolist(),
            "active_dimension": 13,
            "orbitals": "2s2p1d",
            "target_ordering": "e3nn real spherical harmonics",
        },
        "units": {
            "positions_and_cells_stored": "bohr",
            "reported_distances": "angstrom",
            "hamiltonian": "hartree",
        },
        "split": {
            "status": "protocol-matched, split-not-identical",
            "algorithm": "numpy RandomState(seed).shuffle; round(0.8*N), round(0.1*N)",
            "sizes": {
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
        },
        "hamiltonian_cutoff": {
            **cutoff,
            "fit_partition": "train",
            "distance_bin_width_angstrom": distance_bin_width_angstrom,
            "mae_limit_mev": mae_limit,
            "required_retained_squared_fraction": retained_squared_fraction,
            "support_caveat": "criteria are evaluated over offsite blocks present in the released graph",
        },
        "validation": {
            "passed": validation_passed,
            "thresholds": {
                "periodic_shift_max_abs_error_bohr": 1.0e-5,
                "hamiltonian_transpose_max_abs_error_hartree": 1.0e-7,
                "ao_irrep_roundtrip_max_abs_error_hartree": 1.0e-10,
            },
            "periodic_shift_max_abs_error_bohr": periodic_shift_max_error,
            "inverse_edge_failures": inverse_index_failures,
            "hamiltonian_transpose_max_abs_error_hartree": transpose_max_error,
            "ao_irrep_roundtrip_max_abs_error_hartree": conversion_max_error,
            "ao_irrep_roundtrip_validated_pairs": sorted(conversion_validated_pairs),
        },
        "descriptor_neighbor_storage_estimates": {
            str(cutoff): {
                "directed_neighbor_records": count,
                "estimated_bytes_at_32_bytes_per_record": count * 32,
            }
            for cutoff, count in sorted(descriptor_neighbor_counts.items())
        },
        "output_irrep_schemas": transform.schema_manifest(),
    }
    return summary, bin_rows
