"""Lossless native operator data, explicit graph supports and genuine batching.

HDF5 interchange follows [Rx,Ry,Rz,i,j] keys, one-based atoms, Angstrom and
eV, OpenMX orbital order. No Hermitian projection or target cleanup is applied.
The native layout also supports complex spin-major SOC blocks.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from ase.data import chemical_symbols

from data.bond_geometry import bond_frames, bond_neighborhoods
from net.operator_codecs import ShellTensorCodec


def read_native_metadata(directory, *, dtype=torch.float32):
    directory = Path(directory)
    zs = tuple(np.atleast_1d(np.loadtxt(directory / "element.dat", dtype=int)).tolist())
    atoms = tuple(chemical_symbols[z] for z in zs)
    shells = tuple(
        tuple(map(int, line.split()))
        for line in (directory / "orbital_types.dat").read_text().splitlines()
        if line.strip()
    )
    if len(shells) != len(atoms):
        raise ValueError("Orbital metadata must contain exactly one row per atom")
    species = {}
    for atom, ls in zip(atoms, shells):
        if atom in species and species[atom] != ls:
            raise ValueError(f"Inconsistent orbital layout for species {atom}")
        species[atom] = ls
    positions = torch.as_tensor(
        np.loadtxt(directory / "site_positions.dat").reshape(3, -1).T.copy(),
        dtype=dtype,
    )
    if positions.shape != (len(atoms), 3):
        raise ValueError("Native positions must be stored as 3 x number-of-atoms")
    box = torch.as_tensor(np.loadtxt(directory / "lat.dat").T.copy(), dtype=dtype)
    if box.shape != (3, 3):
        raise ValueError("Lattice must be a 3x3 matrix with native column vectors")
    if (
        not torch.isfinite(positions).all()
        or not torch.isfinite(box).all()
        or torch.linalg.det(box).abs() == 0
    ):
        raise ValueError("Native positions and nonsingular lattice must be finite")
    info = json.loads((directory / "info.json").read_text())
    return {
        "atoms": atoms,
        "species_shells": species,
        "positions": positions,
        "box": box,
        "spinful": bool(info.get("isspinful", False)),
    }


def _read_blocks(path, num_atoms, dtype):
    entries = {}
    with h5py.File(path) as handle:
        for raw, value in handle.items():
            key = ast.literal_eval(raw)
            if len(key) != 5 or any(
                not isinstance(v, int) or isinstance(v, bool) for v in key
            ):
                raise ValueError(f"Invalid periodic key: {raw}")
            r1, r2, r3, i, j = key
            if not (1 <= i <= num_atoms and 1 <= j <= num_atoms):
                raise ValueError(f"Native atom indices are one-based: {raw}")
            array = np.asarray(value)
            tensor = torch.as_tensor(array.copy())
            tensor = (
                tensor.to(
                    torch.complex128 if dtype == torch.float64 else torch.complex64
                )
                if tensor.is_complex()
                else tensor.to(dtype)
            )
            canonical = (r1, r2, r3, i - 1, j - 1)
            if canonical in entries:
                raise ValueError(f"Duplicate native periodic key: {raw}")
            entries[canonical] = tensor
    return entries


def load_operator_sample(
    directory,
    codec: ShellTensorCodec,
    *,
    species,
    split="train",
    sample_id=None,
    matrix_files=None,
    support_file="hamiltonians.h5",
    local_coordinates=False,
    num_l=5,
    frame_source="derive",
    target_frame="global",
    graph_source="stored",
    graph_radius=None,
    dtype=torch.float32,
):
    directory = Path(directory)
    meta = read_native_metadata(directory, dtype=dtype)
    if meta["spinful"] != codec.spinful:
        raise ValueError("Dataset spin layout and output codec disagree")
    if codec.species_shells is None:
        raise ValueError(
            "Operator samples require a codec with an explicit species shell layout"
        )
    for atom, shells in meta["species_shells"].items():
        if codec.species_shells.get(atom) != shells:
            raise ValueError(
                f"Native shell layout disagrees with the codec for species {atom}"
            )
    atoms, positions, box = meta["atoms"], meta["positions"], meta["box"]
    if set(atoms) - set(species):
        raise ValueError("Dataset contains species absent from the frozen vocabulary")
    if graph_source == "stored":
        entries = _read_blocks(directory / support_file, len(atoms), dtype)
        # Source builders group HDF5 iteration order by the first atom.
        keys = [key for i in range(len(atoms)) for key in entries if key[3] == i]
    elif graph_source == "radius":
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        if graph_radius is None or not np.isfinite(graph_radius) or graph_radius <= 0:
            raise ValueError("Radius graph requires a finite positive graph_radius")
        structure = Atoms(
            symbols=atoms, positions=positions.numpy(), cell=box.numpy(), pbc=True
        )
        source, destination, images = neighbor_list(
            "ijS", structure, graph_radius, self_interaction=True
        )
        keys = [
            tuple(map(int, (*r, i, j))) for i, j, r in zip(source, destination, images)
        ]
    else:
        raise ValueError("graph_source must be stored or radius")
    if not keys:
        raise ValueError("Operator support contains no edges")
    edges = torch.tensor(keys, dtype=torch.long).T
    index, shifts = edges[3:], edges[:3]
    vectors = positions[index[1]] + shifts.T.to(dtype) @ box - positions[index[0]]
    types = torch.tensor([species.index(a) for a in atoms], dtype=torch.long)
    pair_types = len(species) * types[index[0]] + types[index[1]]
    n_species = len(species)
    lengths = vectors.norm(dim=-1)
    selfloop = (index[0] == index[1]) & (shifts == 0).all(dim=0)
    message_mask = ~selfloop
    inputs = {
        "node_type_idx": types,
        "node_one_hot": torch.nn.functional.one_hot(types, n_species).to(dtype),
        "edge_type_idx": pair_types,
        "edge_one_hot": torch.nn.functional.one_hot(pair_types, n_species**2).to(dtype),
        "edge_index": index,
        "edge_shift": shifts,
        "edge_vectors": vectors,
        "edge_lengths": lengths,
        "node_batch": torch.zeros(len(atoms), dtype=torch.long),
        "edge_batch": torch.zeros(len(keys), dtype=torch.long),
        "selfloop_edge": selfloop,
        "message_edge_index": index[:, message_mask],
        "message_vectors": vectors[message_mask],
        "message_lengths": lengths[message_mask],
        "message_to_output": torch.where(message_mask)[0],
        "positions": positions,
        "box": box,
        "atoms": atoms,
        "edge_keys": edges,
    }
    frames = None
    if local_coordinates or target_frame == "bond":
        if frame_source == "stored":
            raw_frames = _read_blocks(directory / "rc.h5", len(atoms), dtype)
            frames = torch.stack([raw_frames[key] for key in keys])
        elif frame_source == "derive":
            # The native preprocessing derives frames in the original text
            # precision before casting them to the network dtype. Preserve
            # that order so float32 rounding cannot introduce distance ties.
            raw = read_native_metadata(directory, dtype=torch.float64)
            frame_vectors = (
                raw["positions"][index[1]]
                + shifts.T.to(torch.float64) @ raw["box"]
                - raw["positions"][index[0]]
            )
            frames = bond_frames(index, frame_vectors).to(dtype)
        else:
            raise ValueError("frame_source must be stored or derive")
        if frames.shape != (len(keys), 3, 3) or not torch.isfinite(frames).all():
            raise ValueError("Stored bond frames must be finite 3x3 matrices")
        identity = torch.eye(3, dtype=frames.dtype, device=frames.device)
        if (
            not torch.allclose(
                frames.transpose(-1, -2) @ frames,
                identity.expand_as(frames),
                atol=1e-5,
                rtol=1e-5,
            )
            or (torch.linalg.det(frames) <= 0).any()
        ):
            raise ValueError("Bond frames must be proper orthonormal matrices")
        inputs["bond_frames"] = frames
        if local_coordinates:
            inputs.update(bond_neighborhoods(index, vectors, frames, num_l=num_l))
    if target_frame not in {"global", "bond"}:
        raise ValueError("target_frame must be global or bond")
    targets, full_blocks = {}, {}
    matrix_files = (
        {"hamiltonian": "hamiltonians.h5"} if matrix_files is None else matrix_files
    )
    for name, filename in matrix_files.items():
        native = _read_blocks(directory / filename, len(atoms), dtype)
        values, masks = [], []
        for e, key in enumerate(keys):
            i, j = key[3:]
            pair = (atoms[i], atoms[j])
            dimensions = [
                sum(2 * l + 1 for l in meta["species_shells"][atom])
                * (2 if codec.spinful else 1)
                for atom in pair
            ]
            block = native.get(key)
            if block is None:
                block = torch.zeros(
                    *dimensions,
                    dtype=(
                        torch.complex128
                        if codec.spinful and dtype == torch.float64
                        else torch.complex64 if codec.spinful else dtype
                    ),
                )
            if tuple(block.shape) != tuple(dimensions):
                raise ValueError(
                    f"Wrong native block shape for {key}: {block.shape}, expected {dimensions}"
                )
            if target_frame == "bond":
                block = codec.rotation.rotate_blocks(
                    block,
                    frames[e],
                    meta["species_shells"][pair[0]],
                    meta["species_shells"][pair[1]],
                )
            packed, mask = codec.pack_blocks(pair, block, return_mask=True)
            if key not in native:
                mask = torch.zeros_like(mask)
            # Missing values never enter the coupled statistics or loss through
            # zero*NaN. A partial shell remains usable in matrix-space loss;
            # codec.encode_mask conservatively excludes it from fitted stats.
            mask = mask & torch.isfinite(packed)
            packed = torch.where(mask, packed, torch.zeros_like(packed))
            values.append(packed)
            masks.append(mask)
        targets[name] = {"values": torch.stack(values), "mask": torch.stack(masks)}
        full_blocks[name] = native
    sid = str(sample_id if sample_id is not None else directory.resolve())
    return {
        "id": sid,
        "split": split,
        "inputs": inputs,
        "targets": targets,
        "source": str(directory.resolve()),
        "keys": keys,
        "full_blocks": full_blocks,
        "species_shells": meta["species_shells"],
        "target_frame": target_frame,
    }


def collate_operator_samples(samples):
    """Disjoint union with independent offsets for atoms, edges and local relations."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    first_inputs = (
        samples[0]["inputs"] if isinstance(samples[0], dict) else samples[0][0]
    )
    first_targets = (
        samples[0]["targets"] if isinstance(samples[0], dict) else samples[0][1]
    )
    xs, ys = {}, {}
    node_offset, edge_offset = 0, 0
    skip = {"box", "positions", "atoms", "edge_keys"}
    for graph, sample in enumerate(samples):
        x = sample["inputs"] if "inputs" in sample else sample[0]
        targets = sample["targets"] if "targets" in sample else sample[1]
        if x.keys() != first_inputs.keys() or targets.keys() != first_targets.keys():
            raise ValueError(
                "All graphs in a batch must have identical feature and target schemas"
            )
        if any(
            record.keys() != first_targets[name].keys()
            for name, record in targets.items()
        ):
            raise ValueError(
                "All target records in a batch must have identical schemas"
            )
        for key, value in x.items():
            if key in skip:
                continue
            if key in {"edge_index", "message_edge_index", "sub_atom_idx"}:
                value = value + node_offset
            elif key in {"sub_edge_idx", "message_to_output"}:
                value = value + edge_offset
            elif key == "sub_index":
                value = value + 2 * edge_offset
            elif key in {"node_batch", "edge_batch"}:
                value = torch.full_like(value, graph)
            xs.setdefault(key, []).append(value)
        for name, record in targets.items():
            for key, value in record.items():
                ys.setdefault(name, {}).setdefault(key, []).append(value)
        node_offset += len(x["node_type_idx"])
        edge_offset += len(x["edge_lengths"])
    for key, values in xs.items():
        xs[key] = torch.cat(
            values,
            dim=1 if key in {"edge_index", "message_edge_index", "edge_shift"} else 0,
        )
    ys = {
        name: {key: torch.cat(values) for key, values in record.items()}
        for name, record in ys.items()
    }
    return xs, ys


class OperatorDataset(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def statistics_records(samples, codec, *, target="hamiltonian"):
    for sample in samples:
        if sample["split"] != "train":
            raise ValueError("Statistics cannot access validation or test samples")
        record = sample["targets"][target]
        yield {
            "id": sample["id"],
            "split": "train",
            "pair_type": sample["inputs"]["edge_type_idx"],
            "coupled": codec.encode(
                torch.where(
                    record["mask"], record["values"], torch.zeros_like(record["values"])
                )
            ),
            "mask": codec.encode_mask(record["mask"]),
        }


def reconstruct_operator_blocks(
    sample, codec, packed_values, *, output_frame="global", return_masks=False
):
    """Recover native directed full blocks for evaluation/export.

    Predictions are packed shell entries (``codec.decode`` output). Unselected
    shell blocks remain zero and are explicitly marked false in returned
    coverage masks. This operation never projects, transposes, or averages
    reverse edges. SOC blocks retain their complex spin-major native layout.
    """
    if output_frame not in {"global", "bond"}:
        raise ValueError("output_frame must be global or bond")
    if len(packed_values) != len(sample["keys"]):
        raise ValueError("Prediction rows must match the sample output edge keys")
    source_frame = sample["target_frame"]
    frames = sample["inputs"].get("bond_frames")
    if source_frame != output_frame and frames is None:
        raise ValueError("Changing output frame requires explicit sample bond frames")
    blocks, masks = {}, {}
    atoms = sample["inputs"]["atoms"]
    for edge, key in enumerate(sample["keys"]):
        pair = (atoms[key[3]], atoms[key[4]])
        block = codec.unpack_blocks(pair, packed_values[edge])
        coverage = codec.unpack_blocks(
            pair, codec.packed_mask(pair, device=block.device)
        )
        if source_frame != output_frame:
            frame = frames[edge]
            if output_frame == "global":
                frame = frame.T
            block = codec.rotation.rotate_blocks(
                block,
                frame,
                sample["species_shells"][pair[0]],
                sample["species_shells"][pair[1]],
            )
        blocks[key], masks[key] = block, coverage
    return (blocks, masks) if return_masks else blocks
