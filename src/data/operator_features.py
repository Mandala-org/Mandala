"""Lossless native operator data, explicit graph supports and genuine batching.

HDF5 interchange follows [Rx,Ry,Rz,i,j] keys, one-based atoms, Angstrom and
eV, OpenMX orbital order. No Hermitian projection or target cleanup is applied.
The native layout also supports complex spin-major SOC blocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.bond_geometry import bond_frames
from core.operator_layout import ShellTensorCodec


def read_native_metadata(directory, *, dtype=torch.float32):
    """Read the common Snapshot metadata format without a second parser."""
    from data.deeph_e3_parser import load_deeph_e3_metadata

    meta = load_deeph_e3_metadata(directory, dtype=dtype, require_matrices=False)
    species = dict(zip(meta.atoms, meta.orbital_types))
    if (
        not torch.isfinite(meta.positions).all()
        or not torch.isfinite(meta.box).all()
        or torch.linalg.det(meta.box).abs() == 0
    ):
        raise ValueError("Positions and nonsingular lattice must be finite")
    return {
        "atoms": meta.atoms,
        "species_shells": species,
        "positions": meta.positions,
        "box": meta.box,
        "spinful": meta.is_spinful,
    }


def _read_blocks(path, num_atoms, dtype):
    from data.deeph_e3_parser import read_periodic_blocks

    return read_periodic_blocks(path, num_atoms=num_atoms, dtype=dtype)


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
    supports=None,
    neighborhood_support="output",
):
    directory = Path(directory)
    if frame_source not in {"stored", "derive"}:
        raise ValueError("frame_source must be stored or derive")
    meta = read_native_metadata(directory, dtype=dtype)
    matrix_files = (
        {"hamiltonian": "hamiltonians.h5"} if matrix_files is None else matrix_files
    )
    blocks = {
        name: _read_blocks(directory / filename, len(meta["atoms"]), dtype)
        for name, filename in matrix_files.items()
    }
    support = (
        _read_blocks(directory / support_file, len(meta["atoms"]), dtype)
        if graph_source == "stored"
        else {}
    )
    frame_meta = read_native_metadata(directory, dtype=torch.float64)
    stored_frames = (
        _read_blocks(directory / "rc.h5", len(meta["atoms"]), dtype)
        if frame_source == "stored" and (local_coordinates or target_frame == "bond")
        else None
    )
    return build_operator_sample(
        meta,
        blocks,
        support,
        codec,
        species=species,
        split=split,
        sample_id=sample_id if sample_id is not None else str(directory.resolve()),
        local_coordinates=local_coordinates,
        num_l=num_l,
        target_frame=target_frame,
        graph_source=graph_source,
        graph_radius=graph_radius,
        dtype=dtype,
        frame_meta=frame_meta,
        stored_frames=stored_frames,
        supports=supports,
        neighborhood_support=neighborhood_support,
    )


def build_operator_sample(
    meta,
    matrices,
    support,
    codec,
    *,
    species,
    split="train",
    sample_id="snapshot",
    local_coordinates=False,
    num_l=5,
    target_frame="global",
    graph_source="stored",
    graph_radius=None,
    dtype=torch.float32,
    frame_meta=None,
    stored_frames=None,
    supports=None,
    neighborhood_support="output",
):
    """Format-independent graph/target preprocessing on already parsed tensors."""
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
        entries = support
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
    from data.graph_topology import prepare_graph_topology

    # Support policies select relations once, before differentiable geometry.
    if supports:
        for key in (
            "message_edge_index",
            "message_vectors",
            "message_lengths",
            "message_to_output",
        ):
            inputs.pop(key, None)
    prepare_graph_topology(inputs, supports)
    frames = None
    if local_coordinates or target_frame == "bond":
        if stored_frames is not None:
            frames = torch.stack([stored_frames[key] for key in keys])
        else:
            # The native preprocessing derives frames in the original text
            # precision before casting them to the network dtype. Preserve
            # that order so float32 rounding cannot introduce distance ties.
            raw = frame_meta or meta
            frame_vectors = (
                raw["positions"][index[1]]
                + shifts.T.to(torch.float64) @ raw["box"]
                - raw["positions"][index[0]]
            )
            selection = (
                inputs.get(f"{neighborhood_support}_to_output")
                if neighborhood_support != "output"
                else None
            )
            if selection is not None and (selection < 0).any():
                raise ValueError(
                    "Bond frame neighborhoods must reference output support"
                )
            frames = bond_frames(index, frame_vectors, neighbor_selection=selection).to(
                dtype
            )
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
            from data.graph_features import evaluate_graph_geometry

            evaluate_graph_geometry(
                inputs,
                local_coordinates=True,
                num_l=num_l,
                neighborhood_support=neighborhood_support,
            )
    from data.graph_features import evaluate_graph_geometry

    evaluate_graph_geometry(inputs)
    inputs["_geometry_reference"] = {
        "positions": positions.detach().clone(),
        "box": box.detach().clone(),
    }
    if target_frame not in {"global", "bond"}:
        raise ValueError("target_frame must be global or bond")
    targets, full_blocks = {}, {}
    for name, native in matrices.items():
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
    sid = str(sample_id)
    return {
        "id": sid,
        "split": split,
        "inputs": inputs,
        "targets": targets,
        "source": sid,
        "keys": keys,
        "full_blocks": full_blocks,
        "species_shells": meta["species_shells"],
        "target_frame": target_frame,
    }


def statistics_records(
    samples, codec, *, target="hamiltonian", mapper=None, target_frame="global"
):
    for sample in samples:
        if sample["split"] != "train":
            raise ValueError("Statistics cannot access validation or test samples")
        record = sample["targets"][target]
        if not isinstance(record, dict):
            from data.block_matrix import IrrepsBlockData

            if isinstance(record, IrrepsBlockData):
                if mapper is None:
                    raise ValueError("Vector supervision requires its orbital mapper")
                record = record.to_blocks(mapper)
            x = sample["inputs"]
            edges = x.get("edge_keys")
            if edges is None:
                edges = torch.cat((x["edge_shift"], x["edge_index"]))
            values = IrrepsBlockData.pack_matrix(
                record,
                edges,
                codec,
                frames=x.get("bond_frames") if target_frame == "bond" else None,
                operator=target,
            )
            mask = torch.stack(
                [
                    codec.packed_mask(
                        (record.atoms[int(edge[3])], record.atoms[int(edge[4])]),
                        device=values.device,
                    )
                    for edge in edges.T
                ]
            ) & torch.isfinite(values)
            record = {"values": values, "mask": mask}
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
