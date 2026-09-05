import csv
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch

from pair_descriptors import RawNeighborDensityDescriptor
from pair_descriptors.information_certification import (
    adversarial_collision_search,
    constructive_l0_l1_initializer,
    descriptor_jacobian,
    expanded_irrep_scales,
    geometry_signature,
    null_direction_continuation,
    pair_distance_rmsd,
    reconstruct_multistart,
    species_assigned_rmsd,
)


def _fixture():
    descriptor = RawNeighborDensityDescriptor(
        (8, 14),
        radial_basis="zernike",
        n_radial=2,
        l_max=2,
        cutoff=5.0,
    ).double()
    coordinates = torch.tensor(
        [[0.8, 0.1, 0.2], [-0.2, 1.1, 0.3], [0.4, -0.8, 1.0]],
        dtype=torch.float64,
    )
    species = torch.tensor([8, 14, 8])
    scales = expanded_irrep_scales(
        descriptor.irreps_out,
        [1.0] * sum(multiplicity for multiplicity, _ in descriptor.irreps_out),
    )
    return descriptor, coordinates, species, scales


def test_geometry_metrics_are_species_permutation_invariant():
    _descriptor, coordinates, species, _scales = _fixture()
    permutation = torch.tensor([2, 1, 0])
    assert (
        species_assigned_rmsd(
            coordinates[permutation], coordinates, species[permutation]
        )
        < 1e-14
    )
    assert pair_distance_rmsd(coordinates[permutation], coordinates) < 1e-14
    assert torch.allclose(
        geometry_signature(coordinates[permutation], species[permutation]),
        geometry_signature(coordinates, species),
    )


def test_jacobian_inverse_collision_and_continuation_smoke():
    descriptor, coordinates, species, scales = _fixture()
    jacobian = descriptor_jacobian(descriptor, coordinates, species, scales)
    assert jacobian.shape == (descriptor.irreps_out.dim, coordinates.numel())
    assert torch.isfinite(jacobian).all()
    inverse = reconstruct_multistart(
        descriptor,
        coordinates,
        species,
        scales,
        starts=2,
        steps=2,
        polish_steps=1,
        learning_rate=0.02,
        seed=4,
    )
    assert torch.isfinite(torch.tensor(inverse.normalized_descriptor_rms))
    collision = adversarial_collision_search(
        descriptor,
        coordinates,
        species,
        scales,
        starts=2,
        steps=2,
        learning_rate=0.02,
        minimum_geometry_rms=0.05,
        seed=5,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in collision.values())
    continuation = null_direction_continuation(
        descriptor,
        coordinates,
        species,
        scales,
        jacobian,
        steps=2,
        learning_rate=0.01,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in continuation.values())


def test_constructive_l0_l1_initializer_recovers_small_environment():
    descriptor, coordinates, species, scales = _fixture()
    target = descriptor(coordinates, species)[0]
    channels = [
        {
            "species": channel.species,
            "radial_index": channel.radial_index,
            "l": channel.l,
            "start": channel.start,
            "stop": channel.stop,
        }
        for channel in descriptor.channels
    ]
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        recovered, radial_residual = constructive_l0_l1_initializer(
            descriptor,
            target,
            species,
            scales,
            channels,
            starts=4,
            seed=22,
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert radial_residual < 1e-10
    assert species_assigned_rmsd(recovered, coordinates, species) < 1e-8
    assert pair_distance_rmsd(recovered, coordinates) < 1e-8


def test_stage3_cli_tiny_end_to_end(tmp_path):
    descriptor = RawNeighborDensityDescriptor(
        (8, 14), radial_basis="zernike", n_radial=1, l_max=1, cutoff=5.0
    )
    schema = {"key": "tiny", **descriptor.metadata}
    schema_path = tmp_path / "schemas.json"
    schema_path.write_text(json.dumps([schema]))
    channels = []
    offset = 0
    for multiplicity, irrep in descriptor.irreps_out:
        for _copy in range(multiplicity):
            channels.append({"start": offset, "stop": offset + irrep.dim, "rms": 1.0})
            offset += irrep.dim
    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text(
        json.dumps(
            {
                "content_hash": "tiny-normalization",
                "families": {
                    "d1": {"descriptors": [{"key": "tiny", "channels": channels}]}
                },
            }
        )
    )
    cache = tmp_path / "cache" / "shards"
    cache.mkdir(parents=True)
    with h5py.File(cache / "structure_0000.h5", "w") as handle:
        handle.create_dataset("atomic_numbers", data=np.array([8, 14]))
        handle.create_dataset("neighbor_center", data=np.array([0, 0]))
        handle.create_dataset("neighbor_atom", data=np.array([1, 0]))
        handle.create_dataset(
            "neighbor_displacement_angstrom",
            data=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        )
    registry = tmp_path / "shards.csv"
    with registry.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["structure_index", "split"])
        writer.writeheader()
        writer.writerow({"structure_index": 0, "split": "train"})
    output = tmp_path / "output"
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "pair_descriptors"
        / "run_stage3_information_suite.py"
    )
    command = [
        sys.executable,
        str(script),
        "--family",
        "d1",
        str(schema_path),
        "--normalization-json",
        str(normalization_path),
        "--d1-cache-dir",
        str(tmp_path / "cache"),
        "--d1-registry",
        str(registry),
        "--output-dir",
        str(output),
        "--num-workers",
        "1",
        "--seed",
        "9",
        "--neighbor-counts",
        "1",
        "--synthetic-repeats",
        "1",
        "--real-case-count",
        "1",
        "--real-neighbor-count",
        "1",
        "--inverse-starts",
        "1",
        "--inverse-steps",
        "1",
        "--inverse-polish-steps",
        "0",
        "--inverse-learning-rate",
        "0.01",
        "--collision-starts",
        "1",
        "--collision-steps",
        "1",
        "--collision-learning-rate",
        "0.01",
        "--collision-minimum-geometry-rms-angstrom",
        "0.01",
        "--continuation-steps",
        "1",
        "--continuation-learning-rate",
        "0.01",
        "--descriptor-match-tolerance",
        "1e-7",
        "--reconstruction-rmsd-tolerance-angstrom",
        "1e-4",
        "--rank-relative-tolerance",
        "1e-8",
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[3])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"]
    assert summary["task_count"] == 2
    assert (output / "basis_convergence.png").is_file()
