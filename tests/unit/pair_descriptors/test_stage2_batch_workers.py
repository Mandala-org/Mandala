import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

from pair_descriptors import DeterministicCovariantLiftDescriptor


ROOT = Path(__file__).resolve().parents[3]
TARGET_IRREPS = ((0, 1), (1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1), (4, 1))


def _load(name):
    path = ROOT / "scripts" / "pair_descriptors" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    seed = {
        "key": "tiny_d4",
        "d1_key": "tiny_d1",
        "name": "tiny",
        "species": (8, 14),
        "radial_basis": "spherical_bessel",
        "n_radial": 1,
        "l_max": 1,
        "cutoff_angstrom": 5.0,
        "body_degree": 2,
        "lift_max_input_l": 1,
        "lift_max_radial_index": 0,
        "lift_radial_degree_budget": 0,
        "lift_max_intermediate_l": 2,
        "max_paths_per_degree_irrep": 2,
    }
    descriptor = DeterministicCovariantLiftDescriptor(
        (8, 14),
        radial_basis="spherical_bessel",
        n_radial=1,
        l_max=1,
        cutoff=5.0,
        body_degree=2,
        lift_max_input_l=1,
        lift_max_radial_index=0,
        lift_radial_degree_budget=0,
        lift_max_intermediate_l=2,
        target_irreps=TARGET_IRREPS,
        max_paths_per_degree_irrep=2,
    )
    return {**seed, **descriptor.metadata}


@pytest.mark.unit
def test_d4_worker_and_certification_smoke(tmp_path):
    precompute = _load("precompute_d4_sio2")
    certify = _load("certify_stage2_sio2")
    config = _config()
    descriptor = precompute.make_descriptor(config).to(dtype=torch.float64)
    vectors = torch.tensor([[0.8, 0.1, -0.2], [-0.3, 1.1, 0.4]], dtype=torch.float64)
    species = torch.tensor([8, 14])
    d1 = descriptor.base(vectors, species).float().numpy()
    source = tmp_path / "source.h5"
    destination = tmp_path / "destination.h5"
    with h5py.File(source, "w") as handle:
        handle.attrs["source_key"] = 17
        handle.create_dataset("atomic_numbers", data=np.array([8, 14]))
        handle.create_dataset("neighbor_center", data=np.array([0, 0]))
        handle.create_group("descriptors").create_dataset("tiny_d1", data=d1)
    precompute._initialize_worker([config])
    result = precompute._write_one((3, source, destination, "train", "manifest"))
    assert result["source_key"] == 17
    with h5py.File(destination, "r") as handle:
        values = handle["descriptors"]["tiny_d4"][:]
        assert bool(handle.attrs["complete"])
        assert values.shape == (1, descriptor.irreps_out.dim)
        assert np.isfinite(values).all()
    metrics = certify.certify_descriptor("d4", config, 123)
    assert metrics["proper_o3_relative_error"] < 1e-9
    assert metrics["improper_o3_relative_error"] < 1e-9
    assert metrics["puncture_relative_error"] < 1e-10
    assert metrics["metadata_hash_matches"]


@pytest.mark.unit
def test_normalization_worker_accumulates_component_squares(tmp_path):
    normalization = _load("fit_descriptor_normalization_sio2")
    cache = tmp_path / "cache" / "shards"
    cache.mkdir(parents=True)
    with h5py.File(cache / "structure_0000.h5", "w") as handle:
        group = handle.create_group("descriptors")
        group.create_dataset("x", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
    name, atoms, sums = normalization._sum_chunk(
        ("test", str(tmp_path / "cache"), [{"structure_index": "0"}], {"x": 2})
    )
    assert name == "test"
    assert atoms == 2
    assert np.array_equal(sums["x"], np.array([10.0, 20.0]))
