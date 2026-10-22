"""Native HDF5, periodic support, shell masks and genuine disjoint batching."""

import importlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from data.sample import disjoint_union_samples
import pytest
import torch

from data.bond_geometry import bond_frames
from data.operator_features import (
    load_operator_sample,
    reconstruct_operator_blocks,
    statistics_records,
)
from core.operator_layout import ShellTensorCodec
from net.transforms.local_coordinates import (
    LocalCoordinateMessage,
    ScalarGatedNodeUpdate,
)


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def write_native(directory, *, spinful=False, missing_target=False):
    directory.mkdir()
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.2, 1.4, 0.5]])
    box = np.array([[5.0, 0.0, 0.0], [0.3, 6.0, 0.0], [0.4, 0.2, 7.0]])
    np.savetxt(directory / "element.dat", [1, 8, 1], fmt="%d")
    np.savetxt(directory / "site_positions.dat", positions.T)
    np.savetxt(directory / "lat.dat", box.T)
    (directory / "orbital_types.dat").write_text("0 1\n0 1 2\n0 1\n")
    (directory / "info.json").write_text(json.dumps({"isspinful": spinful}))
    keys = [(0, 0, 0, i + 1, j + 1) for i in range(3) for j in range(3)] + [
        (1, 0, 0, 1, 1),
        (-1, 0, 0, 1, 1),
    ]
    dimensions = [4, 9, 4]
    random = np.random.default_rng(977)
    with (
        h5py.File(directory / "hamiltonians.h5", "w") as h,
        h5py.File(directory / "overlaps.h5", "w") as s,
    ):
        for key in reversed(keys):
            factor = 2 if spinful else 1
            shape = dimensions[key[3] - 1] * factor, dimensions[key[4] - 1] * factor
            block = random.normal(size=shape)
            if spinful:
                block = block + 1j * random.normal(size=shape)
            name = str(list(key))
            s[name] = np.zeros(shape)
            h[name] = block
    with h5py.File(directory / "hamiltonians.h5") as h:
        # Match source first-atom grouping of lexical HDF5 iteration order.
        ordered = [
            tuple(json.loads(key))
            for atom in range(1, 4)
            for key in h
            if json.loads(key)[3] == atom
        ]
    edge_index = torch.tensor([[key[3] - 1, key[4] - 1] for key in ordered]).T
    vectors = (
        torch.tensor(positions)[edge_index[1]]
        + torch.tensor([key[:3] for key in ordered]).to(torch.float64)
        @ torch.tensor(box)
        - torch.tensor(positions)[edge_index[0]]
    )
    frames = bond_frames(edge_index, vectors)
    with h5py.File(directory / "rc.h5", "w") as h:
        for key, frame in zip(ordered, frames):
            h[str(list(key))] = frame.numpy()
    if missing_target:
        with (
            h5py.File(directory / "density.h5", "w") as d,
            h5py.File(directory / "hamiltonians.h5") as h,
        ):
            for index, (key, value) in enumerate(h.items()):
                if index != 1:
                    block = np.asarray(value).copy()
                    if index == 2:
                        block[0, 0] = np.nan
                    d[key] = block
    codec = ShellTensorCodec.from_species_shells(
        {"H": [0, 1], "O": [0, 1, 2]}, spinful=spinful, dtype=torch.float64
    )
    return codec, positions, box


@pytest.mark.parametrize("spinful", [False, True])
@pytest.mark.parametrize("frame", ["global", "bond"])
def test_native_periodic_blocks_masks_and_soc_export(tmp_path, spinful, frame):
    directory = tmp_path / "sample"
    codec, positions, box = write_native(directory, spinful=spinful)
    sample = load_operator_sample(
        directory,
        codec,
        species=["H", "O"],
        local_coordinates=True,
        frame_source="stored",
        target_frame=frame,
        dtype=torch.float64,
    )
    x, targets = sample["inputs"], sample["targets"]["hamiltonian"]
    torch.testing.assert_close(x["positions"], torch.tensor(positions))
    torch.testing.assert_close(x["box"], torch.tensor(box))
    assert sum(key[0] != 0 for key in sample["keys"]) == 2
    shifted_self = (x["edge_index"][0] == x["edge_index"][1]) & (
        x["edge_shift"] != 0
    ).any(0)
    assert shifted_self.sum() == 2
    assert not x["selfloop_edge"][shifted_self].any()
    assert torch.isfinite(targets["values"]).all()
    recovered, masks = reconstruct_operator_blocks(
        sample, codec, targets["values"], return_masks=True
    )
    for key, block in recovered.items():
        torch.testing.assert_close(
            block, sample["full_blocks"]["hamiltonian"][key], rtol=3e-12, atol=3e-12
        )
        assert masks[key].all()
        assert block.is_complex() == spinful
    # Both directions remain independent; no cleanup creates reverse blocks.
    forward = recovered[(1, 0, 0, 0, 0)]
    reverse = recovered[(-1, 0, 0, 0, 0)]
    assert not torch.allclose(forward, reverse.T.conj())


@pytest.mark.parametrize("spinful", [False, True])
def test_statistics_mask_missing_and_partial_labels_without_leakage(tmp_path, spinful):
    directory = tmp_path / "sample"
    codec, _, _ = write_native(directory, spinful=spinful, missing_target=True)
    sample = load_operator_sample(
        directory,
        codec,
        species=["H", "O"],
        matrix_files={"density": "density.h5"},
        dtype=torch.float64,
    )
    target = sample["targets"]["density"]
    assert torch.isfinite(target["values"]).all()
    assert (~target["mask"].flatten(1).any(1)).sum() == 1
    record = next(statistics_records([sample], codec, target="density"))
    assert torch.isfinite(record["coupled"]).all()
    torch.testing.assert_close(record["mask"], codec.encode_mask(target["mask"]))
    partial = target["mask"].flatten(1).sum(1)
    assert (record["mask"].sum(1) == 0).any()
    assert partial.max() > 0
    sample["split"] = "validation"
    with pytest.raises(ValueError, match="validation or test"):
        list(statistics_records([sample], codec, target="density"))


def test_disjoint_batch_local_relations_and_training_are_graph_independent(tmp_path):
    codec, _, _ = write_native(tmp_path / "a")
    write_native(tmp_path / "b")
    samples = [
        load_operator_sample(
            tmp_path / name,
            codec,
            species=["H", "O"],
            local_coordinates=True,
            frame_source="stored",
            dtype=torch.float64,
        )
        for name in ("a", "b")
    ]
    xs, ys = disjoint_union_samples(samples)
    first = samples[0]["inputs"]
    n, e, r = (
        len(first["node_type_idx"]),
        len(first["edge_lengths"]),
        len(first["sub_index"]),
    )
    torch.testing.assert_close(xs["edge_index"][:, e:], first["edge_index"] + n)
    torch.testing.assert_close(xs["sub_atom_idx"][r:], first["sub_atom_idx"] + n)
    torch.testing.assert_close(xs["sub_edge_idx"][r:], first["sub_edge_idx"] + e)
    torch.testing.assert_close(xs["sub_index"][r:], first["sub_index"] + 2 * e)
    assert torch.equal(
        xs["node_batch"][xs["edge_index"][0]], xs["node_batch"][xs["edge_index"][1]]
    )
    torch.testing.assert_close(
        ys["hamiltonian"]["values"],
        torch.cat([s["targets"]["hamiltonian"]["values"] for s in samples]),
    )
    node = torch.randn(2 * n, 4)
    edge = torch.randn(2 * e, 3)
    node_module = ScalarGatedNodeUpdate(4, 3)
    local_module = LocalCoordinateMessage(4, 3, 25, 2)

    def predict(x, nfeat, efeat):
        nodes = node_module(
            nfeat, x["edge_index"], efeat, x["node_batch"], x["edge_lengths"]
        )
        return local_module(
            nodes,
            efeat,
            *(
                x[name]
                for name in (
                    "sub_atom_idx",
                    "sub_edge_idx",
                    "sub_edge_ang",
                    "sub_index",
                )
            ),
            x["edge_lengths"],
        )

    whole = predict(xs, node, edge)
    separate = torch.cat(
        [
            predict(
                samples[i]["inputs"],
                node[i * n : (i + 1) * n],
                edge[i * e : (i + 1) * e],
            )
            for i in range(2)
        ]
    )
    torch.testing.assert_close(whole, separate, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("frame_source", ["stored", "derive"])
def test_original_preprocessing_graph_local_angles_and_rotated_targets_match(
    tmp_path, monkeypatch, frame_source
):
    reference_path = os.getenv("MANDALA_DEEPH_REFERENCE")
    if not reference_path or not (Path(reference_path) / "deeph/graph.py").exists():
        pytest.skip("Set MANDALA_DEEPH_REFERENCE for independent preprocessing oracle")
    monkeypatch.syspath_prepend(reference_path)
    geometry = importlib.import_module("deeph.preprocess.get_rc")
    rotation = importlib.import_module("deeph.rotate")
    graph_module = importlib.import_module("deeph.graph")
    codec, positions, box = write_native(tmp_path / "sample")
    directory = tmp_path / "sample"
    # The complete independent source preprocessing writes its own rc and rh.
    geometry.get_rc(str(directory), str(directory), radius=-1)
    rotation.get_rh(str(directory), str(directory))
    sample = load_operator_sample(
        directory,
        codec,
        species=["H", "O"],
        support_file="rc.h5",
        frame_source=frame_source,
        local_coordinates=True,
        target_frame="bond",
        dtype=torch.float64,
    )
    source_graph = graph_module.get_graph(
        torch.tensor(positions),
        torch.tensor(positions @ np.linalg.inv(box)),
        torch.tensor([1, 8, 1]),
        "test",
        -1,
        0,
        1e-8,
        torch.tensor(box),
        torch.float64,
        str(directory),
        "h5",
        5,
        True,
        True,
        False,
    )
    x = sample["inputs"]
    torch.testing.assert_close(x["edge_index"], source_graph.edge_index)
    torch.testing.assert_close(x["edge_lengths"], source_graph.edge_attr[:, 0])
    torch.testing.assert_close(
        x["edge_vectors"],
        source_graph.edge_attr[:, 4:7] - source_graph.edge_attr[:, 1:4],
    )
    for own_name, source_name in (
        ("sub_atom_idx", "subgraph_atom_idx"),
        ("sub_edge_idx", "subgraph_edge_idx"),
        ("sub_index", "subgraph_index"),
        ("sub_edge_ang", "subgraph_edge_ang"),
    ):
        torch.testing.assert_close(
            x[own_name], source_graph.subgraph_dict[source_name], rtol=2e-12, atol=2e-12
        )
    for edge, key in enumerate(sample["keys"]):
        pair = (x["atoms"][key[3]], x["atoms"][key[4]])
        dim_i = sum(2 * l + 1 for l in codec.species_shells[pair[0]])
        dim_j = sum(2 * l + 1 for l in codec.species_shells[pair[1]])
        source_block = source_graph.term_real[edge, :dim_i, :dim_j]
        expected = codec.pack_blocks(pair, source_block)
        torch.testing.assert_close(
            sample["targets"]["hamiltonian"]["values"][edge],
            expected,
            rtol=3e-12,
            atol=3e-12,
        )


@pytest.mark.parametrize("spinful", [False, True])
@pytest.mark.parametrize(
    "package,variable",
    [("deephe3", "MANDALA_DEEPHE3_REFERENCE"), ("maceh", "MANDALA_MACEH_REFERENCE")],
)
def test_native_support_and_labels_match_independent_equivariant_source(
    tmp_path, monkeypatch, package, variable, spinful
):
    path = os.getenv(variable) or (
        os.getenv("MANDALA_MACEH_REFERENCE_ROOT") if package == "maceh" else None
    )
    if not path or not (Path(path) / package / "graph.py").exists():
        pytest.skip(f"Set {variable} for independent native graph comparison")
    monkeypatch.syspath_prepend(path)
    source = importlib.import_module(f"{package}.graph")
    codec, positions, box = write_native(tmp_path / "sample", spinful=spinful)
    sample = load_operator_sample(
        tmp_path / "sample", codec, species=["H", "O"], dtype=torch.float64
    )
    graph = source.get_graph(
        torch.tensor(positions),
        torch.tensor(positions @ np.linalg.inv(box)),
        torch.tensor([1, 8, 1]),
        "test",
        -1,
        0,
        True,
        torch.tensor(box),
        torch.float64,
        str(tmp_path / "sample"),
        target_file_name="hamiltonians.h5",
    )
    inputs = sample["inputs"]
    torch.testing.assert_close(inputs["edge_index"], graph.edge_index)
    expected_keys = graph.edge_key.clone()
    expected_keys[:, 3:] -= 1
    torch.testing.assert_close(inputs["edge_keys"].T, expected_keys)
    torch.testing.assert_close(inputs["edge_lengths"], graph.edge_attr[:, 0])
    torch.testing.assert_close(inputs["edge_vectors"], graph.edge_attr[:, 1:4])
    reconstructed = reconstruct_operator_blocks(
        sample, codec, sample["targets"]["hamiltonian"]["values"]
    )
    for edge, key in enumerate(sample["keys"]):
        block = reconstructed[key]
        expected = graph.Aij[edge, : block.shape[0], : block.shape[1]]
        torch.testing.assert_close(block, expected, rtol=0, atol=0)


def test_factory_preserves_interleaved_shells_and_declared_filenames(tmp_path):
    from data.factory import DatasetFactory
    from net.common import Config

    path = tmp_path / "native"
    write_native(path)
    # All atoms of a given species share an explicit, nonsorted p,s order.
    (path / "orbital_types.dat").write_text("1 0\n1 0\n1 0\n")
    # write_native has H and O atoms; each now has four orbitals.
    import h5py

    for filename in ("hamiltonians.h5", "overlaps.h5"):
        with h5py.File(path / filename, "w") as handle:
            for i in range(3):
                for j in range(3):
                    handle[str([0, 0, 0, i + 1, j + 1])] = torch.eye(4).numpy()
    (path / "hamiltonians.h5").rename(path / "custom.h5")
    cfg = Config(
        verbosity=0,
        apply_cutoff_to_targets=False,
        graph_source="target_edges",
        matrix_targets=["hamiltonian"],
        data={"matrix_files": {"hamiltonian": "custom.h5"}},
    )
    factory = DatasetFactory(cfg, convention="openmx")
    factory.add_snapshot(path / "custom.h5", path / "info.json")
    dataset, _, mapper = factory.create()
    shells = {
        atom: [ir.l for mul, ir in irreps for _ in range(mul)]
        for atom, irreps in mapper.orbital_cfg.element_to_irreps.items()
    }
    assert all(value == [1, 0] for value in shells.values())
    assert all(
        block.shape[-2:] == (4, 4)
        for block in dataset[0][1]["hamiltonian"].pair_blocks.values()
    )


def test_differentiable_local_harmonics_match_values_and_remain_finite_at_poles():
    from data.bond_geometry import tesseral_harmonics, cartesian_tesseral_harmonics

    vectors = torch.randn(19, 3, dtype=torch.float64)
    torch.testing.assert_close(
        cartesian_tesseral_harmonics(vectors, 6),
        tesseral_harmonics(vectors, 6),
        rtol=1e-11,
        atol=1e-12,
    )
    poles = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], requires_grad=True)
    output = tesseral_harmonics(poles, 6)
    (gradient,) = torch.autograd.grad(output.sum(), poles, create_graph=True)
    (curvature,) = torch.autograd.grad(gradient.square().sum(), poles)
    assert (
        torch.isfinite(output).all()
        and torch.isfinite(gradient).all()
        and torch.isfinite(curvature).all()
    )
