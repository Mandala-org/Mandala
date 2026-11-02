"""Scientific failure probes through raw records, real models, and optimization.

Small CPU examples retain each reference model's complete default stage graph.
Assertions use independent samples/analytic labels, never a second batched path.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.operator_features import load_operator_sample
from data.graph_features import complete_graph_metadata
from data.sample import GraphBatch, union_graph_inputs
from net.common import Config
from net.configuration import resolve_configuration
from net.e3gnn import E3GNN

ROOT = Path(__file__).resolve().parents[2]
KINDS = ["deeph", "deeph_e3", "mace_h", "tensor_messages_projected_attention"]


@pytest.fixture(autouse=True)
def _double_cpu():
    previous, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous)
    torch.set_num_threads(threads)


def _recipe(kind):
    group = "reference" if kind in KINDS[:3] else "hybrids"
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / f"configs/{group}/{kind}.yaml"), resolve=False
    )
    if kind == "deeph":
        config["parameters"].update(
            node_width=8, edge_width=12, post_edge_width=8, num_l=3, angular_dim=9
        )
    elif kind in KINDS[:3]:
        config["parameters"].update(
            embedding_width=4,
            radial_width=5,
            node_irreps="4x0e+3x1o+2x2e",
            edge_irreps="4x0e+3x1o+2x2e",
            angular_irreps="1x0e+1x1o+1x2e",
        )
        if kind == "mace_h":
            config["parameters"].update(
                hidden_irreps="3x0e+3x1o+3x2e",
                density_lmax=2,
                density_radial_width=5,
                correlation=2,
                radial_layers=[5],
                avg_num_neighbors=2.0,
            )
    if kind == KINDS[-1]:
        for stage in config["pipeline"]["stages"]:
            if stage["id"] == "output_angular":
                stage["options"]["irreps"] = "0e+1o+2e+3o+4e"
        stages = config["pipeline"]["stages"]
        decode = next(s for s in stages if s["id"] == "decode")
        decode["inputs"] = ["coupled_complete"]
        stages.insert(
            stages.index(decode),
            {
                "id": "readout_completion",
                "role": "readout",
                "transform": "irrep_linear",
                "inputs": ["coupled"],
                "outputs": ["coupled_complete"],
                "options": {"irreps_out": "${layout.coupled_irreps}"},
            },
        )
    config.update(verbosity=0, loss_weighting_mode="off")
    return config


def _model(kind, shells, *, spinful=False, operators=("hamiltonian",), learning=False):
    config = _recipe(kind)
    if learning and kind in ("deeph_e3", "mace_h"):
        config["parameters"].update(
            embedding_width=8,
            radial_width=8,
            node_irreps="8x0e+6x1o+4x2e",
            edge_irreps="8x0e+6x1o+4x2e",
        )
    if learning and kind == "mace_h":
        config["parameters"].update(
            hidden_irreps="5x0e+5x1o+5x2e", density_radial_width=8, radial_layers=[8]
        )
    if len(operators) > 1:
        original = config["pipeline"]
        common = [s for s in original["stages"] if s.get("role") != "readout"]
        heads = [s for s in original["stages"] if s.get("role") == "readout"]
        outputs = {}
        for operator in operators:
            rename = {
                name: f"{operator}_{name}" for s in heads for name in s["outputs"]
            }
            for stage in deepcopy(heads):
                stage["id"] = f"{operator}_{stage['id']}"
                stage["inputs"] = [rename.get(x, x) for x in stage["inputs"]]
                stage["outputs"] = [rename[x] for x in stage["outputs"]]
                if stage["transform"] == "species_irrep_affine":
                    stage.setdefault("options", {})["target"] = operator
                common.append(stage)
            outputs[operator] = rename[original["outputs"]["hamiltonian"]]
        original.update(stages=common, outputs=outputs)
    config["matrix_targets"] = list(operators)
    config, codec = resolve_configuration(
        config, species_shells=shells, spinful=spinful
    )
    mapper = BlockIrrepMapper(
        OrbitalIrrepConfig.from_dict(
            {
                s: "+".join(f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in ls)
                for s, ls in shells.items()
            }
        ),
        dtype=torch.float64,
    )
    torch.manual_seed(348)
    return E3GNN(mapper, Config.from_mapping(config)).double(), codec, config


def _write(
    directory,
    atoms,
    positions,
    shells,
    *,
    spinful=False,
    isolated=False,
    sample=0,
    operators=("hamiltonian",),
):
    directory.mkdir(parents=True)
    cell = np.array(
        [[7 + sample, 0.2, 0], [0.3, 8 + sample, 0.1], [0, 0.4, 9 + sample]]
    )
    np.savetxt(directory / "lat.dat", cell.T)
    np.savetxt(directory / "site_positions.dat", np.asarray(positions).T)
    np.savetxt(
        directory / "element.dat", [1 if a == "H" else 8 for a in atoms], fmt="%d"
    )
    (directory / "orbital_types.dat").write_text(
        "".join(" ".join(map(str, shells[a])) + "\n" for a in atoms)
    )
    (directory / "info.json").write_text(json.dumps({"isspinful": spinful}))
    names = {
        "hamiltonian": "hamiltonians.h5",
        "overlap": "overlaps.h5",
        "density": "density_matrixs.h5",
    }
    for op_idx, op in enumerate(operators):
        with h5py.File(directory / names[op], "w") as handle:
            for i, a in enumerate(atoms):
                for j, b in enumerate(atoms):
                    if isolated and i != j:
                        continue
                    ni, nj = [sum(2 * l + 1 for l in shells[t]) for t in (a, b)]
                    v = np.asarray(positions[j]) - np.asarray(positions[i])
                    r = np.linalg.norm(v)
                    u = v / r if r > 1e-12 else np.zeros(3)
                    block = np.zeros((ni, nj))
                    base = (
                        0.12 + 0.03 * (a == "O") + 0.05 * (b == "O") + 0.5 * np.exp(-r)
                    )
                    block[0, 0] = base
                    if ni >= 4 and nj >= 4:
                        # OpenMX p orbital order is (px, py, pz). All three
                        # independent directions participate in the objective.
                        block[0, 1:4] = 0.4 * np.exp(-r) * u
                        block[1:4, 0] = -0.3 * np.exp(-r) * u
                        block[1:4, 1:4] = 0.06 * np.eye(3) + 0.3 * np.exp(
                            -r
                        ) * np.outer(u, u)
                    block = (1 + 0.3 * op_idx) * block + (0.015 * op_idx) * np.eye(
                        ni, nj
                    )
                    if spinful:
                        block = np.kron(np.eye(2), block).astype(np.complex128)
                        block[:ni, nj:] += 1j * 0.013
                        block[ni:, :nj] -= 1j * 0.017
                    if sample == 1 and i == 0 and j == 0:
                        block[0, -1] = np.nan
                    handle[str([0, 0, 0, i + 1, j + 1])] = block
        if not isolated:
            with h5py.File(directory / names[op], "a") as handle:
                for sign in (-1, 1):
                    ni = sum(2 * l + 1 for l in shells[atoms[0]]) * (
                        2 if spinful else 1
                    )
                    handle[str([sign, 0, 0, 1, 1])] = np.eye(ni) * (0.03 + 0.01 * sign)
    return directory


def _load(path, codec, config, operators=("hamiltonian",)):
    names = {
        "hamiltonian": "hamiltonians.h5",
        "overlap": "overlaps.h5",
        "density": "density_matrixs.h5",
    }
    return load_operator_sample(
        path,
        codec,
        species=list(codec.species_shells),
        dtype=torch.float64,
        local_coordinates=config["data"].get("local_coordinates", False),
        num_l=config.get("parameters", {}).get("num_l", 3),
        target_frame=config["data"].get("target_frame", "global"),
        matrix_files={o: names[o] for o in operators},
        supports=config["data"].get("supports"),
        frame_source=config["data"].get("frame_source", "derive"),
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("spinful", [False, True])
def test_unequal_periodic_species_shells_batch_matches_individual_outputs_losses_and_gradients(
    tmp_path, kind, spinful
):
    shells = {"H": [0, 1], "O": [0, 1, 2]}
    model, codec, config = _model(kind, shells, spinful=spinful)
    paths = [
        _write(
            tmp_path / "a",
            ["H", "O", "H"],
            [[0, 0, 0], [1.1, 0.3, 0.2], [-0.2, 1.4, 0.5]],
            shells,
            spinful=spinful,
        ),
        _write(
            tmp_path / "b",
            ["O", "H", "O", "H"],
            [[0.4, 0.3, 0.1], [1.8, 0.1, 0.6], [0.1, 1.9, 0.7], [0.7, 0.4, 2.1]],
            shells,
            spinful=spinful,
            sample=1,
        ),
        _write(
            tmp_path / "c",
            ["H", "H", "H"] if kind == "deeph" else ["O"],
            (
                [[0, 0, 0], [1.2, 0.1, 0.3], [0.3, 1.4, 0.2]]
                if kind == "deeph"
                else [[0.7, 0.2, 0.5]]
            ),
            shells,
            spinful=spinful,
            isolated=kind != "deeph",
            sample=2,
        ),
    ]
    samples = [_load(p, codec, config) for p in paths]
    model.fit_training_data(SimpleNamespace(samples=samples))
    graphs = [
        (
            complete_graph_metadata(
                s["inputs"], model.mapper, species_order=list(codec.species_shells)
            ),
            s["targets"],
        )
        for s in samples
    ]
    model.eval()
    for graph in graphs:
        graph[0]["positions"] = graph[0]["positions"].detach().requires_grad_(True)
    independent = [model(g[0])["hamiltonian"].packed_values for g in graphs]
    union = union_graph_inputs(graphs, layouts=model.pipeline.input_layouts)
    together = model(union)["hamiltonian"].packed_values
    torch.testing.assert_close(together, torch.cat(independent), rtol=2e-9, atol=2e-10)
    assert [g[0]["edge_index"].shape[1] for g in graphs] == (
        [11, 18, 11] if kind == "deeph" else [11, 18, 1]
    )
    assert [g[0]["message_edge_index"].shape[1] for g in graphs] == (
        [8, 14, 8]
        if kind == "deeph"
        else [8, 12, 0] if kind == KINDS[-1] else [8, 14, 0]
    )
    assert (~graphs[1][1]["hamiltonian"]["mask"]).any()
    counts = [int(g[1]["hamiltonian"]["mask"].sum()) for g in graphs]
    losses = [model._graph_step(g, 0, "train")[0] for g in graphs]
    expected = sum(l * n for l, n in zip(losses, counts)) / sum(counts)
    reference_grads = torch.autograd.grad(
        expected,
        tuple(model.parameters()) + tuple(g[0]["positions"] for g in graphs),
        allow_unused=True,
    )
    actual = model._shared_step(GraphBatch(graphs), 0, "train")
    torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)
    actual_grads = torch.autograd.grad(
        actual,
        tuple(model.parameters()) + tuple(g[0]["positions"] for g in graphs),
        allow_unused=True,
    )
    for got, wanted in zip(actual_grads, reference_grads):
        if wanted is None:
            assert got is None
        else:
            torch.testing.assert_close(got, wanted, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.slow(
    reason="500 Adam steps through three raw geometries and three independently supervised operator heads"
)
def test_short_fit_learns_distance_species_and_angular_operator_targets(tmp_path, kind):
    """An untrained or geometry-independent predictor cannot meet these bounds."""
    shells = {"H": [0, 1], "O": [0, 1]}
    operators = ("hamiltonian", "density", "overlap")
    model, codec, config = _model(kind, shells, operators=operators, learning=True)
    geometries = [
        [[0, 0, 0], [1.0, 0.2, 0.1], [0.1, 1.0, 0.3]],
        [[0, 0, 0], [1.7, -0.2, 0.3], [-0.3, 1.5, 0.8]],
        [[0, 0, 0], [0.8, 0.7, -0.1], [0.6, -0.2, 1.4]],
    ]
    samples = [
        _load(
            _write(
                tmp_path / str(i),
                ["H", "O", "H"],
                pos,
                shells,
                sample=3 + i,
                operators=operators,
            ),
            codec,
            config,
            operators,
        )
        for i, pos in enumerate(geometries)
    ]
    model.fit_training_data(SimpleNamespace(samples=samples))
    graphs = [
        (
            complete_graph_metadata(
                s["inputs"], model.mapper, species_order=list(shells)
            ),
            s["targets"],
        )
        for s in samples
    ]
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)

    def errors():
        predictions = [model(x) for x, _ in graphs]
        return {
            op: torch.stack(
                [
                    (p[op].packed_values - y[op]["values"])[y[op]["mask"]]
                    .abs()
                    .square()
                    .mean()
                    for p, (_, y) in zip(predictions, graphs)
                ]
            ).mean()
            for op in operators
        }

    with torch.no_grad():
        initial = {k: float(v) for k, v in errors().items()}
    # The baseline has an independent constant for every matrix entry and edge
    # index: beating it requires learning variation between actual geometries.
    baseline = {}
    for op in operators:
        targets = torch.stack([y[op]["values"] for _, y in graphs])
        baseline[op] = float((targets - targets.mean(0)).abs().square().mean())
        assert baseline[op] > 1e-7
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=500, eta_min=0.0001
    )
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = model._shared_step(GraphBatch(graphs), 0, "train")
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        final = {k: float(v) for k, v in errors().items()}
    for op in operators:
        assert final[op] < 0.2 * initial[op], (kind, op, initial, final, baseline)
        assert final[op] < 0.6 * baseline[op], (kind, op, initial, final, baseline)


def _unpack(record):
    return torch.tensor(
        record["values"], dtype=getattr(torch, record["dtype"])
    ).reshape(record["shape"])


def _source_parameter_name(kind, name):
    _, stage, tail = name.split(".", 2)
    if kind == "deeph_e3":
        prefix = {
            "embedding": "embedding",
            "node_readout": "lin_node",
            "edge_readout": "lin_edge",
        }.get(stage)
        if stage.startswith("node_update_"):
            prefix = f"node_update_blocks.{stage.rsplit('_', 1)[-1]}"
        if stage.startswith("edge_update_"):
            prefix = f"edge_update_blocks.{stage.rsplit('_', 1)[-1]}"
    else:
        prefix = {
            "node_embedding": "mace.node_embedding.linear",
            "density_radial_basis": "mace.radial_embedding",
            "edge_readout": "edge_update.lin_edge",
        }.get(stage)
        for start, source in [
            ("density_", "mace.interactions"),
            ("product_", "mace.products"),
            ("angular_expansion_", "expansion.node_order_expansion_blocks"),
            ("edge_update_", "edge_update.edge_update_blocks"),
        ]:
            if stage.startswith(start) and stage.rsplit("_", 1)[-1].isdigit():
                prefix = f"{source}.{stage.rsplit('_', 1)[-1]}"
    assert prefix is not None, name
    return f"{prefix}.{tail}"


@pytest.mark.parametrize(
    "kind,frame_source",
    [
        ("deeph", "derive"),
        ("deeph", "stored"),
        ("deeph_e3", "derive"),
        ("mace_h", "derive"),
    ],
)
def test_raw_native_records_reproduce_pinned_source_preprocessing_and_three_adam_updates(
    tmp_path, kind, frame_source
):
    """Always runs: upstream source execution is frozen, never skipped/mocked."""
    import gzip

    with gzip.open(
        ROOT / f"tests/fixtures/raw_training/{kind}.json.gz", "rt"
    ) as handle:
        oracle = json.load(handle)
    from scripts.generate_reference_fixtures import PINS

    assert oracle["source"]["commit"] == PINS[kind]
    assert oracle["preprocessing_dtype"] == oracle["evaluation_dtype"] == "float64"
    assert len(oracle["steps"]) == 3
    assert oracle["source"]["sha256"]
    assert all(len(digest) == 64 for digest in oracle["source"]["sha256"].values())
    raw = oracle["raw"]
    assert raw["shells"] == [0, 1]
    assert any(json.loads(key)[:3] != [0, 0, 0] for key in raw["hamiltonians"])
    np.savetxt(tmp_path / "site_positions.dat", np.asarray(raw["positions"]).T)
    np.savetxt(tmp_path / "lat.dat", np.asarray(raw["box"]).T)
    np.savetxt(tmp_path / "element.dat", raw["atomic_numbers"], fmt="%d")
    (tmp_path / "orbital_types.dat").write_text(
        (" ".join(map(str, raw["shells"])) + "\n") * len(raw["positions"])
    )
    (tmp_path / "info.json").write_text('{"isspinful":false}')
    with h5py.File(tmp_path / "hamiltonians.h5", "w") as h:
        for key, block in raw["hamiltonians"].items():
            h[key] = block
    config = oracle["configuration"]
    if kind == "deeph":
        with h5py.File(tmp_path / "rc.h5", "w") as h:
            for key, frame in raw["frames"].items():
                h[key] = frame
        config["data"]["frame_source"] = frame_source
    species = "H" if kind == "deeph" else "Si"
    _, codec = resolve_configuration(config, species_shells={species: raw["shells"]})
    mapper = BlockIrrepMapper(
        OrbitalIrrepConfig.from_dict(
            {
                species: "+".join(
                    f"1x{l}{'e' if l%2 == 0 else 'o'}" for l in raw["shells"]
                )
            }
        ),
        dtype=torch.float64,
    )
    torch.set_default_dtype(getattr(torch, oracle["initialization_dtype"]))
    torch.manual_seed(oracle["seed"])
    model = E3GNN(mapper, Config.from_mapping(config)).double()
    torch.set_default_dtype(torch.float64)
    if kind == "mace_h":
        # Accepted training policy excludes upstream full-population leakage.
        # A declared identity calibration isolates source forward/backward
        # maths; real training-only fitted statistics are checked separately.
        affine = model.pipeline.transforms["output_affine"]
        affine.fit(
            [
                {
                    "id": "source-identity-calibration",
                    "split": "train",
                    "pair_type": torch.zeros(1, dtype=torch.long),
                    "coupled": torch.zeros(1, affine.irreps.dim),
                }
            ],
            training_ids=["source-identity-calibration"],
        )
    sample = _load(tmp_path, codec, config)
    x, y = sample["inputs"], sample["targets"]["hamiltonian"]
    torch.testing.assert_close(
        x["edge_index"], _unpack(oracle["edge_index"]), rtol=0, atol=0
    )
    source_attrs = _unpack(oracle["edge_attr"])
    source_vectors = (
        source_attrs[:, 4:7] - source_attrs[:, 1:4]
        if kind == "deeph"
        else source_attrs[:, 1:4]
    )
    torch.testing.assert_close(
        x["edge_vectors"], source_vectors, rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(
        x["edge_lengths"], source_attrs[:, 0], rtol=2e-12, atol=2e-12
    )
    if kind == "deeph":
        for name, source in [
            ("sub_atom_idx", "subgraph_atom_idx"),
            ("sub_edge_idx", "subgraph_edge_idx"),
            ("sub_index", "subgraph_index"),
            ("sub_edge_ang", "subgraph_edge_ang"),
        ]:
            torch.testing.assert_close(
                x[name],
                _unpack(oracle["source_subgraph"][source]),
                rtol=2e-12,
                atol=2e-12,
            )
    torch.testing.assert_close(
        y["values"], _unpack(oracle["target"]), rtol=2e-12, atol=2e-12
    )
    assert y["mask"].all()
    parameters = list(model.pipeline.named_parameters())
    coordinates = (
        dict(zip(oracle["initial_parameters"], (p for _, p in parameters)))
        if kind == "deeph"
        else {_source_parameter_name(kind, n): p for n, p in parameters}
    )
    assert coordinates.keys() == oracle["initial_parameters"].keys()
    for name, p in coordinates.items():
        torch.testing.assert_close(
            p, _unpack(oracle["initial_parameters"][name]), rtol=0, atol=0
        )
    optimizer = torch.optim.Adam(model.parameters(), **oracle["optimizer"])
    for step in oracle["steps"]:
        optimizer.zero_grad()
        output = model(x)["hamiltonian"].packed_values
        torch.testing.assert_close(
            output, _unpack(step["output"]), rtol=2e-8, atol=2e-10
        )
        loss = (output - y["values"]).abs().square().mean()
        torch.testing.assert_close(
            loss, torch.tensor(step["loss"]), rtol=2e-9, atol=2e-10
        )
        loss.backward()
        for name, p in coordinates.items():
            expected = step["gradients"][name]
            if expected is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(
                    p.grad, _unpack(expected), rtol=2e-8, atol=2e-10
                )
        if kind == "deeph":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.2)
        optimizer.step()
        for name, p in coordinates.items():
            torch.testing.assert_close(
                p, _unpack(step["updated_parameters"][name]), rtol=2e-8, atol=2e-10
            )


@pytest.mark.parametrize("kind", KINDS[:3])
def test_masked_train_and_held_out_labels_cannot_affect_statistics_or_first_update(
    tmp_path, kind
):
    from scripts.train import build_components

    shells = {"H": [0, 1], "O": [0, 1]}
    pos = [[0, 0, 0], [1.1, 0.3, 0.2], [-0.2, 1.4, 0.5]]
    paths = [
        _write(tmp_path / str(i), ["H", "O", "H"], pos, shells, sample=i)
        for i in range(4)
    ]
    document = _recipe(kind)
    document["data"]["support_operator"] = "hamiltonian"
    document["training"].update(dtype="float64", batch_size=2)
    document["data"].update(
        splits={
            "train": [str(p) for p in paths[:2]],
            "val": [str(paths[2])],
            "test": [str(paths[3])],
        }
    )
    _, _, baseline, base_datasets, _ = build_components(document, resume=True)
    # Mutate the actual held-out raw files before constructing the second run.
    # Atomic species and orbital metadata remain byte-identical.
    for p in paths[2:]:
        with h5py.File(p / "hamiltonians.h5", "r+") as h:
            for key in h:
                h[key][...] = np.asarray(h[key]) * 1e8 + 3e9
    _, _, changed, changed_datasets, _ = build_components(document, resume=True)
    count = 0
    for sample in changed_datasets["train"].samples:
        record = sample["targets"]["hamiltonian"]
        count += int((~record["mask"]).sum())
        record["values"][~record["mask"]] = 1e12
    assert count > 0
    baseline.fit_training_data(base_datasets["train"])
    changed.fit_training_data(changed_datasets["train"])
    for name, buffer in baseline.named_buffers():
        other = dict(changed.named_buffers())[name]
        torch.testing.assert_close(buffer, other, rtol=0, atol=0)
    gradients = []
    updates = []
    losses = []
    for model, datasets in [(baseline, base_datasets), (changed, changed_datasets)]:
        model.train()
        graphs = [datasets["train"][i] for i in range(len(datasets["train"]))]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        optimizer.zero_grad()
        loss = model._shared_step(GraphBatch(graphs), 0, "train")
        loss.backward()
        losses.append(loss.detach())
        gradients.append(
            {
                name: None if p.grad is None else p.grad.clone()
                for name, p in model.named_parameters()
            }
        )
        optimizer.step()
        updates.append(
            {name: p.detach().clone() for name, p in model.named_parameters()}
        )
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for name in updates[0]:
        if gradients[0][name] is None:
            assert gradients[1][name] is None
        else:
            torch.testing.assert_close(
                gradients[0][name], gradients[1][name], rtol=0, atol=0
            )
        torch.testing.assert_close(updates[0][name], updates[1][name], rtol=0, atol=0)
    # Positive control: unmasked training supervision really reaches the loss.
    graph = changed_datasets["train"][0]
    altered = deepcopy(graph[1])
    mask = altered["hamiltonian"]["mask"]
    altered["hamiltonian"]["values"][mask] += 10.0
    assert not torch.isclose(
        changed._graph_step(graph, 0, "train")[0],
        changed._graph_step((graph[0], altered), 0, "train")[0],
    )


def test_local_frame_models_reject_truly_isolated_support_instead_of_forging_gauge(
    tmp_path,
):
    shells = {"H": [0]}
    _, codec, config = _model("deeph", shells)
    path = _write(
        tmp_path / "isolated", ["H"], [[0.0, 0.0, 0.0]], shells, isolated=True
    )
    with pytest.raises(ValueError, match="isolated atom"):
        _load(path, codec, config)


def test_configured_batch_normalization_uses_explicit_pooled_unequal_graph_statistics(
    tmp_path,
):
    from net.pipeline import FeaturePipeline

    recipe = _recipe("deeph")
    for stage in recipe["pipeline"]["stages"]:
        if stage["transform"] == "scalar_gated_node":
            stage["options"]["normalization"] = "batch"
    config, codec = resolve_configuration(recipe, species_shells={"H": [0]})
    pipeline = FeaturePipeline(config["pipeline"], num_species=1).double().train()
    positions = [
        [[0, 0, 0], [1.0, 0.2, 0.3], [0.1, 1.3, 0.2]],
        [[0, 0, 0], [1.7, 0.3, 0.4], [0.2, 1.9, 0.5], [0.6, 0.1, 2.1]],
    ]
    samples = [
        _load(
            _write(tmp_path / str(i), ["H"] * len(pos), pos, {"H": [0]}, sample=i + 2),
            codec,
            config,
        )
        for i, pos in enumerate(positions)
    ]
    union = union_graph_inputs(samples, layouts=pipeline.input_layouts)
    captured = []

    def check_pooled(module, args, actual):
        x = args[0]
        assert x.shape[0] == 7
        mean, population_var = x.mean(0), x.var(0, correction=0)
        expected = (x - mean) / (population_var + module.eps).sqrt()
        expected = expected * module.weight + module.bias
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(
            module.running_mean, module.momentum * mean, rtol=2e-12, atol=2e-12
        )
        torch.testing.assert_close(
            module.running_var,
            (1 - module.momentum) + module.momentum * x.var(0, correction=1),
            rtol=2e-12,
            atol=2e-12,
        )
        # Two independent per-graph normalizations have different means and
        # cannot accidentally satisfy the intended pooled-batch contract.
        assert not torch.allclose(x[:3].mean(0), x[3:].mean(0))
        captured.append(module)

    handles = [
        module.register_forward_hook(check_pooled)
        for module in pipeline.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
        and module is not pipeline.transforms["local_coordinate_readout"].bn
    ]
    pipeline(union)["hamiltonian"].square().sum().backward()
    assert len(captured) == 5
    for handle in handles:
        handle.remove()
