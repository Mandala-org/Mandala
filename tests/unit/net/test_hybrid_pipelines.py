"""Configuration-only mixtures exercise independent architecture streams."""

from pathlib import Path
import io
import pytest
import torch
from e3nn import o3
from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config
from net.configuration import resolve_configuration
from net.e3gnn import E3GNN
from net.pipeline import FeatureSpec, build_transform

ROOT = Path(__file__).resolve().parents[3] / "configs" / "hybrids"


def inputs():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.3, 0.2], [-0.4, 1.3, 0.7]])
    edges = torch.tensor([(i, j) for i in range(3) for j in range(3)]).T
    vectors = positions[edges[1]] - positions[edges[0]]
    lengths = vectors.norm(dim=-1)
    onsite = edges[0] == edges[1]
    return dict(
        node_one_hot=torch.ones(3, 1),
        node_type_idx=torch.zeros(3, dtype=torch.long),
        edge_one_hot=torch.ones(9, 1),
        edge_type_idx=torch.zeros(9, dtype=torch.long),
        edge_index=edges,
        edge_vectors=vectors,
        edge_lengths=lengths,
        node_batch=torch.zeros(3, dtype=torch.long),
        edge_batch=torch.zeros(9, dtype=torch.long),
        selfloop_edge=onsite,
        message_edge_index=edges[:, ~onsite],
        message_vectors=vectors[~onsite],
        message_lengths=lengths[~onsite],
        bond_frames=torch.eye(3).repeat(9, 1, 1),
    )


def build(path):
    config, codec = resolve_configuration(path, species_shells={"H": [0, 1]})
    mapper = BlockIrrepMapper(
        OrbitalIrrepConfig.from_dict({"H": "1s1p"}), dtype=torch.get_default_dtype()
    )
    return (
        E3GNN(
            mapper,
            Config(pipeline=config["pipeline"], optimization=config["optimization"]),
        ),
        codec,
    )


@pytest.mark.parametrize("path", sorted(ROOT.glob("*.yaml")), ids=lambda p: p.stem)
def test_config_only_hybrid_forward_gradient_optimizer_and_serialization(path):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(13)
        model, codec = build(path)
        x = inputs()
        output = model(x)["hamiltonian"]
        assert output.shape == (9, codec.packed_dim)
        assert torch.isfinite(output).all()
        optimizer = model.configure_optimizers()
        before = {
            key: value.detach().clone() for key, value in model.named_parameters()
        }
        for _ in range(2):
            optimizer.zero_grad()
            model(x)["hamiltonian"].square().mean().backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            optimizer.step()
        assert any(
            not torch.equal(before[key], value)
            for key, value in model.named_parameters()
        )
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored, _ = build(path)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(
            restored(x)["hamiltonian"], model(x)["hamiltonian"], rtol=0, atol=0
        )
    finally:
        torch.set_default_dtype(old)


def test_cyclic_and_xyz_features_require_explicit_basis_conversion():
    xyz = FeatureSpec("node", "irreps", "2x0e+2x1o", angular_basis="xyz")
    cyclic = FeatureSpec("node", "irreps", "2x0e+2x1o", angular_basis="yzx")
    with pytest.raises(ValueError, match="Angular basis mismatch"):
        build_transform("concat", {}, [xyz, cyclic])
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        forward, spec = build_transform(
            "irrep_cartesian_basis", {"target": "yzx"}, [xyz]
        )
        backward, _ = build_transform("irrep_cartesian_basis", {"target": "xyz"}, spec)
        values = torch.randn(5, 8, requires_grad=True)
        torch.testing.assert_close(
            backward(forward(values)), values, atol=1e-12, rtol=1e-12
        )
        vectors = torch.randn(5, 3)
        sh = o3.spherical_harmonics(
            "0e+1o", vectors, normalize=True, normalization="component"
        )
        adapter, _ = build_transform(
            "irrep_cartesian_basis",
            {"target": "yzx"},
            [FeatureSpec("edge", "irreps", "0e+1o")],
        )
        expected = o3.spherical_harmonics(
            "0e+1o", vectors[:, [1, 2, 0]], normalize=True, normalization="component"
        )
        torch.testing.assert_close(adapter(sh), expected, atol=1e-12, rtol=1e-12)
        backward(forward(values)).sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))
    finally:
        torch.set_default_dtype(old)


def test_local_readout_is_invariant_under_joint_spatial_and_bond_frame_rotation():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model, _ = build(ROOT / "equivariant_nodes_bond_frame_readout.yaml")
        model.eval()
        x = inputs()
        rotation = o3.rand_matrix()
        rotated = dict(x)
        rotated["edge_vectors"] = x["edge_vectors"] @ rotation.T
        rotated["message_vectors"] = x["message_vectors"] @ rotation.T
        rotated["bond_frames"] = rotation[None] @ x["bond_frames"]
        torch.testing.assert_close(
            model(rotated)["hamiltonian"], model(x)["hamiltonian"], atol=3e-8, rtol=3e-8
        )
    finally:
        torch.set_default_dtype(old)


def test_global_hybrids_have_declared_native_orbital_rotation_covariance():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        from net.operator_codecs import OrbitalFrameRotation

        for path in sorted(ROOT.glob("*.yaml")):
            if path.name.startswith("equivariant_nodes"):
                continue
            model, codec = build(path)
            model.eval()
            x = inputs()
            rotation = o3.rand_matrix()
            rotated = {
                **x,
                "edge_vectors": x["edge_vectors"] @ rotation.T,
                "message_vectors": x["message_vectors"] @ rotation.T,
            }
            observed = codec.unpack_blocks(("H", "H"), model(rotated)["hamiltonian"])
            blocks = codec.unpack_blocks(("H", "H"), model(x)["hamiltonian"])
            expected = OrbitalFrameRotation(dtype=torch.float64).rotate_blocks(
                blocks, rotation.T, [0, 1], [0, 1]
            )
            torch.testing.assert_close(observed, expected, rtol=1e-7, atol=1e-7)
    finally:
        torch.set_default_dtype(old)


def test_partitioned_readout_handles_rectangular_multispecies_and_shifted_self():
    from net.transforms.mandala_components import PartitionedOperatorReadout

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        mapper = BlockIrrepMapper(
            OrbitalIrrepConfig.from_dict({"H": "1s", "O": "1s1p"}), dtype=torch.float64
        )
        readout = PartitionedOperatorReadout(
            "2x0e+1x1o",
            "2x0e+1x1o",
            mapper=mapper,
            species_shells={"H": [0], "O": [0, 1]},
            irreps_neck="2x0e+1x1o",
            config={
                "neck_depth": 1,
                "head_e3mlp_layers": 1,
                "e3mlp_variant": "basic",
                "head_e3mlp_variant": "basic",
                "separate_shifted_self": False,
            },
        )
        node = torch.randn(2, 5, requires_grad=True)
        edge = torch.randn(5, 5, requires_grad=True)
        indices = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 0, 1]])
        pair_types = torch.tensor([0, 0, 1, 2, 3])
        output = readout(
            node,
            edge,
            indices,
            pair_types,
            torch.tensor([True, False, False, False, True]),
        )
        assert output.shape == (5, 16)
        assert output[2, readout.codec.packed_mask(("H", "O"))].numel() == 4
        output.square().sum().backward()
        assert torch.isfinite(node.grad).all() and torch.isfinite(edge.grad).all()
        assert torch.count_nonzero(
            edge.grad[1]
        )  # shifted self follows offsite branch when unsplit
    finally:
        torch.set_default_dtype(old)


def test_partitioned_readout_pair_ids_follow_resolved_atomic_order_not_mapper_order():
    """H,C atomic-number order differs from the mapper's C,H alphabetic order."""
    from net.transforms.mandala_components import PartitionedOperatorReadout

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        mapper = BlockIrrepMapper(
            OrbitalIrrepConfig.from_dict({"C": "1s1p", "H": "1s"}), dtype=torch.float64
        )
        assert tuple(mapper.edge_types) == ("C-C", "C-H", "H-C", "H-H")
        readout = PartitionedOperatorReadout(
            "2x0e+1x1o",
            "2x0e+1x1o",
            mapper=mapper,
            species_shells={"C": [0, 1], "H": [0]},
            species_order=["H", "C"],
            irreps_neck="2x0e+1x1o",
            config={"neck_depth": 1, "head_e3mlp_layers": 1},
        )
        pairs = ("H-H", "H-C", "C-H", "C-C")
        assert readout.edge_types == pairs
        assert tuple(readout.codec.species_shells) == ("H", "C")
        known = {
            pair: torch.arange(
                mapper.block_dims(pair)[0] * mapper.block_dims(pair)[1],
                dtype=torch.float64,
            ).reshape(1, *mapper.block_dims(pair))
            + 10 * (i + 1)
            for i, pair in enumerate(pairs)
        }

        class KnownPairHead(torch.nn.Module):
            separate_shifted_self = True

            def __init__(self):
                super().__init__()
                self.mapper = mapper

            def forward(self, node, edge, pair_edges, partitions):
                # Each named projection must receive the actual associated edge.
                for i, pair in enumerate(pairs):
                    assert partitions[pair]["global_idx"].tolist() == [i]
                return {
                    pair: mapper.blocks_to_vectors(pair, block)
                    for pair, block in known.items()
                }

        readout.head = KnownPairHead()
        output = readout(
            torch.zeros(2, 5),
            torch.zeros(4, 5),
            torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
            torch.arange(4),
            torch.tensor([True, False, False, True]),
        )
        for i, pair in enumerate(pairs):
            restored = readout.codec.unpack_blocks(
                tuple(pair.split("-")), output[i : i + 1]
            )
            # Native s and p orbital entries have x,y,z order, also used by the
            # mapper; this independently checks rectangular channel placement.
            torch.testing.assert_close(restored, known[pair], rtol=1e-12, atol=1e-12)
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize("order", [["H", "H"], ["H"], ["H", "C", "O"]])
def test_partitioned_readout_rejects_invalid_frozen_species_vocabulary(order):
    from net.transforms.mandala_components import PartitionedOperatorReadout

    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": "1s", "C": "1s1p"}))
    with pytest.raises(ValueError, match="exactly once"):
        PartitionedOperatorReadout(
            "1x0e",
            "1x0e",
            mapper=mapper,
            species_shells={"H": [0], "C": [0, 1]},
            species_order=order,
            irreps_neck="1x0e",
        )
