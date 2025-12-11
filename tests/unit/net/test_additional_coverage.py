import pytest
import torch
from e3nn.o3 import Irreps

from net.common import Config, build_hidden_irreps
from net.common import RadialMLP
from net.activations import scalar_activation, make_nonlinearity
from net.encoders import NodeEncoder, EdgeEncoder
from net.layers import EdgeUpdateBlock, NodeUpdateBlock, MessageBlock
from core.sparse_math import trace_matmul_sparse
from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig, OrbitalIrrepConfigError


@pytest.mark.unit
def test_node_encoder_shape_and_dtype():
    cfg = Config(safety_checks=True)
    enc = NodeEncoder(node_one_hot_dim=3, cfg=cfg)
    x = torch.tensor([0, 1], dtype=torch.long)
    h = enc(x)
    assert h.shape == (2, enc.irreps_out.dim)
    assert h.dtype == torch.float32


@pytest.mark.unit
def test_edge_encoder_forward():
    cfg = Config(n_radial=16, l_max_gnn=2, safety_checks=True)
    n_types = 2
    sh_ir = Irreps.spherical_harmonics(cfg.l_max_gnn)
    out_ir = Irreps("5x0e")

    enc = EdgeEncoder(n_types, out_ir, cfg)
    E = 4
    edge_type_idx = torch.randint(0, n_types, (E,), dtype=torch.long)
    length_emb = torch.rand(E, cfg.n_radial)
    sh = torch.rand(E, sh_ir.dim)
    h = enc(edge_type_idx, length_emb, sh)
    assert h.shape == (E, out_ir.dim)


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.unit
def test_edge_update_block_shape(residual):
    cfg = Config(edge_update_residual=residual, safety_checks=True)
    hid_ir = Irreps("3x0e")
    blk = EdgeUpdateBlock(hid_ir, cfg)
    N, E = 5, 3
    node = torch.randn(N, hid_ir.dim)
    edge = torch.randn(E, hid_ir.dim)
    # create simple edge_index linking first E nodes
    idx = torch.stack([torch.arange(E), torch.arange(E) + 1])
    out = blk(node, edge, idx)
    assert out.shape == (E, hid_ir.dim)


@pytest.mark.unit
def test_node_update_block_shape():
    cfg = Config(safety_checks=True)
    hid_ir = Irreps("4x0e")
    blk = NodeUpdateBlock(hid_ir, cfg)
    N, E = 4, 2
    node = torch.randn(N, hid_ir.dim)
    edge = torch.randn(E, hid_ir.dim)
    idx = torch.tensor([[0, 2], [1, 3]])
    out = blk(node, edge, idx)
    assert out.shape == (N, hid_ir.dim)


@pytest.mark.unit
def test_message_block_edge_and_node_update():
    cfg = Config(node_update_message_agg="attention", safety_checks=True)
    hid_ir = Irreps("8x0e")
    blk = MessageBlock(hid_ir, cfg)
    N, E = 3, 2
    node = torch.randn(N, hid_ir.dim)
    edge = torch.randn(E, hid_ir.dim)
    idx = torch.tensor([[0, 1], [1, 2]])
    node2, edge2 = blk(node, edge, idx)
    assert node2.shape == (N, hid_ir.dim)
    assert edge2.shape == (E, hid_ir.dim)


@pytest.mark.unit
def test_scalar_activation_and_invalid():
    relu = scalar_activation("ReLU")
    assert isinstance(relu, torch.nn.Module)
    with pytest.raises(ValueError):
        scalar_activation("unknown_act")


@pytest.mark.unit
def test_bad_nonlinearity():
    ir = Irreps("2x0e+1x1o")
    cfg = Config(nonlin_kind="bogus", safety_checks=True)
    with pytest.raises(ValueError):
        make_nonlinearity(ir, cfg)


@pytest.mark.parametrize(
    "l_max_gnn, base_dim, expected",
    [(2, 4, "4x0e+4x0o+2x1e+2x1o+1x2e+1x2o"), (0, 3, "3x0e+3x0o")],
)
@pytest.mark.unit
def test_build_hidden_irreps(l_max_gnn, base_dim, expected):
    ir = build_hidden_irreps(l_max_gnn, base_dim)
    assert str(ir) == expected


@pytest.mark.unit
def test_radial_mlp_output_shape_and_layers():
    mlp = RadialMLP(5, out_dim=2, layers=(3,))
    x = torch.randn(4, 5)
    y = mlp(x)
    assert y.shape == (4, 2)
    # test when layer equals out_dim
    mlp2 = RadialMLP(5, out_dim=5, layers=(5,))
    y2 = mlp2(x)
    assert y2.shape == (4, 5)


@pytest.mark.unit
def test_trace_matmul_sparse_basic():
    # two blocks: identity and 2*identity
    I = torch.eye(2)
    blocks_a = torch.stack([I, I, I, I])
    blocks_b = torch.stack([2 * I, 3 * I, 5 * I, 3 * I])
    edge_idx = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, -1, 1, 0],
        ]
    )
    # Tr(I*3I) + Tr(I*2I) + Tr(I*5I) + Tr(I*3I) = 2*3 + 2*2 + 2*5 + 2*3 = 6+4+10+6=26
    val = trace_matmul_sparse(blocks_a, blocks_b, edge_idx)
    assert torch.isclose(val, torch.tensor(26.0))


@pytest.mark.unit
def test_block_irrep_mapper_roundtrip_and_vector_dim():
    cfg = OrbitalIrrepConfig.from_dict({"A": ["1x0e", "1x1o"], "B": ["2x0e"]})
    mapper = BlockIrrepMapper(cfg)
    # random block for A-B
    d_i, d_j = mapper.block_dims("A-B")
    blk = torch.randn(3, d_i, d_j)
    vec = mapper.blocks_to_vectors(("A", "B"), blk)
    blk2 = mapper.vectors_to_blocks("A-B", vec)
    assert blk2.shape == blk.shape
    # unrecognized key
    with pytest.raises(KeyError):
        mapper.blocks_to_vectors("C-D", blk)


@pytest.mark.unit
def test_orbital_irrep_config_from_dict_and_to_dict():
    d = {"X": "2x0e+1x1o", "Y": ["3x0e"]}
    cfg = OrbitalIrrepConfig.from_dict(d)
    out = cfg.to_dict()
    assert out["X"] == ["2x0e", "1x1o"]
    assert out["Y"] == ["3x0e"]
    # invalid format
    with pytest.raises(OrbitalIrrepConfigError):
        OrbitalIrrepConfig.from_dict({"Z": 42})
