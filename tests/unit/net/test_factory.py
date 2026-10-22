import pytest
import torch

# ────────────────────────────────────────────────────────────────────────────
# tests/test_dataset_factory.py
# ────────────────────────────────────────────────────────────────────────────
from net.common import Config
from net.e3gnn import E3GNN

# ---------------------------------------------------------------------------
# Paths to the data files
# ---------------------------------------------------------------------------


# ════════════════════════════════════════════════════════════════════════
# 2.  Basic shapes / split sizes
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_dataset_lengths(factory_results):
    train_ds, val_ds, _ = factory_results
    assert len(train_ds) == 2, "train split should contain the two registered pairs"
    assert len(val_ds) == 1, "val split should contain the one registered pair"


# ════════════════════════════════════════════════════════════════════════
# 3.  OrbitalIrrepConfig must hold **all** elements (H, O, Si)
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_orbital_irrep_config_union(factory_results):
    _, _, mapper = factory_results
    elems = set(mapper.orbital_cfg.elements())
    assert {"H", "O", "Si"} <= elems, f"missing elements - got {elems}"
    # quick sanity: highest ℓ not outrageous
    assert mapper.orbital_cfg.max_l() <= 10


# ════════════════════════════════════════════════════════════════════════
# 4.  Mapper instance must be **shared** by all datasets
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_shared_mapper_identity(factory_results):
    train_ds, val_ds, mapper = factory_results
    assert train_ds.mapper is mapper
    assert val_ds.mapper is mapper


# ════════════════════════════════════════════════════════════════════════
# 5.  Mapping availability & vector dimensions
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.unit
def test_mapper_vector_dims_and_roundtrips_cover_union(factory_results):
    _, _, mapper = factory_results
    elements = mapper.orbital_cfg.elements()
    assert {"H", "O", "Si"} <= set(elements)
    for first in elements:
        for second in elements:
            pair = f"{first}-{second}"
            dim_i, dim_j = mapper.block_dims(pair)
            assert mapper.vector_dim(pair) == dim_i * dim_j
            blocks = torch.arange(2 * dim_i * dim_j, dtype=torch.float32).reshape(
                2, dim_i, dim_j
            )
            torch.testing.assert_close(
                mapper.vectors_to_blocks(pair, mapper.blocks_to_vectors(pair, blocks)),
                blocks,
            )


# ════════════════════════════════════════════════════════════════════════
# 6.  Dataset samples contain only known species & coherent tensors
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_sample_coherence(factory_results):
    train_ds, _, mapper = factory_results
    elements = train_ds.species_order
    element_to_idx = {element: index for index, element in enumerate(elements)}

    for x, y in train_ds:
        expected_node_types = torch.tensor(
            [element_to_idx[element] for element in x["atoms_tuple"]],
            dtype=x["node_type_idx"].dtype,
        )
        torch.testing.assert_close(x["node_type_idx"], expected_node_types)

        expected_edge_types = torch.tensor(
            [
                train_ds.edge_type2idx[
                    f"{x['atoms_tuple'][src]}-{x['atoms_tuple'][dst]}"
                ]
                for src, dst in x["edge_index"].T.tolist()
            ],
            dtype=x["edge_type_idx"].dtype,
        )
        torch.testing.assert_close(x["edge_type_idx"], expected_edge_types)
        assert torch.all(x["edge_type_idx"] >= 0)
        assert torch.all(x["edge_type_idx"] < len(mapper.edge_types))

        # target block shapes agree with mapper dims
        for key, block in y["hamiltonian"].pair_blocks.items():
            assert block.shape[1:] == mapper.block_dims(
                key
            ), f"Block dim mismatch for {key}"


# ════════════════════════════════════════════════════════════════════════
# 7.  Forward-pass smoke test with tiny batch (CPU)
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_model_forward_cpu(factory_results):
    train_ds, _, mapper = factory_results

    sample = train_ds[0]
    x, y = sample

    cfg = Config(
        cutoff_radius=7.0,
        l_max=1,
        hidden_base_dim=2,
        hidden_irreps="2x0e+2x0o+1x1e+1x1o",
        n_radial=4,
        radial_layers=[4],
        num_layers_gnn=1,
        neck_depth=1,
        internal_e3mlp_layers=1,
        head_e3mlp_layers=1,
        matrix_targets=["hamiltonian", "overlap", "density"],
        dropout=0.0,
        safety_checks=True,
    )
    model = E3GNN(
        mapper=mapper,
        cfg=cfg,
    )

    out = model(x)

    # we expect all three predicted IrrepsBlockData objects
    for key in ("hamiltonian", "overlap", "density"):
        assert key in out
        assert out[key].pair_vectors, f"{key} vectors should not be empty"
