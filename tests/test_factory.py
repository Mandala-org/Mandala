# ────────────────────────────────────────────────────────────────────────────
# tests/test_dataset_factory.py
# ────────────────────────────────────────────────────────────────────────────
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths to the *actual* data files (relative to repo root)
# ---------------------------------------------------------------------------
PAIR_TRAIN_1 = (
    Path("data/small/H2O/original/H2O.matrix"),
    Path("data/small/H2O/original/H2O.info.out"),
)
PAIR_TRAIN_2 = (
    Path("data/big/silicon/900K/Si_DM"),
    Path("data/big/silicon/900K/info.txt"),
)
PAIR_VAL_1 = (
    Path("data/big/silicon/2700K/Si_DM"),
    Path("data/big/silicon/2700K/info.txt"),
)


# ════════════════════════════════════════════════════════════════════════
# 2.  Basic shapes / split sizes
# ════════════════════════════════════════════════════════════════════════
def test_dataset_lengths(factory_results):
    train_ds, val_ds, _ = factory_results
    assert len(train_ds) == 2, "train split should contain the two registered pairs"
    assert len(val_ds) == 1, "val split should contain the one registered pair"


# ════════════════════════════════════════════════════════════════════════
# 3.  OrbitalIrrepConfig must hold **all** elements (H, O, Si)
# ════════════════════════════════════════════════════════════════════════
def test_orbital_irrep_config_union(factory_results):
    _, _, mapper = factory_results
    elems = set(mapper.orbital_cfg.elements())
    assert {"H", "O", "Si"} <= elems, f"missing elements – got {elems}"
    # quick sanity: highest ℓ not outrageous
    assert mapper.orbital_cfg.max_l() <= 10


# ════════════════════════════════════════════════════════════════════════
# 4.  Mapper instance must be **shared** by all datasets
# ════════════════════════════════════════════════════════════════════════
def test_shared_mapper_identity(factory_results):
    train_ds, val_ds, mapper = factory_results
    assert train_ds.mapper is mapper
    assert val_ds.mapper is mapper


# ════════════════════════════════════════════════════════════════════════
# 5.  Mapping availability & vector dimensions
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "pair",
    ["H-H", "H-O", "O-H", "O-O", "Si-Si"],  # common ordered pairs
)
def test_mapper_vector_dim_positive(factory_results, pair):
    _, _, mapper = factory_results
    # Not all pairs may exist; skip gracefully
    try:
        dim = mapper.vector_dim(pair)
    except KeyError:
        pytest.skip(f"Pair {pair} not present in merged config")
    assert dim > 0, f"vector_dim for {pair} should be > 0 (got {dim})"


# ════════════════════════════════════════════════════════════════════════
# 6.  Dataset samples contain only known species & coherent tensors
# ════════════════════════════════════════════════════════════════════════
def test_sample_coherence(factory_results):
    train_ds, _, mapper = factory_results
    known = set(mapper.orbital_cfg.elements())

    for x_gnn, x_mat, y in train_ds:
        # node_one_hot sanity: argmax maps to known element
        argmax_idx = x_gnn["node_one_hot"].argmax(dim=-1)
        elem_list = mapper.orbital_cfg.elements()
        for idx in argmax_idx.tolist():
            assert elem_list[idx] in known

        # edge_one_hot sanity: indices must match edge_types list length
        assert x_gnn["edge_one_hot"].shape[1] == len(train_ds.edge_types)
        assert x_mat["edge_one_hot"].shape[1] == len(train_ds.edge_types)

        # target vector shapes agree with mapper dims
        for key, vec in y["hamiltonian"].pair_vectors.items():
            assert vec.shape[-1] == mapper.vector_dim(
                key
            ), f"Vec dim mismatch for {key}"


# ════════════════════════════════════════════════════════════════════════
# 7.  Forward-pass smoke test with tiny batch (CPU)
# ════════════════════════════════════════════════════════════════════════
def test_model_forward_cpu(factory_results):
    train_ds, _, mapper = factory_results

    # Lazy import to avoid heavy deps if not needed
    from net.e3gnn import E3GNN
    from net.common import HyperParams

    sample = train_ds[0]
    x_gnn, x_mat, y = sample

    model = E3GNN(
        mapper=mapper,
        edge_types=train_ds.edge_types,
        hp=HyperParams(dropout=0.0, batch_norm=False),  # keep things small
        lr=1e-3,
        device="cpu",
    )

    out = model(
        x_gnn,
        x_mat,
    )

    # we expect all three predicted IrrepsBlockData objects
    for key in ("hamiltonian", "overlap", "density"):
        assert key in out
        assert out[key].pair_vectors, f"{key} vectors should not be empty"
