import pytest
from pathlib import Path
import torch

from net.common import Config
from net.e3gnn import E3GNN
from data.factory import DatasetFactory
from data.block_matrix import BlockMatrix, IrrepsBlockData


@pytest.fixture(scope="module")
def prepared_data():
    """
    Fixture to load H2O data and prepare a dataset, mapper, and a single
    (x, y) sample tuple. This is shared across all tests in this module.
    """
    paths = [
        (
            Path("data/small/H2O/original/H2O.matrix"),
            Path("data/small/H2O/original/H2O.info.out"),
        )
    ]
    # The default config creates targets in 'irreps' format.
    cfg = Config(device="cpu")
    fac = DatasetFactory(cfg)
    for m, i in paths:
        fac.add_snapshot(m, i)
    train_ds, _, mapper = fac.create()
    x, y = train_ds[0]
    return x, y, mapper


@pytest.mark.parametrize("train_target", ["matrix", "irreps"])
def test_train_target_option(prepared_data, train_target):
    """
    Tests that the `train_target` config option correctly sets the target data type
    and that the training step runs without errors.
    """
    x, y_irreps, mapper = prepared_data
    cfg = Config(train_target=train_target, device="cpu")
    model = E3GNN(mapper, cfg)

    # The fixture `y` is in irreps format. If we're testing the 'matrix'
    # case, we need to convert the targets to BlockMatrix format.
    if train_target == "matrix":
        y = {
            "hamiltonian": y_irreps["hamiltonian"].to_blocks(mapper),
            "overlap": y_irreps["overlap"].to_blocks(mapper),
            "density": y_irreps["density"].to_blocks(mapper),
            "energy": y_irreps["energy"],
            "num_electrons": y_irreps["num_electrons"],
        }
        target_hamiltonian = y["hamiltonian"]
        assert isinstance(target_hamiltonian, BlockMatrix)
    else:
        y = y_irreps
        target_hamiltonian = y["hamiltonian"]
        assert isinstance(target_hamiltonian, IrrepsBlockData)

    # Run a training step and assert it completes successfully
    loss = model.training_step((x, y), batch_idx=0)
    assert loss is not None
    assert loss.requires_grad
    print(f"Successfully ran training step with train_target='{train_target}'")


# Define the 5 specific subsets to test
matrix_target_subsets = [
    ["hamiltonian", "overlap", "density"],
    ["hamiltonian", "density"],
    ["hamiltonian"],
    ["density"],
    ["overlap"],
]


@pytest.mark.parametrize("matrix_targets", matrix_target_subsets)
def test_matrix_subset_training(prepared_data, matrix_targets):
    """
    Tests that the model correctly constructs heads and computes loss for a
    given subset of matrix targets.
    """
    x, y, mapper = prepared_data
    cfg = Config(
        matrix_targets=matrix_targets,
        train_target="matrix",
        train_on_energy=False,
        train_on_num_electrons=False,
        loss_coef_energy=0.0,
        loss_coef_num_electrons=0.0,
        device="cpu",
    )
    model = E3GNN(mapper, cfg)

    # 1. Check that the model only has heads for the specified targets
    assert sorted(list(model.heads.keys())) == sorted(matrix_targets)

    # 2. Run a training step and capture the logged metrics
    # We mock the logger to intercept the metrics without a full Trainer.
    logged_metrics = {}

    def mock_log_dict(metrics, **kwargs):
        logged_metrics.update(metrics)

    model.log_dict = mock_log_dict
    model.training_step((x, y), batch_idx=0)

    # 3. Manually calculate the expected matrix loss for the subset
    with torch.no_grad():
        preds = model(x)
        expected_loss_matrix = torch.tensor(0.0)
        for name in matrix_targets:
            p_vecs = preds[name].pair_vectors
            t_vecs = y[name].pair_vectors
            for key in p_vecs:
                expected_loss_matrix += torch.mean((p_vecs[key] - t_vecs[key]) ** 2)

    # 4. Assert that the logged matrix loss is close to the manually calculated one
    assert "train_loss_matrix" in logged_metrics
    assert torch.allclose(
        logged_metrics["train_loss_matrix"],
        expected_loss_matrix,
        atol=1e-6,
    )
    print(
        f"Successfully tested training with matrix_targets = {matrix_targets}. "
        f"Manual Loss: {expected_loss_matrix:.6f}, Model Loss: {logged_metrics['train_loss_matrix']:.6f}"
    )
