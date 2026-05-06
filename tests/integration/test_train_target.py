import pytest
from pathlib import Path
import torch


from net.common import Config
from net.e3gnn import E3GNN
from data.factory import DatasetFactory


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
    cfg = Config(
        device="cpu",
        train_on_energy=False,
        train_on_num_electrons=False,
        safety_checks=True,
        cutoff_radius=7.0,
    )
    fac = DatasetFactory(cfg)
    for m, i in paths:
        fac.add_snapshot(m, i)
    train_ds, _, mapper = fac.create()
    x, y_matrix = train_ds[0]

    # Also create an irreps version of y for convenience in tests
    y_irreps = {
        "hamiltonian": y_matrix["hamiltonian"].to_vectors(mapper),
        "overlap": y_matrix["overlap"].to_vectors(mapper),
        "density": y_matrix["density"].to_vectors(mapper),
        "energy": y_matrix["energy"],
        "num_electrons": y_matrix["num_electrons"],
        "forces": y_matrix["forces"],
        "stress": y_matrix["stress"],
    }
    return x, y_irreps, y_matrix, mapper, cfg


# Define the matrix target subsets to test
matrix_target_subsets = [
    ["hamiltonian", "overlap", "density"],
    ["hamiltonian", "density"],
    ["hamiltonian"],
    ["density"],
    ["overlap"],
]


@pytest.mark.parametrize("train_target", ["matrix", "irreps"])
@pytest.mark.parametrize("matrix_targets", matrix_target_subsets)
def test_model_training_configurations(prepared_data, train_target, matrix_targets):
    """
    Tests that the model correctly:
    1. Constructs heads for a given subset of matrix targets.
    2. Runs a training step for both 'matrix' and 'irreps' train_target settings.
    3. Computes the loss correctly for the specified subset.
    """
    x, y_irreps, y_matrix, mapper, cfg = prepared_data

    cfg.matrix_targets = matrix_targets
    cfg.train_target = train_target
    model = E3GNN(mapper, cfg)

    # 1. Check that the model only has heads for the specified targets
    assert sorted(list(model.heads.keys())) == sorted(matrix_targets)

    # Select the correct target format based on the test parameter
    y = y_matrix if train_target == "matrix" else y_irreps

    # 2. Manually calculate the expected matrix loss for the subset
    with torch.no_grad():
        preds_irreps = model(x)
        expected_loss_matrix = torch.tensor(0.0)

        if train_target == "matrix":
            # Loss is on BlockMatrix blocks
            preds_matrix = {
                name: preds_irreps[name].to_blocks(mapper) for name in matrix_targets
            }
            if cfg.symmetrize_output:
                for name, matrix in preds_matrix.items():
                    preds_matrix[name] = (matrix + matrix.transpose()) * 0.5
            for name in matrix_targets:
                p_blocks = preds_matrix[name].pair_blocks
                t_blocks = y[name].pair_blocks

                p_edges = preds_matrix[name].pair_edges
                t_edges = y[name].pair_edges

                # Iterate over target keys only (matching model behavior)
                for key in t_blocks.keys():
                    if key not in p_blocks.keys():
                        raise ValueError(
                            f"Edge key {key} missing in predicted {name} matrix."
                        )

                    preds = p_blocks[key]
                    targets = t_blocks[key]

                    target_n = targets.shape[0]
                    if preds.shape[0] < target_n:
                        raise ValueError(
                            f"Predicted blocks for {name}, key {key} are too short: "
                            f"pred_len={preds.shape[0]} target_len={target_n}"
                        )
                    preds = preds[:target_n]
                    targets = targets[:target_n]

                    if cfg.safety_checks:
                        if not torch.equal(
                            p_edges[key][:, :target_n], t_edges[key][:, :target_n]
                        ):
                            raise ValueError(
                                f"Edge indices for predicted and target {name} matrices do not match."
                            )

                    expected_loss_matrix += torch.mean((preds - targets) ** 2)
        else:  # train_target == "irreps"
            # Loss is on IrrepsBlockData vectors
            for name in matrix_targets:
                p_vecs = preds_irreps[name].pair_vectors
                t_vecs = y[name].pair_vectors

                # Iterate over target keys only (matching model behavior)
                for key in t_vecs.keys():
                    if key not in p_vecs.keys():
                        raise ValueError(
                            f"Edge key {key} missing in predicted {name} vectors."
                        )

                    preds = p_vecs[key]
                    targets = t_vecs[key]

                    target_n = targets.shape[0]
                    if preds.shape[0] < target_n:
                        raise ValueError(
                            f"Predicted vectors for {name}, key {key} are too short: "
                            f"pred_len={preds.shape[0]} target_len={target_n}"
                        )
                    preds = preds[:target_n]
                    targets = targets[:target_n]

                    expected_loss_matrix += torch.mean((preds - targets) ** 2)

    # 3. Run a training step and capture the logged metrics
    logged_metrics = {}

    def mock_log_dict(metrics, **kwargs):
        logged_metrics.update(metrics)

    model.log_dict = mock_log_dict
    loss = model.training_step((x, y), batch_idx=0)

    assert loss is not None
    assert loss.requires_grad

    # 4. Assert that the logged matrix loss is close to the manually calculated one
    assert "train/loss_matrix_total" in logged_metrics
    assert torch.allclose(
        logged_metrics["train/loss_matrix_total"],
        expected_loss_matrix,
        atol=1e-6,
    )
    print(
        f"Success: train_target='{train_target}', matrix_targets={matrix_targets}. "
        f"Manual Loss: {expected_loss_matrix:.6f}, Model Loss: {logged_metrics['train/loss_matrix_total']:.6f}"
    )
