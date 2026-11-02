"""Physical metric invariance and end-to-end pilot selection/evaluation."""

import h5py
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from net.common import Config
from scripts.train import build_components, main
from scripts.evaluate_operator_checkpoint import evaluate
from utils.units import HARTREE_TO_EV


@pytest.fixture(autouse=True)
def preserve_default_dtype():
    previous = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous)


def fixture_config(root):
    splits = {}
    for sample, split in enumerate(("train", "val", "test")):
        path = root / split
        path.mkdir()
        np.savetxt(path / "element.dat", [1, 6], fmt="%d")
        (path / "orbital_types.dat").write_text("0\n0 1\n")
        np.savetxt(path / "site_positions.dat", [[0, 0, 0], [1, 0.2, 0.3]])
        np.savetxt(path / "lat.dat", np.eye(3) * 8)
        (path / "info.json").write_text('{"isspinful": false}')
        full = np.arange(25).reshape(5, 5) * 0.001
        full = (full + full.T) / 2 + sample * 0.01
        slices = [slice(0, 1), slice(1, 5)]
        with h5py.File(path / "hamiltonians.h5", "w") as stream:
            for i in range(2):
                for j in range(2):
                    stream[str([0, 0, 0, i + 1, j + 1])] = (
                        full[slices[i], slices[j]] * HARTREE_TO_EV
                    )
        splits[split] = [str(path)]
    cfg = Config(
        cutoff_radius=3,
        l_max=2,
        hidden_base_dim=4,
        hidden_irreps="4x0e+2x1o+1x2e",
        num_layers_gnn=1,
        n_radial=4,
        radial_layers=[8],
        neck_depth=1,
        matrix_targets=["hamiltonian"],
        enable_energy=False,
        enable_num_electrons=False,
        train_observables_on_gt=False,
        log_partial_gt_observables=False,
        log_per_irrep_metrics=False,
        log_per_pair_loss_metrics=False,
        log_hamiltonian_irrep_contrib_metrics=False,
        log_hamiltonian_pair_contrib_metrics=False,
        shuffle_snapshot_load_order=False,
        snapshot_cache_dir=None,
        dataset_device="cpu",
        max_epochs=1,
        revert_on_spike=False,
        use_lr_scheduler=False,
        max_wall_clock_seconds=None,
        checkpoint_monitor="val/hamiltonian_mae_elementwise",
        device="cpu",
        gpus=0,
        verbosity=0,
    )
    document = cfg.to_mapping()
    document["data"] = {
        "splits": splits,
        "representation": "orbital_blocks",
        "graph_source": "stored",
        "source_basis": "openmx",
        "source_cartesian_axes": "xyz",
    }
    path = root / "config.yaml"
    OmegaConf.save(OmegaConf.create(document), path)
    return path


def test_elementwise_metric_is_independent_of_diagnostic_flags(tmp_path):
    from net.operator_metrics import matrix_error_metrics

    config = fixture_config(tmp_path)
    _, _, model, datasets, _ = build_components(OmegaConf.load(config))
    model.eval()
    x, y = datasets["val"][0]
    with torch.no_grad():
        prediction = model.predict_matrices(x)["hamiltonian"]
        expected = matrix_error_metrics(
            prediction, y["hamiltonian"], scale=HARTREE_TO_EV
        )
        _, first, _ = model._graph_step((x, y), 0, "val")
        model.cfg.log_hamiltonian_irrep_contrib_metrics = True
        model.cfg.log_hamiltonian_pair_contrib_metrics = True
        _, second, _ = model._graph_step((x, y), 0, "val")
    for key, value in expected.items():
        assert first[f"val/hamiltonian_{key}"].item() == pytest.approx(float(value))
        torch.testing.assert_close(
            first[f"val/hamiltonian_{key}"], second[f"val/hamiltonian_{key}"]
        )
    assert "val/hamiltonian_mae_elementwise_projected" in first


def test_factory_harmonics_use_converted_cartesian_axes_once(tmp_path):
    config = fixture_config(tmp_path)
    _, _, _, datasets, _ = build_components(OmegaConf.load(config))
    x, _ = datasets["train"][0]
    selected = (x["edge_index"][0] == 0) & (x["edge_index"][1] == 1)
    selected &= (x["edge_shift"] == 0).all(dim=0)
    assert int(selected.sum()) == 1
    # Source displacement (1,.2,.3) becomes (.2,.3,1) once. Component-normalized
    # degree-one e3nn harmonics are sqrt(3) times the unit Cartesian vector.
    vector = torch.tensor([0.2, 0.3, 1.0])
    expected = 3**0.5 * vector / vector.norm()
    torch.testing.assert_close(x["edge_sh"][selected, 1:4][0], expected)
    torch.testing.assert_close(x["positions"][1], vector)


@pytest.mark.parametrize("nested_double", [False, True])
def test_shared_training_cli_and_best_checkpoint_evaluator(tmp_path, nested_double):
    config = fixture_config(tmp_path)
    if nested_double:
        document = OmegaConf.load(config)
        del document["dtype"]
        document.training.dtype = "float64"
        OmegaConf.save(document, config)
    output = tmp_path / "trained"
    metrics = main(
        [
            "--config",
            str(config),
            "--output",
            str(output),
            "--accelerator",
            "cpu",
            "--log-dir",
            str(tmp_path / "metrics"),
        ]
    )
    assert np.isfinite(metrics["val/hamiltonian_mae_elementwise"])
    # Evaluation must honor the recipe's declared precision rather than the
    # ambient default or a flat-only lookup that silently rounds float64 state.
    torch.set_default_dtype(torch.float32)
    result = evaluate(config, output / "best_model.pt", split="test", accelerator="cpu")
    assert result["dtype"] == ("float64" if nested_double else "float32")
    assert result["structures"] == 1
    assert result["metrics"]["hamiltonian_mae_elementwise"] > 0
    assert len(result["per_structure"]) == 1
    checkpoint = torch.load(output / "latest_checkpoint.pt", weights_only=False)
    assert checkpoint["epoch_boundary_resume"]["safe_to_resume"]
    assert checkpoint["epoch"] == 1
    if nested_double:
        validation = evaluate(
            config, output / "best_model.pt", split="val", accelerator="cpu"
        )
        assert validation["metrics"]["hamiltonian_mae_elementwise"] == pytest.approx(
            metrics["val/hamiltonian_mae_elementwise"], rel=2e-12, abs=2e-13
        )


def test_segment_runner_caps_attempts_and_evaluates_frozen_test_once(tmp_path):
    from scripts.run_training_segment import main as segment

    config = fixture_config(tmp_path)
    output, records = tmp_path / "model", tmp_path / "records"
    arguments = [
        "--config",
        str(config),
        "--output",
        str(output),
        "--records",
        str(records),
        "--segments",
        "2",
        "--budget-seconds",
        "60",
        "--accelerator",
        "cpu",
        "--evaluate-final-test",
    ]
    segment(arguments + ["--segment", "1"])
    assert not (records / "best_test_evaluation.json").exists()
    checkpoint_time = (output / "latest_checkpoint.pt").stat().st_mtime_ns
    segment(arguments + ["--segment", "2"])
    assert (records / "best_test_evaluation.json").is_file()
    assert (output / "latest_checkpoint.pt").stat().st_mtime_ns == checkpoint_time
    with pytest.raises(FileExistsError, match="already attempted"):
        segment(arguments + ["--segment", "2"])
    with pytest.raises(SystemExit):
        segment(arguments + ["--segment", "3"])
