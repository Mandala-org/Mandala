from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix
from net.artifacts import ArtifactCheckpointCallback
from net.artifacts import save_distance_error_curve_plot, save_dos_comparison_plot


def _make_tiny_block(value: float) -> BlockMatrix:
    atoms = ("H",)
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})
    pair_blocks = {"H-H": torch.tensor([[[value]]], dtype=torch.float32)}
    pair_edges = {"H-H": torch.tensor([[0], [0], [0], [0], [0]], dtype=torch.long)}
    lookup = {(0, 0, 0, 0, 0): ("H-H", 0)}
    return BlockMatrix(
        atoms=atoms,
        atom_counts=Counter(atoms),
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=orbital_cfg,
        basis="e3nn",
    )


def _make_batch_and_module():
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})
    mapper = BlockIrrepMapper(
        orbital_cfg, diagonal=False, device="cpu", dtype=torch.float32
    )
    target = _make_tiny_block(1.0)
    pred = _make_tiny_block(2.0)
    x = {
        "positions": torch.tensor([[0.0, 0.0, 0.0]]),
        "box": torch.eye(3),
        "atoms": ("H",),
        "node_type_idx": torch.tensor([0], dtype=torch.long),
    }
    y = {
        "hamiltonian": target,
        "overlap": target,
        "density": target,
    }

    class DummyModule:
        def __init__(self):
            self.cfg = SimpleNamespace(
                matrix_targets=["hamiltonian", "overlap", "density"],
                print_per_irrep_metrics=False,
                log_interval=1,
                adaptive_log_interval=False,
                enable_energy=False,
                enable_num_electrons=False,
            )
            self.mapper = mapper
            self.device = torch.device("cpu")

        def eval(self):
            return self

        def __call__(self, batch_x):
            return {
                "hamiltonian": pred.to_vectors(mapper),
                "overlap": pred.to_vectors(mapper),
                "density": pred.to_vectors(mapper),
            }

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            return batch

        def on_after_batch_transfer(self, batch, dataloader_idx):
            return batch

    return mapper, (x, y), DummyModule()


def test_checkpoint_callback_saves_artifacts(tmp_path):
    _, batch, module = _make_batch_and_module()

    saved = []
    logged_payloads = []

    class DummyExperiment:
        def __init__(self):
            self.summary = {}

        def log(self, payload):
            logged_payloads.append(payload)

    class DummyTrainer:
        def __init__(self):
            self.current_epoch = 0
            self.sanity_checking = False
            self.callback_metrics = {"val/loss_total": torch.tensor(1.0)}
            self.val_dataloaders = [[batch]]
            self.logger = SimpleNamespace(experiment=DummyExperiment())
            self.optimizers = [SimpleNamespace(param_groups=[{"lr": 1e-3}])]

        def save_checkpoint(self, path):
            saved.append(path)
            torch.save({"path": path}, path)

    trainer = DummyTrainer()
    callback = ArtifactCheckpointCallback(
        tmp_path,
        generate_video=True,
        log_per_irrep_images=True,
        distance_bins=4,
    )

    callback.on_fit_start(trainer, module)
    callback.on_validation_epoch_end(trainer, module)
    callback.on_fit_end(trainer, module)

    assert str(tmp_path / "latest_checkpoint.pt") in saved
    assert str(tmp_path / "best_model.pt") in saved
    assert str(tmp_path / "final_model.pt") in saved
    assert (tmp_path / "dos_comparison_final.png").exists()
    assert (tmp_path / "distance_error_curve_hamiltonian.png").exists()
    assert (tmp_path / "distance_error_curve_hamiltonian.json").exists()
    assert (tmp_path / "per_irrep_images" / "hamiltonian" / "0e.png").exists()
    assert (tmp_path / "per_irrep_images" / "overlap" / "0e.png").exists()
    assert (tmp_path / "per_irrep_images" / "density" / "0e.png").exists()
    assert (tmp_path / "training_progress_hamiltonian.mp4").exists()
    assert any("initial/mae_H" in payload for payload in logged_payloads)
    assert any("mae_H_mod" in payload for payload in logged_payloads)
    assert any("final/mae_H" in payload for payload in logged_payloads)


def test_plot_helpers_write_files(tmp_path):
    H_gt = _make_tiny_block(1.0)
    H_pred = _make_tiny_block(1.25)
    S = _make_tiny_block(1.0)

    dos_path = tmp_path / "dos.png"
    dos_metrics = save_dos_comparison_plot(H_pred, H_gt, S, dos_path)
    assert dos_path.exists()
    assert set(dos_metrics) >= {
        "eig_abs_mean",
        "eig_abs_max",
        "eig_rel_mean",
        "eig_rel_max",
    }

    curve = {
        "bin_centers": [0.0, 1.0],
        "l1_abs": [0.1, 0.2],
        "l2_abs": [0.3, 0.4],
        "l1_rel": [0.5, 0.6],
        "l2_rel": [0.7, 0.8],
        "l1_abs_min": [0.05, 0.1],
        "l1_abs_max": [0.15, 0.25],
        "l2_abs_min": [0.2, 0.3],
        "l2_abs_max": [0.4, 0.5],
        "l1_rel_min": [0.4, 0.5],
        "l1_rel_max": [0.6, 0.7],
        "l2_rel_min": [0.6, 0.7],
        "l2_rel_max": [0.8, 0.9],
        "n_bins": 2,
    }
    curve_path = tmp_path / "curve.png"
    save_distance_error_curve_plot(curve, curve_path)
    assert curve_path.exists()


def test_checkpoint_callback_logs_force_and_rescale_metrics(tmp_path):
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})
    mapper = BlockIrrepMapper(
        orbital_cfg, diagonal=False, device="cpu", dtype=torch.float32
    )
    target = _make_tiny_block(1.0)
    pred_h = _make_tiny_block(1.0)
    pred_s = _make_tiny_block(1.0)
    pred_d = _make_tiny_block(2.0)
    x = {
        "positions": torch.tensor([[0.0, 0.0, 0.0]]),
        "box": torch.eye(3),
        "atoms": ("H",),
        "node_type_idx": torch.tensor([0], dtype=torch.long),
    }
    y = {
        "hamiltonian": target,
        "overlap": target,
        "density": target,
        "num_electrons": torch.tensor(1.0),
        "forces": torch.ones(1, 3),
    }

    logged_payloads = []

    class DummyExperiment:
        def __init__(self):
            self.summary = {}

        def log(self, payload):
            logged_payloads.append(payload)

    class DummyModule:
        def __init__(self):
            self.cfg = SimpleNamespace(
                matrix_targets=["hamiltonian", "overlap", "density"],
                print_per_irrep_metrics=False,
                log_interval=1,
                adaptive_log_interval=False,
                enable_energy=False,
                enable_num_electrons=True,
                enable_forces=True,
                rescale_density_to_num_electrons=True,
                require_exact_edge_match=False,
            )
            self.mapper = mapper
            self.device = torch.device("cpu")

        def eval(self):
            return self

        def __call__(self, batch_x):
            return {
                "hamiltonian": pred_h.to_vectors(mapper),
                "overlap": pred_s.to_vectors(mapper),
                "density": pred_d.to_vectors(mapper),
            }

        def get_forces(self, predictions, positions, box):
            return torch.zeros_like(positions)

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            return batch

        def on_after_batch_transfer(self, batch, dataloader_idx):
            return batch

    class DummyTrainer:
        def __init__(self):
            self.current_epoch = 0
            self.sanity_checking = False
            self.callback_metrics = {"val/loss_total": torch.tensor(0.0)}
            self.val_dataloaders = [[(x, y)]]
            self.logger = SimpleNamespace(experiment=DummyExperiment())
            self.optimizers = [SimpleNamespace(param_groups=[{"lr": 1e-3}])]

        def save_checkpoint(self, path):
            torch.save({"path": path}, path)

    trainer = DummyTrainer()
    callback = ArtifactCheckpointCallback(tmp_path, generate_video=False)
    module = DummyModule()

    callback.on_fit_start(trainer, module)
    callback.on_validation_epoch_end(trainer, module)
    callback.on_fit_end(trainer, module)

    def _find_value(key: str):
        for payload in logged_payloads:
            if key in payload:
                return payload[key]
        raise AssertionError(f"Missing logged key {key!r}")

    assert _find_value("initial/num_electrons_mae_pre_correction") == 1.0
    assert _find_value("initial/num_electrons_mae") == 0.0
    assert _find_value("initial/mae_F") == 1.0
    assert _find_value("initial/mse_F") == 1.0
    assert _find_value("val/num_electrons_mae_pre_correction") == 1.0
    assert _find_value("mae_F") == 1.0
    assert _find_value("mse_F") == 1.0
    assert _find_value("final/num_electrons_mae_pre_correction") == 1.0
    assert _find_value("final/num_electrons_mae") == 0.0
    assert _find_value("final/mae_F") == 1.0
    assert _find_value("final/mse_F") == 1.0
