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
                matrix_targets=["hamiltonian", "overlap", "density"]
            )
            self.mapper = mapper

        def eval(self):
            return self

        def __call__(self, batch_x):
            return {
                "hamiltonian": pred.to_vectors(mapper),
                "overlap": pred.to_vectors(mapper),
                "density": pred.to_vectors(mapper),
            }

    return mapper, (x, y), DummyModule()


def test_checkpoint_callback_saves_artifacts(tmp_path):
    _, batch, module = _make_batch_and_module()

    saved = []

    class DummyTrainer:
        def __init__(self):
            self.current_epoch = 0
            self.sanity_checking = False
            self.callback_metrics = {"val/loss_total": torch.tensor(1.0)}
            self.val_dataloaders = [[batch]]
            self.logger = SimpleNamespace(experiment=SimpleNamespace(summary={}))

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
    assert (tmp_path / "per_irrep_images" / "hamiltonian" / "0e.png").exists()
    assert (tmp_path / "per_irrep_images" / "overlap" / "0e.png").exists()
    assert (tmp_path / "per_irrep_images" / "density" / "0e.png").exists()
    assert (tmp_path / "training_progress_hamiltonian.gif").exists()


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
    }
    curve_path = tmp_path / "curve.png"
    save_distance_error_curve_plot(curve, curve_path)
    assert curve_path.exists()
