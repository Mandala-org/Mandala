from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from data.factory import DatasetFactory  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402
from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402


# This one-batch workflow intentionally avoids worker processes and external loggers.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.workflow,
    pytest.mark.filterwarnings(
        "ignore:The 'train_dataloader' does not have many workers.*"
    ),
    pytest.mark.filterwarnings(
        "ignore:The 'val_dataloader' does not have many workers.*"
    ),
    pytest.mark.filterwarnings(
        r"ignore:You called `self.log\('lr'.*but have no logger configured.*"
    ),
    pytest.mark.filterwarnings(
        r"ignore:You called `self.log\('grad_norm'.*but have no logger configured.*"
    ),
]


@pytest.mark.parametrize(
    "train_target",
    ["irreps", "matrix"],
)
def test_full_training_workflow(
    train_target, tmp_path, monkeypatch, small_angular_snapshot_e3nn
):
    """
    Tests the full training workflow from data loading to model training for one epoch.
    This test is parameterized to run for both 'irreps' and 'matrix' training targets.
    """
    # 1. Create a configuration for a quick test run
    cfg = Config(
        # Data
        cutoff_radius=8.0,
        # Model
        l_max=1,
        hidden_base_dim=2,
        hidden_irreps="2x0e+2x0o+1x1e+1x1o",
        n_radial=4,
        radial_layers=[4],
        num_layers_gnn=1,
        neck_depth=1,
        internal_e3mlp_layers=1,
        head_e3mlp_layers=1,
        # Training
        train_target=train_target,
        max_epochs=1,
        gpus=0,
        num_workers=0,
        log_on_step=False,
        log_on_epoch=True,
        safety_checks=True,
        log_per_irrep_metrics=False,
        log_hamiltonian_irrep_contrib_metrics=False,
        log_hamiltonian_pair_contrib_metrics=False,
        verbosity=0,
    )

    # 2. Create datasets using the factory
    fac = DatasetFactory(cfg)

    monkeypatch.setattr(
        Snapshot,
        "from_openmx",
        lambda *args, **kwargs: small_angular_snapshot_e3nn,
    )
    monkeypatch.setattr(
        DatasetFactory,
        "_load_info",
        lambda self, *paths: SimpleNamespace(orbital_set={"H": "1s1p"}),
    )
    fac.add_snapshot("synthetic-train.matrix", "synthetic-train.out", "train")
    fac.add_snapshot("synthetic-val.matrix", "synthetic-val.out", "val")

    train_ds, val_ds, mapper = fac.create()

    assert train_ds is not None
    assert val_ds is not None

    # Create DataLoaders
    def _dl(ds, shuffle=False):
        return DataLoader(
            ds or [],
            batch_size=1,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: b[0],
        )

    train_loader = _dl(train_ds, shuffle=True)
    val_loader = _dl(val_ds, shuffle=False)

    # 3. Create the model
    model = E3GNN(mapper=mapper, cfg=cfg)

    # 4. Create a trainer and run for one epoch
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,  # Disable logging for tests
        callbacks=[],
        default_root_dir=str(tmp_path / "lightning"),
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )

    # 5. Run the training
    try:
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    except Exception as e:
        pytest.fail(
            f"Training failed for train_target='{train_target}' with error: {e}"
        )
