"""Configuration mistakes and old checkpoint configuration compatibility."""

import pickle
from pathlib import Path

import pytest

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config
from net.configuration import resolve_configuration
from net.e3gnn import E3GNN


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"optimisation": {}}, "sections"),
        ({"data": {"symmetrize": True}}, "data options"),
        ({"objective": {"epoch_weight": "element"}}, "epoch_weight"),
        ({"objective": {"complex_reduction": "absolute"}}, "complex_reduction"),
        ({"objective": {"weights": {"hamiltoinan": 2.0}}}, "unknown operators"),
        ({"data": {"units": {"energy": "Hartree"}}}, "interchange"),
    ],
)
def test_invalid_policies_are_not_silently_ignored(overrides, match):
    with pytest.raises(ValueError, match=match):
        resolve_configuration(
            ROOT / "configs/reference/deeph.yaml",
            species_shells={"H": [0]},
            overrides=overrides,
        )


def test_old_slotted_config_checkpoint_receives_new_optional_defaults():
    cfg = Config(
        hidden_irreps="2x0e",
        l_max=0,
        n_radial=2,
        num_layers_gnn=0,
        neck_depth=1,
        head_e3mlp_layers=1,
        verbosity=0,
    )
    for key in ("pipeline", "objective", "optimization"):
        delattr(cfg, key)
    restored = pickle.loads(pickle.dumps(cfg))
    assert not hasattr(restored, "pipeline")
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": ["1x0e"]}))
    E3GNN(mapper, restored)
    assert restored.pipeline is None
    assert restored.objective is None
    assert restored.optimization is None


@pytest.mark.parametrize(
    "override",
    [
        {"train_on_energy": True},
        {"train_on_forces": True},
        {"spectral_loss_enabled": True},
        {"freeze_backbone_train_heads_only": True},
    ],
)
def test_modular_constructor_rejects_ignored_legacy_scientific_flags(override):
    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": ["1x0e"]}))
    with pytest.raises(ValueError, match="legacy Config overrides"):
        E3GNN(mapper, Config(pipeline={}, **override))
