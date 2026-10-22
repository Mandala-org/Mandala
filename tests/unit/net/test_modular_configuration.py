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
    assert restored.pipeline["stages"]
    assert restored.objective is None
    assert restored.optimization is None


@pytest.mark.parametrize(
    "override",
    [
        {"train_on_energy": True, "loss_coef_observables": 1.0},
        {"train_on_forces": True, "enable_forces": True, "loss_coef_forces": 1.0},
        {"spectral_loss_enabled": True},
        {"freeze_backbone_train_heads_only": True},
    ],
)
def test_explicit_graph_uses_common_scientific_configuration(override):
    from net.component_defaults import default_graph

    mapper = BlockIrrepMapper(OrbitalIrrepConfig.from_dict({"H": ["1x0e"]}))
    cfg = Config(
        hidden_irreps="2x0e",
        l_max=0,
        n_radial=2,
        num_layers_gnn=0,
        neck_depth=1,
        head_e3mlp_layers=1,
        verbosity=0,
        **override,
    )
    cfg.pipeline = default_graph(cfg, 1)
    model = E3GNN(mapper, cfg)
    assert all(getattr(model.cfg, name) == value for name, value in override.items())
    if cfg.freeze_backbone_train_heads_only:
        assert not any(
            parameter.requires_grad for parameter in model.node_enc.parameters()
        )
        assert any(
            parameter.requires_grad
            for parameter in model.heads["hamiltonian"].parameters()
        )


def test_orbital_block_configuration_does_not_require_native_shell_codec():
    resolved, codec = Config.resolve(
        Config().to_mapping(), species_shells={"X": [0, 4]}, spinful=False
    )
    assert resolved["species"]["shells"] == {"X": [0, 4]}
    assert codec is None
    assert resolved["pipeline"] is None
