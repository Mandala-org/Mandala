import pytest
from e3nn.o3 import Irreps

from core.orbital_irrep_config import (
    OrbitalIrrepConfig,
    OrbitalIrrepConfigError,
)


@pytest.fixture()
def minimal_cfg():
    yaml_text = """
orbitals:
  Si: ["2x0e", "2x1o", "1x2e"]
  H:  ["1x0e"]
"""
    return yaml_text


def test_parse_yaml_happy(minimal_cfg):
    cfg = OrbitalIrrepConfig.from_yaml(minimal_cfg)

    assert set(cfg.elements()) == {"H", "Si"}
    assert isinstance(cfg.element_to_irreps["H"], Irreps)
    assert cfg.element_to_irreps["H"].dim == 1  # 1 s orbital
    # Si should have 2*s + 2*p + 1*d = (1) + (3) + (5) reps * multiplicities
    assert cfg.element_to_irreps["Si"].dim == 2 * 1 + 2 * 3 + 1 * 5  # 2+6+5=13


@pytest.fixture()
def alternative_cfg():
    yaml_text = """
orbitals:
    Si: "2s2p1d"
    H:  "1s"
"""
    return yaml_text


def test_parse_yaml_alternative(alternative_cfg):
    cfg = OrbitalIrrepConfig.from_yaml(alternative_cfg)

    assert set(cfg.elements()) == {"H", "Si"}
    assert isinstance(cfg.element_to_irreps["H"], Irreps)
    assert cfg.element_to_irreps["H"].dim == 1  # 1 s orbital
    # Si should have 2*s + 2*p + 1*d = (1) + (3) + (5) reps * multiplicities
    assert cfg.element_to_irreps["Si"].dim == 2 * 1 + 2 * 3 + 1 * 5  # 2+6+5=13


@pytest.mark.parametrize(
    "bad_yaml",
    [
        "not_yaml:",  # missing orbitals
        "orbitals: 123",
        "orbitals:#test# He:# 1x0e",
    ],
)
def test_parse_yaml_errors(bad_yaml):
    with pytest.raises(OrbitalIrrepConfigError):
        conf = OrbitalIrrepConfig.from_yaml(bad_yaml)
        print(conf)


def test_dict_interface():
    cfg_dict = {"C": ["1x0e", "1x1o"], "O": ["2x0e", "2x1o", "1x2e"]}
    cfg = OrbitalIrrepConfig.from_dict(cfg_dict)
    assert cfg.max_l() == 2
    assert cfg.element_to_irreps["C"].dim == 4  # 1 + 3
