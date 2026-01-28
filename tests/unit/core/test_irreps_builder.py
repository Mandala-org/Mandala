import pytest
from e3nn.o3 import Irreps

from core.irreps_builder import IrrepsAutoBuilder, IrrepsBuilderError


@pytest.fixture(scope="module")
def builder():
    return IrrepsAutoBuilder(l_max=2, base_dim=16)  # 16 -> 8 -> 4 multiplicities


@pytest.mark.unit
def test_hidden_irreps_shape(builder):
    ir = builder.hidden_irreps
    # expect 0e/0o 16 each, 1e/1o 8 each, 2e/2o 4 each
    assert ir.dim == (16 + 16) * 1 + (8 + 8) * 3 + (4 + 4) * 5  # 32 + 48 + 40 = 120


@pytest.mark.unit
def test_sh_irreps(builder):
    sh = builder.sh_irreps
    assert sh.dim == 1 + 3 + 5  # up to l=2 -> total dim 9


@pytest.mark.unit
def test_tp_path_ok(builder):
    scalars = Irreps("1x0e")
    vectors = Irreps("1x1o")
    builder.require_tp_path(scalars, vectors, Irreps("1x1o"))  # should pass


@pytest.mark.unit
def test_tp_path_bad(builder):
    vec_l1 = Irreps("1x1o")
    vec_l2 = Irreps("1x1o")
    with pytest.raises(IrrepsBuilderError):
        # two odd vectors cannot couple to parity-odd scalar
        builder.require_tp_path(vec_l1, vec_l2, Irreps("1x0o"))  # should fail
