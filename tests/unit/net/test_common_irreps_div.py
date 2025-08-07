import pytest
from e3nn.o3 import Irreps

from net.common import irreps_div


@pytest.mark.unit
@pytest.mark.parametrize(
    "irreps_in, n, irreps_out",
    [
        ("64x0e+32x1o", 4, "16x0e+8x1o"),
        ("10x0e+20x1e", 5, "2x0e+4x1e"),
        ("12x2e", 1, "12x2e"),
        ("12x2e", 12, "1x2e"),
        ("0x0e+16x1o", 8, "2x1o"),
    ],
)
def test_irreps_div_happy_path(irreps_in, n, irreps_out):
    """Tests that irreps_div correctly divides multiplicities."""
    result = irreps_div(Irreps(irreps_in), n)
    assert result == Irreps(irreps_out)


@pytest.mark.unit
@pytest.mark.parametrize(
    "irreps_in, n",
    [
        ("64x0e+31x1o", 4),
        ("10x0e", 3),
    ],
)
def test_irreps_div_raises_on_indivisible(irreps_in, n):
    """Tests that irreps_div raises a ValueError for non-divisible multiplicities."""
    with pytest.raises(ValueError, match="is not divisible by"):
        irreps_div(Irreps(irreps_in), n)


@pytest.mark.unit
@pytest.mark.parametrize("n", [0, -1, 1.5])
def test_irreps_div_raises_on_invalid_n(n):
    """Tests that irreps_div raises a ValueError for invalid n."""
    with pytest.raises(ValueError, match="n must be a positive integer"):
        irreps_div(Irreps("10x0e"), n)
