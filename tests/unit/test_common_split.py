import pytest
import torch
from e3nn.o3 import Irreps, rand_matrix

from net.common import split_in_half, split_into_three, irreps_div


@pytest.mark.unit
def test_split_in_half_happy_path():
    """Test splitting with even multiplicities."""
    irreps = Irreps("64x0e+32x1o+16x2e")
    activations = torch.randn(10, irreps.dim)

    split1, split2 = split_in_half(activations, irreps)

    half_irreps = Irreps("32x0e+16x1o+8x2e")
    assert split1.shape == (10, half_irreps.dim)
    assert split2.shape == (10, half_irreps.dim)


@pytest.mark.unit
def test_split_in_half_odd_multiplicity_raises():
    """Test that an odd multiplicity raises a ValueError."""
    irreps = Irreps("64x0e+31x1o+16x2e")
    activations = torch.randn(10, irreps.dim)

    with pytest.raises(ValueError, match="is not even"):
        split_in_half(activations, irreps)


@pytest.mark.unit
def test_split_into_three_happy_path():
    """Test splitting with multiplicities divisible by 4."""
    irreps = Irreps("64x0e+32x1o+16x2e")
    activations = torch.randn(10, irreps.dim)

    s1, s2, s3 = split_into_three(activations, irreps)

    quarter_irreps = Irreps("16x0e+8x1o+4x2e")
    half_irreps = Irreps("32x0e+16x1o+8x2e")

    assert s1.shape == (10, quarter_irreps.dim)
    assert s2.shape == (10, quarter_irreps.dim)
    assert s3.shape == (10, half_irreps.dim)


@pytest.mark.unit
def test_split_into_three_indivisible_raises():
    """Test that a multiplicity not divisible by 4 raises a ValueError."""
    irreps = Irreps("64x0e+30x1o+16x2e")
    activations = torch.randn(10, irreps.dim)

    with pytest.raises(ValueError, match="is not divisible by 4"):
        split_into_three(activations, irreps)


@pytest.mark.unit
def test_split_values_are_correct():
    """Check that the values are split correctly, not just shapes."""
    irreps = Irreps("4x0e")  # Simple case
    # batch=1, mul=4, dim=1 -> shape (1, 4)
    activations = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # Test split in half
    s1_half, s2_half = split_in_half(activations, irreps)
    assert torch.equal(s1_half, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(s2_half, torch.tensor([[3.0, 4.0]]))

    # Test split into three
    s1_three, s2_three, s3_three = split_into_three(activations, irreps)
    assert torch.equal(s1_three, torch.tensor([[1.0]]))
    assert torch.equal(s2_three, torch.tensor([[2.0]]))
    assert torch.equal(s3_three, torch.tensor([[3.0, 4.0]]))


@pytest.mark.unit
def test_split_into_three_equivariance():
    """Test that the three output parts of split_into_three are equivariant."""
    # 1. Setup irreps and random input tensor
    irreps_in = Irreps("8x0e + 12x1o + 4x2e")  # Multiplicities divisible by 4
    irreps_q = irreps_div(irreps_in, 4)
    irreps_h = irreps_div(irreps_in, 2)

    batch_size = 2
    activations = irreps_in.randn(batch_size, -1)

    # 2. Create a random rotation and corresponding Wigner-D matrices
    R = rand_matrix()
    D_in = irreps_in.D_from_matrix(R)
    D_q = irreps_q.D_from_matrix(R)
    D_h = irreps_h.D_from_matrix(R)

    # 3. Rotate the input activations
    rotated_activations = activations @ D_in.T

    # 4. Split the original and the rotated activations
    p1, p2, p3 = split_into_three(activations, irreps_in)
    rp1, rp2, rp3 = split_into_three(rotated_activations, irreps_in)

    # 5. Rotate the original output parts and check for equality
    p1_rotated = p1 @ D_q.T
    p2_rotated = p2 @ D_q.T
    p3_rotated = p3 @ D_h.T

    assert torch.allclose(p1_rotated, rp1, atol=1e-6)
    assert torch.allclose(p2_rotated, rp2, atol=1e-6)
    assert torch.allclose(p3_rotated, rp3, atol=1e-6)
