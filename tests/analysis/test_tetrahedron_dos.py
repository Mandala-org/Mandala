import torch

from analysis.evaluation import _tetrahedron_cdf_pdf


def test_tetrahedron_pdf_is_exactly_zero_outside_nearly_flat_band():
    energies = torch.tensor(
        [[-5.0, -5.0 + 1e-8, -5.0 + 2e-8, -5.0 + 3e-8]],
        dtype=torch.float64,
    )
    grid = torch.tensor([-10.0, -5.1, -4.9, 15.0], dtype=torch.float64)

    cdf, pdf = _tetrahedron_cdf_pdf(energies, grid)

    torch.testing.assert_close(
        cdf, torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(pdf, torch.zeros_like(pdf))


def test_tetrahedron_pdf_integrates_to_one_for_well_separated_energies():
    energies = torch.tensor([[-2.0, -0.5, 0.75, 3.0]], dtype=torch.float64)
    grid = torch.linspace(-3.0, 4.0, 20001, dtype=torch.float64)

    cdf, pdf = _tetrahedron_cdf_pdf(energies, grid)

    assert torch.all(pdf >= 0)
    torch.testing.assert_close(cdf[:, 0], torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(cdf[:, -1], torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(
        torch.trapezoid(pdf, grid),
        torch.ones(1, dtype=torch.float64),
        atol=2e-7,
        rtol=0,
    )
