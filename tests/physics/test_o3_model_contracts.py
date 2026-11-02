"""Symmetry contracts must survive training and mixed, unsorted parities."""

import copy

from e3nn import o3
import pytest
import torch

from net.activations import GateScalarsMLP, GateMagnitudes
from net.common import Config, E3MLP
from net.e3mlp_variants import InvariantFiLMActivation
from net.training_policies import SpeciesIrrepAffine
from tests.contract_support import double_precision, tiny_model, raw_inputs

MIXED = o3.Irreps("1x1e+2x0o+2x0e+1x2o+2x1o")


def test_spin_rotation_promotes_real_native_blocks_instead_of_losing_imaginary_parts():
    from core.operator_layout import OrbitalFrameRotation

    with double_precision():
        angle = torch.tensor(0.73)
        c, s = angle.cos(), angle.sin()
        spatial = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        block = torch.tensor([[1.0, 0.4], [0.4, -0.5]])
        unitary = torch.diag(torch.exp(torch.tensor([-1j, 1j]) * angle / 2))
        expected = unitary.mH @ block.to(unitary) @ unitary
        observed = OrbitalFrameRotation(spinful=True).rotate_blocks(
            block, spatial, [0], [0]
        )
        assert observed.is_complex() and observed.imag.abs().max() > 0.1
        torch.testing.assert_close(observed, expected, atol=2e-14, rtol=2e-14)


def _assert_covariance(module, irreps, values, *, atol=3e-8):
    proper = o3.rand_matrix(dtype=torch.float64)
    for spatial in (proper, -proper, -torch.eye(3, dtype=torch.float64)):
        representation = irreps.D_from_matrix(spatial)
        observed = module(values @ representation.T)
        expected = module(values) @ representation.T
        torch.testing.assert_close(observed, expected, atol=atol, rtol=atol)


@pytest.mark.parametrize(
    "variant",
    [
        "basic",
        "normact",
        "gate",
        "gatemagnitudes",
        "film",
        "resnormact",
        "resgatemagnitudes",
        "bilinear",
    ],
)
def test_mixed_parity_e3mlp_remains_o3_equivariant_after_asymmetric_updates(variant):
    with double_precision(920):
        cfg = Config(e3mlp_variant=variant, nonlin_kind="gate_scalars_mlp")
        module = E3MLP(MIXED, MIXED, MIXED, 2, cfg=cfg, activate_last=True).double()
        values = torch.randn(4, MIXED.dim)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.03)
        probe = torch.linspace(-1.3, 0.7, values.numel()).reshape_as(values)
        _assert_covariance(module, MIXED, values)
        initial = copy.deepcopy(module.state_dict())
        for _ in range(2):
            optimizer.zero_grad()
            (module(values) * probe).sum().backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in module.parameters()
                if p.grad is not None
            )
            optimizer.step()
        assert any(
            not torch.equal(initial[name], parameter)
            for name, parameter in module.named_parameters()
        )
        _assert_covariance(module, MIXED, values)


@pytest.mark.parametrize("kind", ["film", "scalar_gate", "norm_gate"])
def test_unsorted_activation_channels_and_zero_fields_have_finite_o3_vjps(kind):
    with double_precision(922):
        module = {
            "film": lambda: InvariantFiLMActivation(MIXED, hidden_dim=8),
            "scalar_gate": lambda: GateScalarsMLP(MIXED, torch.nn.SiLU()),
            "norm_gate": lambda: GateMagnitudes(
                MIXED, torch.nn.SiLU(), torch.nn.Softplus()
            ),
        }[kind]().double()
        values = torch.randn(3, MIXED.dim)
        values[0] = 0
        values.requires_grad_(True)
        _assert_covariance(module, MIXED, values)
        first = torch.autograd.grad(
            module(values).square().sum(), values, create_graph=True
        )[0]
        second = torch.autograd.grad(first.square().sum(), values)[0]
        assert torch.isfinite(first).all() and torch.isfinite(second).all()


def test_fitted_statistics_cannot_add_a_pseudoscalar_mean():
    with double_precision():
        module = SpeciesIrrepAffine(1, "1x0e+1x0o+1x1e")
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [4.0, 7.0, 8.0, 9.0, 10.0]])
        module.fit(
            [
                {
                    "id": "train",
                    "pair_type": torch.zeros(2, dtype=torch.long),
                    "coupled": values,
                }
            ],
            training_ids=["train"],
        )
        assert module.mean[0, 0] == 2.5
        torch.testing.assert_close(module.mean[0, 1:], torch.zeros(4))
        x = torch.randn(3, 5)
        _assert_covariance(
            lambda x: module(
                x, torch.zeros(3, dtype=torch.long), physical_derivatives=True
            ),
            module.irreps,
            x,
        )


def _sp_orbital_action(rotation, shells):
    # Native s,p orbitals are scalar plus Cartesian x,y,z. No production
    # codec, basis converter or frame rotation generates this expectation.
    return torch.block_diag(
        *(torch.ones(1, 1) if degree == 0 else rotation for degree in shells)
    )


def _spin_action(axis, angle):
    pauli = torch.tensor(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
        dtype=torch.complex128,
    )
    return torch.cos(angle / 2) * torch.eye(2, dtype=torch.complex128) - 1j * torch.sin(
        angle / 2
    ) * torch.einsum("i,ijk->jk", axis.to(pauli), pauli)


@pytest.mark.parametrize(
    "kind", ["deeph_e3", "mace_h", "tensor_messages_projected_attention"]
)
@pytest.mark.parametrize("spinful", [False, True])
def test_raw_geometry_native_operator_covariance_with_fitted_statistics(kind, spinful):
    with double_precision(923):
        shells = {"H": [0], "O": [0, 1]}
        model = tiny_model(kind, species_shells=shells, spinful=spinful).eval()
        positions = torch.tensor([[0.1, 0.2, 0.3], [1.3, 0.4, 0.6], [-0.2, 1.5, 0.8]])
        box = torch.tensor([[4.0, 0.2, 0.3], [0.1, 4.4, 0.2], [0.3, 0.1, 4.7]])
        atoms = ("H", "O", "O")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for trained in range(2):
            base = model(raw_inputs(model, atoms=atoms, positions=positions, box=box))[
                "hamiltonian"
            ].to_blocks(model.mapper)
            assert base.pair_blocks["O-O"].abs().max() > 1e-7
            if spinful:
                assert base.pair_blocks["O-O"].imag.abs().max() > 1e-8
            axis = torch.tensor([1.0, 2.0, -3.0])
            axis = axis / axis.norm()
            angle = torch.tensor(0.73)
            cross = torch.tensor(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            )
            proper = (
                torch.eye(3) * torch.cos(angle)
                + (1 - torch.cos(angle)) * torch.outer(axis, axis)
                + torch.sin(angle) * cross
            )
            for spatial in (proper, -proper, -torch.eye(3)):
                observed = model(
                    raw_inputs(
                        model,
                        atoms=atoms,
                        positions=positions @ spatial.T,
                        box=box @ spatial.T,
                    )
                )["hamiltonian"].to_blocks(model.mapper)
                spin = (
                    torch.eye(2, dtype=torch.complex128)
                    if torch.equal(spatial, -torch.eye(3))
                    else _spin_action(axis, angle)
                )
                for edge, (key, row) in base.lookup.items():
                    first, second = key.split("-")
                    left, right = _sp_orbital_action(
                        spatial, shells[first]
                    ), _sp_orbital_action(spatial, shells[second])
                    if spinful:
                        left, right = torch.kron(spin, left.to(spin)), torch.kron(
                            spin, right.to(spin)
                        )
                    expected = (
                        left.to(base.pair_blocks[key])
                        @ base.pair_blocks[key][row]
                        @ right.to(base.pair_blocks[key]).T.conj()
                    )
                    actual = observed[edge]
                    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-8)
            if trained == 0:
                optimizer.zero_grad()
                loss = sum(
                    (
                        value.real
                        * torch.linspace(-0.9, 1.2, value.numel()).reshape_as(value)
                    ).sum()
                    + 0.1 * value.abs().square().sum()
                    for value in base.pair_blocks.values()
                )
                loss.backward()
                assert all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                )
                before = [p.detach().clone() for p in model.parameters()]
                optimizer.step()
                assert any(
                    not torch.equal(a, b) for a, b in zip(before, model.parameters())
                )


@pytest.mark.parametrize("variant", ["resnormact", "bilinear"])
def test_equivariant_legacy_scale_migration_preserves_uniform_fields_and_rejects_broken_ones(
    variant,
):
    with double_precision():
        cfg = Config(e3mlp_variant=variant)
        module = E3MLP(MIXED, MIXED, MIXED, 2, cfg=cfg)
        state = copy.deepcopy(module.state_dict())
        for name in state:
            if name.endswith(("layerscale", "skip")):
                scales = state[name]
                state[name] = torch.cat(
                    [
                        field[:, None].expand(mul, ir.dim).flatten()
                        for field, (mul, ir) in zip(
                            scales.split([mul for mul, _ in MIXED]), MIXED
                        )
                    ]
                )
        restored = E3MLP(MIXED, MIXED, MIXED, 2, cfg=cfg)
        restored.load_state_dict(state)
        x = torch.randn(2, MIXED.dim)
        torch.testing.assert_close(restored(x), module(x), rtol=0, atol=0)
        broken = copy.deepcopy(state)
        name = next(name for name in state if name.endswith(("layerscale", "skip")))
        broken[name][0] += 0.1
        with pytest.raises(RuntimeError, match="violates rotational symmetry"):
            restored.load_state_dict(broken)
