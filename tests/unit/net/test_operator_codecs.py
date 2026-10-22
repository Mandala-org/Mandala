"""Native-shell codecs: analytic checks and optional independent source oracles.

Set MANDALA_DEEPHE3_REFERENCE to the pinned upstream checkout to enable the
live differential tests. They execute upstream code, never a local duplicate.
"""

import importlib
import os
from pathlib import Path

import pytest
import torch
from e3nn.o3 import rand_matrix

from core.operator_layout import (
    OrbitalFrameRotation,
    ShellTensorCodec,
    build_shell_targets,
)


@pytest.fixture(autouse=True)
def double_precision_defaults():
    # e3nn constructs SO(3) generators in the global default dtype, including
    # when a double rotation tensor is supplied. Match the reference runtime.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def source(monkeypatch):
    root = os.getenv("MANDALA_DEEPHE3_REFERENCE")
    if not root or not (Path(root) / "deephe3/e3modules.py").is_file():
        pytest.skip(
            "Set MANDALA_DEEPHE3_REFERENCE to enable the independent upstream oracle."
        )
    monkeypatch.syspath_prepend(root)
    modules = importlib.import_module("deephe3.e3modules")
    utils = importlib.import_module("deephe3.utils")
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield modules, utils
    torch.set_default_dtype(old)


@pytest.mark.parametrize("spinful", [False, True])
@pytest.mark.parametrize("sort_irreps", [False, True])
def test_codec_round_trip_and_gradients(spinful, sort_irreps):
    torch.manual_seed(817)
    codec = ShellTensorCodec(
        [(0, 1), (1, 2), (2, 0), (2, 2)],
        spinful=spinful,
        sort_irreps=sort_irreps,
        dtype=torch.float64,
    )
    x = torch.randn(2, 3, codec.coupled_dim, dtype=torch.float64, requires_grad=True)
    reconstructed = codec.encode(codec.decode(x))
    torch.testing.assert_close(reconstructed, x, rtol=2e-13, atol=2e-13)
    gradient = torch.autograd.grad(reconstructed.square().sum(), x)[0]
    torch.testing.assert_close(gradient, 2 * x, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("spinful", [False, True])
def test_shared_species_channels_pack_masks_and_reconstruction(spinful):
    codec = ShellTensorCodec.from_species_shells(
        {"H": [0], "O": [0, 1, 2]}, spinful=spinful, dtype=torch.float64
    )
    dtype = torch.complex128 if spinful else torch.float64
    factor = 2 if spinful else 1
    blocks = torch.randn(3, factor, 9 * factor, dtype=dtype, requires_grad=True)
    packed, mask = codec.pack_blocks(("H", "O"), blocks, return_mask=True)
    assert mask.dtype == torch.bool
    assert mask.shape == packed.shape
    assert mask[0].sum() == factor**2 * 9
    assert not mask.all()  # shared O-O channels absent on H-O
    torch.testing.assert_close(codec.unpack_blocks(("H", "O"), packed), blocks)
    decoded = codec.decode_blocks(("H", "O"), codec.encode_blocks(("H", "O"), blocks))
    torch.testing.assert_close(decoded, blocks, rtol=2e-13, atol=2e-13)
    grad = torch.autograd.grad(decoded.abs().square().sum(), blocks)[0]
    torch.testing.assert_close(grad, 2 * blocks, rtol=3e-13, atol=3e-13)


def test_explicit_shared_channel_selection_and_validation():
    shells = {"H": [0, 1], "O": [0, 0, 1, 2]}
    selected = [{"H O": [1, 3], ("O", "O"): [2, 3]}]
    codec = ShellTensorCodec.from_species_shells(shells, channel_assignments=selected)
    assert codec.shell_pairs == ((1, 2),)
    blocks = torch.ones(1, 4, 10)
    recovered = codec.unpack_blocks(("H", "O"), codec.pack_blocks(("H", "O"), blocks))
    expected = torch.zeros_like(blocks)
    expected[:, 1:, 5:] = 1
    torch.testing.assert_close(recovered, expected)
    with pytest.raises(ValueError, match="identical shell angular"):
        build_shell_targets(shells, requested=[{"H O": [0, 0], "O O": [2, 3]}])
    with pytest.raises(ValueError, match="unknown species or shell"):
        build_shell_targets(shells, requested=[{"H O": [4, 0]}])
    with pytest.raises(ValueError, match="all, diag, or offdiag"):
        build_shell_targets(shells, block_kind="unknown")


def test_soc_constants_and_masks_survive_module_dtype_transfer():
    codec = ShellTensorCodec.from_species_shells(
        {"H": [0], "O": [0, 1, 2]},
        spinful=True,
        sort_irreps=True,
        dtype=torch.float64,
    ).to(dtype=torch.float32)
    assert codec.spin_basis.dtype == torch.complex64
    assert codec.spin_basis.imag.abs().sum() > 0
    packed_mask = codec.packed_mask(("H", "O")).unsqueeze(0)
    torch.testing.assert_close(
        codec.convert_mask(packed_mask)[0], codec.coupled_mask(("H", "O"))
    )
    x = torch.randn(2, codec.coupled_dim)
    torch.testing.assert_close(codec.encode(codec(x)), x, rtol=2e-5, atol=2e-5)
    partial = packed_mask.clone()
    partial[0, 0, 0] = False
    with pytest.raises(ValueError, match="complete shell channels"):
        codec.convert_mask(partial)


@pytest.mark.parametrize("spinful", [False, True])
def test_rotation_identity_inverse_and_basis_roundtrip(spinful):
    dtype = torch.complex128 if spinful else torch.float64
    rotation = OrbitalFrameRotation(dtype, spinful=spinful)
    factor = 2 if spinful else 1
    block = torch.randn(2, 9 * factor, 4 * factor, dtype=dtype)
    r = rand_matrix(dtype=torch.float64)
    transformed = rotation.rotate_blocks(block, r, [0, 1, 2], [0, 1])
    restored = rotation.rotate_blocks(transformed, r.T, [0, 1, 2], [0, 1])
    torch.testing.assert_close(restored, block, rtol=2e-12, atol=2e-12)
    coupling = rotation.to_coupling_basis(block, [0, 1, 2], [0, 1])
    torch.testing.assert_close(
        rotation.from_coupling_basis(coupling, [0, 1, 2], [0, 1]), block
    )


@pytest.mark.parametrize("spinful", [False, True])
@pytest.mark.parametrize("sort_irreps", [False, True])
@pytest.mark.parametrize("ignore_parity", [False, True])
def test_matches_independent_source_shell_values_and_vjps(
    source, spinful, sort_irreps, ignore_parity
):
    modules, _ = source
    shell_pairs = [(0, 1), (1, 2), (2, 0), (2, 2)]
    codec = ShellTensorCodec(
        shell_pairs,
        spinful=spinful,
        sort_irreps=sort_irreps,
        ignore_parity=ignore_parity,
        dtype=torch.float64,
    )
    reference = modules.e3TensorDecomp(
        None,
        shell_pairs,
        torch.float64,
        spinful=spinful,
        if_sort=sort_irreps,
        no_parity=ignore_parity,
    )
    assert codec.irreps_out == reference.required_irreps_out
    torch.manual_seed(819)
    x = torch.randn(3, codec.coupled_dim, dtype=torch.float64, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    actual, expected = codec.decode(x), reference.get_H(ref_x)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    cotangent = torch.randn_like(actual)
    grad = torch.autograd.grad((actual.conj() * cotangent).real.sum(), x)[0]
    ref_grad = torch.autograd.grad((expected.conj() * cotangent).real.sum(), ref_x)[0]
    torch.testing.assert_close(grad, ref_grad, rtol=1e-13, atol=1e-13)
    h = actual.detach().clone().requires_grad_(True)
    ref_h = h.detach().clone().requires_grad_(True)
    encoded, ref_encoded = codec.encode(h), reference.get_net_out(ref_h)
    torch.testing.assert_close(encoded, ref_encoded, rtol=1e-13, atol=1e-13)
    grad_h = torch.autograd.grad(encoded.square().sum(), h)[0]
    ref_grad_h = torch.autograd.grad(ref_encoded.square().sum(), ref_h)[0]
    torch.testing.assert_close(grad_h, ref_grad_h, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize(
    "block_kind,source_kind", [("all", "a"), ("diag", "d"), ("offdiag", "o")]
)
def test_shared_target_order_matches_independent_source(
    source, block_kind, source_kind
):
    _, utils = source
    shells = {"1": [0, 1], "8": [0, 0, 1, 2]}
    actual = build_shell_targets(shells, block_kind=block_kind)
    expected, _, _ = utils.orbital_analysis(shells, source_kind, False)
    normalized = tuple(
        {tuple(k.split()): tuple(v) for k, v in target.items()} for target in expected
    )
    assert actual == normalized


def test_soc_shared_masks_match_independent_source(source):
    modules, _ = source
    codec = ShellTensorCodec.from_species_shells(
        {"1": [0], "8": [0, 1, 2]},
        spinful=True,
        sort_irreps=True,
        dtype=torch.float64,
    )
    reference = modules.e3TensorDecomp(
        None, codec.shell_pairs, torch.float64, spinful=True, if_sort=True
    )
    mask = codec.packed_mask(("1", "8")).unsqueeze(0).repeat(3, 1, 1)
    torch.testing.assert_close(codec.convert_mask(mask), reference.convert_mask(mask))


@pytest.mark.parametrize("spinful", [False, True])
@pytest.mark.parametrize("improper", [False, True])
def test_matches_independent_source_orbital_and_spin_rotation(
    source, spinful, improper
):
    modules, _ = source
    dtype = torch.complex128 if spinful else torch.float64
    actual = OrbitalFrameRotation(dtype, spinful=spinful)
    reference = modules.Rotate(dtype, spinful=spinful)
    torch.manual_seed(823)
    factor = 2 if spinful else 1
    h = torch.randn(3, 9 * factor, 4 * factor, dtype=dtype)
    r = rand_matrix(dtype=torch.float64)
    if improper:
        r = -r
    got = actual.rotate_blocks(h, r, [0, 1, 2], [0, 1])
    expected = reference.rotate_openmx_H_full(h, r, [0, 1, 2], [0, 1])
    torch.testing.assert_close(got, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("spinful", [False, True])
def test_per_edge_local_frames_match_independent_source_loop(source, spinful):
    modules, _ = source
    dtype = torch.complex128 if spinful else torch.float64
    rotation = OrbitalFrameRotation(dtype, spinful=spinful)
    reference = modules.Rotate(dtype, spinful=spinful)
    factor = 2 if spinful else 1
    blocks = torch.randn(3, factor * 9, factor * 4, dtype=dtype)
    rotations = rand_matrix(3, dtype=torch.float64)
    expected = torch.stack(
        [
            reference.rotate_openmx_H_full(h, r, [0, 1, 2], [0, 1])
            for h, r in zip(blocks, rotations)
        ]
    )
    actual = rotation.rotate_blocks(blocks, rotations, [0, 1, 2], [0, 1])
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
