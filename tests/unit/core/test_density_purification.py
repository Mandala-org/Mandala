from collections import Counter

import pytest
import torch
from e3nn.o3 import rand_matrix

from core.density_purification import McWeenyPurifier, purify_density
from core.orbital_irrep_config import OrbitalIrrepConfig
from core.sparse_math import build_matmul_alignment, matmul_sparse_block_matrix_aligned
from core.sparse_math import (
    build_trace_alignment,
    trace_matmul_sparse_block_matrix_aligned,
)
from data.block_matrix import BlockMatrix


def make_matrix(atoms=("H", "O"), shifts=(-1, 0, 1), seed=12):
    cfg = OrbitalIrrepConfig.from_dict({"H": "1s", "O": "1s1p"})
    generator = torch.Generator().manual_seed(seed)
    blocks, edges, lookup = {}, {}, {}
    for i, a in enumerate(atoms):
        for j, b in enumerate(atoms):
            key = f"{a}-{b}"
            for shift in shifts:
                edge = (shift, 0, 0, i, j)
                lookup[edge] = (key, len(blocks.get(key, [])))
                blocks.setdefault(key, []).append(
                    torch.randn(
                        *cfg.block_dims(key), generator=generator, dtype=torch.float64
                    )
                )
                edges.setdefault(key, []).append(edge)
    return BlockMatrix(
        atoms,
        Counter(atoms),
        {k: torch.stack(v) for k, v in blocks.items()},
        {k: torch.tensor(v).T for k, v in edges.items()},
        lookup,
        cfg,
        "e3nn",
    )


def slow_product(a, b, output):
    result = {k: torch.zeros_like(v) for k, v in output.pair_blocks.items()}
    for (sx, sy, sz, i, j), (key, index) in output.lookup.items():
        for (tx, ty, tz, source, k), (ka, ia) in a.lookup.items():
            match = b.lookup.get((sx - tx, sy - ty, sz - tz, k, j))
            if source == i and match is not None:
                kb, ib = match
                result[key][index] += a.pair_blocks[ka][ia] @ b.pair_blocks[kb][ib]
    return output._replace_pair_blocks(result, basis=output.basis)


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_periodic_product_matches_explicit_shift_convolution(chunk_size):
    a, b = make_matrix(), make_matrix(seed=14)
    output = make_matrix(shifts=(0,))
    plan = build_matmul_alignment(a, b, output)
    result = matmul_sparse_block_matrix_aligned(
        a, b, output, plan, chunk_size=chunk_size
    )
    expected = slow_product(a, b, output)
    assert plan.num_paths > 0
    for key in result.pair_blocks:
        torch.testing.assert_close(result.pair_blocks[key], expected.pair_blocks[key])
        assert torch.equal(result.pair_edges[key], output.pair_edges[key])
    assert result.lookup == output.lookup


def test_product_matches_dense_and_autograd():
    a, b = make_matrix(shifts=(0,)), make_matrix(shifts=(0,), seed=19)
    for matrix in (a, b):
        for value in matrix.pair_blocks.values():
            value.requires_grad_()
    result = matmul_sparse_block_matrix_aligned(
        a, b, a, build_matmul_alignment(a, b, a)
    )
    expected = a.to_dense() @ b.to_dense()
    torch.testing.assert_close(result.to_dense(), expected)
    parameters = list(a.pair_blocks.values()) + list(b.pair_blocks.values())
    actual_grad = torch.autograd.grad(
        result.to_dense().square().sum(), parameters, retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected)


def test_periodic_product_onsite_trace_matches_existing_sparse_trace():
    a, b = make_matrix(), make_matrix(seed=19)
    result = matmul_sparse_block_matrix_aligned(
        a, b, a, build_matmul_alignment(a, b, a)
    )
    actual = sum(
        torch.trace(result.pair_blocks[key][index])
        for (sx, sy, sz, i, j), (key, index) in result.lookup.items()
        if (sx, sy, sz) == (0, 0, 0) and i == j
    )
    expected = trace_matmul_sparse_block_matrix_aligned(
        a, b, build_trace_alignment(a, b)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("iterations", [0, 1, 2, 3])
def test_mcweeny_matches_nonorthogonal_dense_polynomial(iterations):
    template = make_matrix(atoms=("O",), shifts=(0,))
    s = torch.diag(torch.tensor([1.2, 0.8, 1.7, 1.1], dtype=torch.float64))
    d = torch.diag(torch.tensor([0.8, 0.2, 0.9, 0.1], dtype=torch.float64)) / s.diag()
    density = template._replace_pair_blocks({"O-O": d.unsqueeze(0)}, basis="e3nn")
    overlap = template._replace_pair_blocks({"O-O": s.unsqueeze(0)}, basis="e3nn")
    result = purify_density(density, overlap, iterations=iterations)
    expected = d
    for _ in range(iterations):
        expected = (
            3 * expected @ s @ expected - 2 * expected @ s @ expected @ s @ expected
        )
    torch.testing.assert_close(result.to_dense(), expected)
    projector = (
        torch.diag(torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)) / s.diag()
    )
    assert (expected - projector).norm() <= (d - projector).norm()


def test_truncated_periodic_mcweeny_matches_explicit_products_and_preserves_edges():
    d, s = make_matrix(), make_matrix(seed=3)
    d = d * 0.02
    actual = McWeenyPurifier(d, s)(d, iterations=3)
    expected = d
    for _ in range(3):
        sd = slow_product(s, expected, d)
        dsd = slow_product(expected, sd, d)
        cubic = slow_product(dsd, sd, d)
        expected = dsd * 3 - cubic * 2
    for key in actual.pair_blocks:
        torch.testing.assert_close(actual.pair_blocks[key], expected.pair_blocks[key])
        assert torch.equal(actual.pair_edges[key], d.pair_edges[key])


def test_purification_o3_equivariance_and_finite_gradient():
    d, s = make_matrix(), make_matrix(seed=3)
    d = d * 0.02
    for block in d.pair_blocks.values():
        block.requires_grad_()
    purified = purify_density(d, s)

    def rotate(matrix, q):
        u = {
            el: ir.D_from_matrix(q)
            for el, ir in matrix.orbital_cfg.element_to_irreps.items()
        }
        return matrix._replace_pair_blocks(
            {
                key: u[key.split("-")[0]] @ block @ u[key.split("-")[1]].T
                for key, block in matrix.pair_blocks.items()
            },
            basis=matrix.basis,
        )

    rotation = rand_matrix(dtype=torch.float64)
    for q in (
        rotation,
        -torch.eye(3, dtype=torch.float64),
        torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)),
        -rotation,
    ):
        actual = purify_density(rotate(d, q), rotate(s, q))
        expected = rotate(purified, q)
        for key in actual.pair_blocks:
            torch.testing.assert_close(
                actual.pair_blocks[key], expected.pair_blocks[key], atol=1e-7, rtol=1e-5
            )
    sum(v.square().sum() for v in purified.pair_blocks.values()).backward()
    assert all(torch.isfinite(v.grad).all() for v in d.pair_blocks.values())


def test_no_paths_produces_zero_with_zero_gradients():
    a = make_matrix(shifts=(1,))
    output = make_matrix(shifts=(0,))
    for value in a.pair_blocks.values():
        value.requires_grad_()
    result = matmul_sparse_block_matrix_aligned(
        a, a, output, build_matmul_alignment(a, a, output)
    )
    sum(v.sum() for v in result.pair_blocks.values()).backward()
    assert all(torch.count_nonzero(v) == 0 for v in result.pair_blocks.values())
    assert all(torch.count_nonzero(v.grad) == 0 for v in a.pair_blocks.values())


def test_rejects_stale_topology_and_invalid_iteration_count():
    d, s = make_matrix(), make_matrix(seed=3)
    purifier = McWeenyPurifier(d, s)
    changed = make_matrix(shifts=(0,))
    with pytest.raises(ValueError, match="topology"):
        purifier(changed)
    for count in (-1, 0.5, True):
        with pytest.raises(ValueError, match="iterations"):
            purifier(d, iterations=count)


def test_noncommuting_dense_mcweeny_and_overlap_gradient():
    template = make_matrix(atoms=("O",), shifts=(0,))
    generator = torch.Generator().manual_seed(21)
    a = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    s = (a @ a.T + torch.eye(4, dtype=torch.float64)).requires_grad_()
    d = (
        torch.randn(4, 4, generator=generator, dtype=torch.float64) * 0.03
    ).requires_grad_()
    density = template._replace_pair_blocks({"O-O": d[None]}, basis="e3nn")
    overlap = template._replace_pair_blocks({"O-O": s[None]}, basis="e3nn")
    actual = purify_density(density, overlap, iterations=3, chunk_size=1).to_dense()
    expected = d
    for _ in range(3):
        expected = (
            3 * expected @ s @ expected - 2 * expected @ s @ expected @ s @ expected
        )
    torch.testing.assert_close(actual, expected)
    got_grad = torch.autograd.grad(actual.square().sum(), (d, s), retain_graph=True)
    ref_grad = torch.autograd.grad(expected.square().sum(), (d, s))
    for got, ref in zip(got_grad, ref_grad):
        torch.testing.assert_close(got, ref)
