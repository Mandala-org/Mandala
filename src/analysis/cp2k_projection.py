"""Numerical-PAO/Gaussian cross overlaps without Gaussian-fitting the PAO.

Angular integration is analytic; only the one-dimensional PAO radial integral
uses quadrature. Gaussian p/d overlaps are analytic derivatives of an s-Gaussian
overlap. This preserves complete angular shells and allows the same radial map
for every magnetic component. Lengths/exponents use bohr/bohr**-2.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
import torch
from scipy.special import ive


def read_cp2k_shells(path: str | Path, element: str, name: str):
    """Return (l, exponents, normalized primitive coefficients) per radial shell."""
    lines = [
        line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines()
    ]
    lines = [line for line in lines if line]
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.split()[0] == element and name in line.split()[1:]
        ),
        None,
    )
    if start is None:
        raise ValueError(f"{element} {name} absent from {path}")
    nset = int(lines[start + 1])
    index = start + 2
    shells = []
    for _ in range(nset):
        _, low, high, nprim, *counts = map(int, lines[index].split())
        index += 1
        if len(counts) != high - low + 1:
            raise ValueError("Malformed CP2K contraction counts")
        table = np.array(
            [list(map(float, line.split())) for line in lines[index : index + nprim]]
        )
        index += nprim
        alpha = table[:, 0]
        offset = 1
        if table.shape != (nprim, 1 + sum(counts)) or np.any(alpha <= 0):
            raise ValueError("Malformed CP2K primitive table")
        for l, count in zip(range(low, high + 1), counts):
            gram = (
                2
                * np.sqrt(alpha[:, None] * alpha[None, :])
                / (alpha[:, None] + alpha[None, :])
            ) ** (l + 1.5)
            for radial in range(count):
                coeff = table[:, offset + radial].copy()
                norm = coeff @ gram @ coeff
                if norm <= 0:
                    raise ValueError("Nonpositive Gaussian contraction norm")
                shells.append((l, alpha.copy(), coeff / np.sqrt(norm)))
            offset += count
    if any(l > 2 for l, _, _ in shells):
        raise ValueError("Cross-overlap derivatives currently support s/p/d targets")
    return shells


def pao_quadrature(radii, values_by_l, counts, *, order=4):
    """Piecewise Gauss quadrature and OpenMX-like cubic-Hermite radial values."""
    radii = np.asarray(radii, dtype=float)
    knots = np.r_[0.0, radii]
    x, w = np.polynomial.legendre.leggauss(order)
    widths = np.diff(knots)
    r = (knots[:-1, None] + (x + 1) * widths[:, None] / 2).ravel()
    weights = (widths[:, None] * w / 2).ravel()
    radials = []
    for l, count in enumerate(counts):
        if not count:
            continue
        values = np.asarray(values_by_l[l], dtype=float)[:, :count]
        slopes = np.gradient(values, radii, axis=0, edge_order=2)
        spline = CubicHermiteSpline(radii, values, slopes, axis=0)
        result = spline(np.maximum(r, radii[0]))
        low = r < radii[0]
        result[low] = (r[low, None] / radii[0]) ** l * values[0]
        radials.append((l, result))
    return r, weights, radials


def _solid_harmonics(d):
    x, y, z = d.unbind(-1)
    one = torch.ones_like(x)
    return (
        one[..., None] / math.sqrt(4 * math.pi),
        torch.stack((x, y, z), -1) * math.sqrt(3 / (4 * math.pi)),
        torch.stack(
            (
                math.sqrt(5 / (16 * math.pi)) * (2 * z * z - x * x - y * y),
                math.sqrt(15 / (16 * math.pi)) * (x * x - y * y),
                math.sqrt(15 / (4 * math.pi)) * x * y,
                math.sqrt(15 / (4 * math.pi)) * x * z,
                math.sqrt(15 / (4 * math.pi)) * y * z,
            ),
            -1,
        ),
    )


def gaussian_pao_cross_overlap(
    displacements, shells, r, weights, radials, *, batch_size=16
):
    """Integrate chi_target(r+d) phi_source(r) in native CP2K/OpenMX orders.

    The returned array has (number of displacements, target AOs, source AOs).
    Differentiation is with respect to the center displacement, not fitted data.
    """
    if any(l > 2 for l, _ in radials):
        raise ValueError("Source angular momenta above d are not implemented")
    dtype = torch.float64
    radius = torch.as_tensor(r, dtype=dtype)
    radial_weights = torch.as_tensor(
        np.asarray(weights) * np.asarray(r) ** 2, dtype=dtype
    )
    source = [(l, torch.as_tensor(values, dtype=dtype)) for l, values in radials]
    alphas = np.unique(np.concatenate([a for _, a, _ in shells]))
    alpha = torch.as_tensor(alphas, dtype=dtype)[:, None]
    ar = 2 * alpha * radius[None, :]
    primitive_norm = (2 * alpha / math.pi) ** 0.75

    def base(d):
        norm2 = (d * d).sum()
        distance = torch.sqrt(norm2.clamp_min(1e-20))
        arg = ar * distance
        safe = arg.clamp_min(1e-8)
        exponential = torch.exp(-2 * safe)
        i0 = -torch.expm1(-2 * safe) / (2 * safe)
        i1 = ((safe - 1) + (safe + 1) * exponential) / (2 * safe * safe)
        scaled = (i0, i1, i0 - 3 * i1 / safe)
        harmonics = _solid_harmonics(d)
        columns = []
        for l, values in source:
            # At the origin use an analytic series in |d|**2; this avoids
            # differentiating d/|d| and makes coincident centers regular.
            series = torch.ones_like(arg)
            term = torch.ones_like(arg)
            for j in range(1, 9):
                term = term * (ar * ar * norm2) / (2 * j * (2 * l + 2 * j + 1))
                series = series + term
            double_factorial = math.prod(range(2 * l + 1, 0, -2))
            regular = (
                ar**l
                / double_factorial
                * series
                * torch.exp(-alpha * (radius[None, :] ** 2 + norm2))
            )
            large = (
                scaled[l]
                / distance**l
                * torch.exp(-alpha * (radius[None, :] - distance) ** 2)
            )
            kernel = torch.where(arg < 0.5, regular, large)
            integrals = (kernel * radial_weights[None, :]) @ values
            integrals = integrals * (4 * math.pi * (-1) ** l) * primitive_norm
            columns.append(
                (integrals[:, :, None] * harmonics[l][None, None, :]).flatten(1)
            )
        return torch.cat(columns, dim=1)

    gradient = torch.func.jacfwd(base)
    hessian = torch.func.jacfwd(gradient)
    batches = []
    positions = torch.as_tensor(displacements, dtype=dtype)
    for start in range(0, len(positions), batch_size):
        d = positions[start : start + batch_size]
        val = torch.vmap(base)(d)
        jac = torch.vmap(gradient)(d)
        hes = torch.vmap(hessian)(d)
        target = []
        for l, exponents, coeff in shells:
            indices = torch.as_tensor(np.searchsorted(alphas, exponents))
            a = torch.as_tensor(exponents, dtype=dtype)[None, :, None]
            c = torch.as_tensor(coeff, dtype=dtype)[None, :, None, None]
            if l == 0:
                primitive = val[:, indices, :, None]
            elif l == 1:
                primitive = -jac[:, indices][:, :, :, [1, 2, 0]] / torch.sqrt(
                    a[..., None]
                )
            else:
                h = hes[:, indices]
                primitive = (
                    torch.stack(
                        (
                            h[..., 0, 1],
                            h[..., 1, 2],
                            (2 * h[..., 2, 2] - h[..., 0, 0] - h[..., 1, 1])
                            / (2 * math.sqrt(3)),
                            h[..., 0, 2],
                            (h[..., 0, 0] - h[..., 1, 1]) / 2,
                        ),
                        -1,
                    )
                    / a[..., None]
                )
            target.append((primitive * c).sum(1).transpose(-1, -2))
        batches.append(torch.cat(target, 1).detach().numpy())
    return np.concatenate(batches)


def gaussian_pao_cross_overlap_numpy(
    displacements, shells, r, weights, radials, *, batch_size=64
):
    """Same s/p/d integrals, with explicit analytic derivatives instead of AD.

    If q_l(t)=i_l(b*sqrt(t))/sqrt(t)**l, then q_l'=b*q_(l+1)/2
    and q_l''=b**2*q_(l+2)/4. These identities avoid repeated forward-mode
    differentiation when preparing larger periodic cells. Units/order match
    ``gaussian_pao_cross_overlap`` exactly.
    """
    r, weights = np.asarray(r), np.asarray(weights)
    alpha = np.unique(np.concatenate([a for _, a, _ in shells]))
    a = alpha[None, :, None]
    b = 2 * a * r[None, None, :]
    normalization = (2 * alpha / np.pi) ** 0.75
    identity = np.eye(3)
    # Quadratic matrices for the normalized native OpenMX d harmonics.
    quadratic = np.zeros((5, 3, 3))
    quadratic[0] = np.diag([-1.0, -1.0, 2.0]) * np.sqrt(5 / (16 * np.pi))
    quadratic[1] = np.diag([1.0, -1.0, 0.0]) * np.sqrt(15 / (16 * np.pi))
    for index, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 2):
        quadratic[index, i, j] = quadratic[index, j, i] = np.sqrt(15 / (4 * np.pi)) / 2
    result = []
    positions = np.asarray(displacements, dtype=float)
    for start in range(0, len(positions), batch_size):
        d = positions[start : start + batch_size]
        t = np.sum(d * d, axis=1)[:, None, None]
        distance = np.sqrt(t)
        x = b * distance
        safe = np.maximum(x, 1e-100)
        exponential = np.exp(-a * (r[None, None, :] - distance) ** 2)
        kernels = []
        for ell in range(5):
            series = np.ones_like(x)
            term = np.ones_like(x)
            for j in range(1, 10):
                term = term * (b * b * t) / (2 * j * (2 * ell + 2 * j + 1))
                series += term
            regular = (
                b**ell
                / math.prod(range(2 * ell + 1, 0, -2))
                * series
                * np.exp(-a * (r[None, None, :] ** 2 + t))
            )
            large = (
                np.sqrt(np.pi / (2 * safe))
                * ive(ell + 0.5, safe)
                / np.maximum(distance, 1e-20) ** ell
                * exponential
            )
            kernels.append(np.where(x < 0.5, regular, large))
        columns, gradients, hessians = [], [], []
        for ell, radial in radials:
            if ell > 2:
                raise ValueError("Only s/p/d source shells supported")
            q = kernels[ell]
            qp = -a * q + b * kernels[ell + 1] / 2
            qpp = a * a * q - a * b * kernels[ell + 1] + b * b * kernels[ell + 2] / 4
            radial = np.asarray(radial)
            weighted = weights * r * r
            f = np.stack([(v * weighted) @ radial for v in (q, qp, qpp)], axis=-1)
            f *= (4 * np.pi * (-1) ** ell) * normalization[None, :, None, None]
            if ell == 0:
                y = np.ones((len(d), 1)) / np.sqrt(4 * np.pi)
                dy = np.zeros((len(d), 1, 3))
                ddy = np.zeros((1, 3, 3))
            elif ell == 1:
                y = d * np.sqrt(3 / (4 * np.pi))
                dy = np.broadcast_to(
                    identity * np.sqrt(3 / (4 * np.pi)), (len(d), 3, 3)
                )
                ddy = np.zeros((3, 3, 3))
            else:
                y = np.einsum("bi,mij,bj->bm", d, quadratic, d)
                dy = 2 * np.einsum("mij,bj->bmi", quadratic, d)
                ddy = 2 * quadratic
            v = f[..., 0, None] * y[:, None, None, :]
            g = (
                f[..., 0, None, None] * dy[:, None, None, :, :]
                + 2
                * f[..., 1, None, None]
                * y[:, None, None, :, None]
                * d[:, None, None, None, :]
            )
            second = (
                d[:, None, :, None] * dy[:, :, None, :]
                + dy[:, :, :, None] * d[:, None, None, :]
                + y[:, :, None, None] * identity[None, None, :, :]
            )
            h = (
                f[..., 0, None, None, None] * ddy[None, None, None, :, :, :]
                + 2 * f[..., 1, None, None, None] * second[:, None, None, :, :, :]
                + 4
                * f[..., 2, None, None, None]
                * y[:, None, None, :, None, None]
                * d[:, None, None, None, :, None]
                * d[:, None, None, None, None, :]
            )
            columns.append(v.reshape(len(d), len(alpha), -1))
            gradients.append(g.reshape(len(d), len(alpha), -1, 3))
            hessians.append(h.reshape(len(d), len(alpha), -1, 3, 3))
        val = np.concatenate(columns, axis=2)
        jac = np.concatenate(gradients, axis=2)
        hes = np.concatenate(hessians, axis=2)
        target = []
        for ell, exponents, coefficients in shells:
            indices = np.searchsorted(alpha, exponents)
            aa = np.asarray(exponents)[None, :, None, None]
            if ell == 0:
                primitive = val[:, indices, :, None]
            elif ell == 1:
                primitive = -jac[:, indices][..., [1, 2, 0]] / np.sqrt(aa)
            elif ell == 2:
                hh = hes[:, indices]
                primitive = (
                    np.stack(
                        (
                            hh[..., 0, 1],
                            hh[..., 1, 2],
                            (2 * hh[..., 2, 2] - hh[..., 0, 0] - hh[..., 1, 1])
                            / (2 * np.sqrt(3)),
                            hh[..., 0, 2],
                            (hh[..., 0, 0] - hh[..., 1, 1]) / 2,
                        ),
                        axis=-1,
                    )
                    / aa
                )
            else:
                raise ValueError("Only s/p/d target shells supported")
            target.append(
                np.sum(
                    primitive * np.asarray(coefficients)[None, :, None, None], axis=1
                ).transpose(0, 2, 1)
            )
        result.append(np.concatenate(target, axis=1))
    return np.concatenate(result)


def infer_image_layout(
    stored_overlaps,
    physical_overlaps,
    cells,
    *,
    block_size,
    tolerance=2e-8,
    printed_diagonal_triangle=False,
):
    """Certify each native atom block against the +L or -L analytic overlap.

    CP2K's symmetric neighbor-list storage must not be treated as an ordinary
    full physical matrix for every image. A native overlap is the layout oracle;
    it contains no converged-density or Hamiltonian information.
    """
    nimage, nao, _ = stored_overlaps.shape
    natom = nao // block_size
    layout = np.zeros((nimage, natom, natom), dtype=np.int8)
    max_error = 0.0
    for image, cell in enumerate(cells):
        positive = physical_overlaps[tuple(cell)]
        negative = physical_overlaps[tuple(-np.asarray(cell))]
        for i in range(natom):
            for j in range(natom):
                rows = slice(i * block_size, (i + 1) * block_size)
                cols = slice(j * block_size, (j + 1) * block_size)
                block = stored_overlaps[image, rows, cols]
                if np.max(abs(block)) < 1e-14:
                    continue
                plus_difference = block - positive[rows, cols]
                minus_difference = block - negative[rows, cols]
                # CP2K's AO text printer mirrors the upper AO triangle even
                # inside a nonzero-image diagonal atom block. Such a block
                # need not itself be symmetric. Binary payloads remain full.
                if printed_diagonal_triangle and i == j:
                    plus_difference = np.triu(plus_difference)
                    minus_difference = np.triu(minus_difference)
                plus = np.max(abs(plus_difference))
                minus = np.max(abs(minus_difference))
                error = min(plus, minus)
                if error > tolerance:
                    raise ValueError(
                        f"Uncertified CP2K image block {(image, i, j)}: {plus}, {minus}"
                    )
                layout[image, i, j] = 1 if plus <= minus else -1
                max_error = max(max_error, float(error))
    return layout, max_error


def unpack_images(stored, cells, layout, *, block_size):
    """Restore full physical directed images, including Hermitian reverses."""
    nao = stored.shape[1]
    matrices = {}
    assigned = {}
    for image, cell in enumerate(cells):
        for i in range(layout.shape[1]):
            for j in range(layout.shape[2]):
                sign = int(layout[image, i, j])
                if sign == 0:
                    continue
                shift = tuple(sign * np.asarray(cell))
                reverse = tuple(-np.asarray(shift))
                rows = slice(i * block_size, (i + 1) * block_size)
                cols = slice(j * block_size, (j + 1) * block_size)
                block = stored[image, rows, cols]
                for key, a, b, value in (
                    (shift, i, j, block),
                    (reverse, j, i, block.T),
                ):
                    matrices.setdefault(key, np.zeros((nao, nao)))
                    mask = assigned.setdefault(
                        key, np.zeros(layout.shape[1:], dtype=bool)
                    )
                    r = slice(a * block_size, (a + 1) * block_size)
                    c = slice(b * block_size, (b + 1) * block_size)
                    if mask[a, b] and np.max(abs(matrices[key][r, c] - value)) > 1e-9:
                        raise ValueError("Inconsistent native reverse-image values")
                    matrices[key][r, c] = value
                    mask[a, b] = True
    return matrices


def pack_images(physical, cells, layout, *, block_size):
    """Pack physical directed density blocks in a certified native layout."""
    nao = next(iter(physical.values())).shape[0]
    stored = np.zeros((len(cells), nao, nao))
    for image, cell in enumerate(cells):
        for i in range(layout.shape[1]):
            for j in range(layout.shape[2]):
                sign = int(layout[image, i, j])
                if sign == 0:
                    continue
                key = tuple(sign * np.asarray(cell))
                r = slice(i * block_size, (i + 1) * block_size)
                c = slice(j * block_size, (j + 1) * block_size)
                stored[image, r, c] = physical[key][r, c]
    return stored


def metric_project_density(
    density, overlap, cross_overlap, nelectrons, *, threshold=1e-5
):
    """Least-squares AO projection followed by a covariant occupation repair.

    Equal k weights; restricted spin-summed occupations in [0,2]. The repair is
    the nearest metric density under a shared scalar water-filling shift. Report
    the change: it is not an exact transformation between different AO spaces.
    """
    overlap = 0.5 * (overlap + overlap.swapaxes(-1, -2).conj())
    values, vectors = np.linalg.eigh(overlap)
    if np.min(values) <= threshold:
        raise ValueError(
            "Target overlap has discarded directions; rank-aware projection required"
        )
    root = (vectors * np.sqrt(values)[:, None, :]) @ vectors.swapaxes(-1, -2).conj()
    inverse_root = (vectors / np.sqrt(values)[:, None, :]) @ vectors.swapaxes(
        -1, -2
    ).conj()
    transform = np.linalg.solve(overlap, cross_overlap)
    projected = transform @ density @ transform.swapaxes(-1, -2).conj()
    metric = root @ projected @ root
    metric = 0.5 * (metric + metric.swapaxes(-1, -2).conj())
    occupations, orbitals = np.linalg.eigh(metric)
    lo, hi = -float(np.max(occupations)) - 2, 2 - float(np.min(occupations)) + 2
    for _ in range(90):
        shift = 0.5 * (lo + hi)
        count = np.clip(occupations + shift, 0, 2).sum(axis=1).mean()
        if count < nelectrons:
            lo = shift
        else:
            hi = shift
    repaired = np.clip(occupations + 0.5 * (lo + hi), 0, 2)
    metric_fixed = (orbitals * repaired[:, None, :]) @ orbitals.swapaxes(-1, -2).conj()
    result = inverse_root @ metric_fixed @ inverse_root
    diagnostics = {
        "target_overlap_min_eigenvalue": float(np.min(values)),
        "raw_projected_electrons": float(occupations.sum(axis=1).mean()),
        "raw_occupation_min": float(np.min(occupations)),
        "raw_occupation_max": float(np.max(occupations)),
        "occupation_shift": 0.5 * (lo + hi),
        "repaired_electrons": float(repaired.sum(axis=1).mean()),
        "metric_repair_relative_norm": float(
            np.linalg.norm(metric_fixed - metric) / np.linalg.norm(metric)
        ),
    }
    return result, diagnostics
