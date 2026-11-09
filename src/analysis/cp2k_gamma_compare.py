"""Gamma occupied-space transfer helpers for matched CP2K comparisons.

Lengths are bohr internally. Complete native CP2K angular shells are used for
target rows; source columns use native OpenMX order. No ML weights are changed.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import itertools
import math
from multiprocessing import get_context
import re

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from analysis.cp2k_projection import gaussian_pao_cross_overlap_numpy

BOHR_PER_ANGSTROM = 1.8897261254578281
CP2K_FROM_OPENMX_SPD = np.array([0, 1, 3, 4, 2, 6, 7, 5, 10, 12, 8, 11, 9])


def input_geometry(path):
    text = path.read_text()
    if not re.search(r"Atoms.SpeciesAndCoordinates.Unit\s+Ang", text, re.I):
        raise ValueError("An explicit Angstrom source geometry is required")
    if not re.search(r"Atoms.UnitVectors.Unit\s+Ang", text, re.I):
        raise ValueError("An explicit Angstrom source cell is required")

    def body(name):
        return re.search(r"<" + name + r"[^\n]*\n(.*?)" + name + r">", text, re.S)[1]

    rows = [
        s.split() for s in body("Atoms.SpeciesAndCoordinates").splitlines() if s.strip()
    ]
    atoms = tuple(s[1] for s in rows)
    if any(a not in {"Si", "O"} for a in atoms):
        raise ValueError("This comparison is restricted to Si/O s2p2d1 systems")
    positions = np.array([[float(x) for x in s[2:5]] for s in rows])
    cell = np.array(
        [
            list(map(float, s.split()))
            for s in body("Atoms.UnitVectors").splitlines()
            if s.strip()
        ]
    )
    if cell.shape != (3, 3) or np.linalg.det(cell) <= 0:
        raise ValueError("Invalid source cell")
    # The selected inputs are orthorhombic, so CP2K 2023.1 canonicalization
    # cannot silently rotate the experimental coordinate system.
    if np.max(abs(cell - np.diag(np.diag(cell)))) > 1e-12:
        raise ValueError(
            "General cell canonicalization needs a separately checked input"
        )
    return atoms, positions, cell


def gamma_from_dump(section, block_sizes):
    offsets = np.cumsum([0, *block_sizes])
    result = np.zeros((offsets[-1], offsets[-1]))
    for (i, j, _), block in section.blocks.items():
        result[offsets[i - 1] : offsets[i], offsets[j - 1] : offsets[j]] += block
    return result


def gamma_from_blocks(matrix):
    if matrix.basis != "openmx":
        raise ValueError("Decode to native OpenMX before summing Gamma blocks")
    result = np.zeros((13 * len(matrix.atoms), 13 * len(matrix.atoms)))
    for key, edges in matrix.pair_edges.items():
        for index, (_, _, _, i, j) in enumerate(edges.T.tolist()):
            result[13 * i : 13 * (i + 1), 13 * j : 13 * (j + 1)] += (
                matrix.pair_blocks[key][index].double().numpy()
            )
    return result


def occupied_from_hamiltonian(h, s, electrons):
    """Zero-temperature restricted filling: same recipe for both transfer arms."""
    if electrons % 2 or electrons // 2 >= len(s):
        raise ValueError("Need an even closed-shell electron count below capacity")
    h, s = (h + h.T) / 2, (s + s.T) / 2
    nocc = electrons // 2
    values, vectors = eigh(
        h, s, subset_by_index=[0, nocc], driver="gvx", check_finite=True
    )
    c = vectors[:, :nocc]
    error = float(np.max(abs(c.T @ s @ c - np.eye(nocc))))
    if error > 2e-7:
        raise ValueError(f"Source occupied metric orthogonality failure {error}")
    return c, dict(
        source_electrons=float(2 * np.trace(c.T @ s @ c)),
        occupied_metric_error=error,
        zero_temperature=True,
        source_homo_lumo_gap_hartree=float(values[nocc] - values[nocc - 1]),
    )


def occupied_from_density(p, s, electrons):
    """Explicit source-metric rank-N projector control for the Si density head."""
    v, u = eigh((s + s.T) / 2)
    if v.min() <= 1e-8:
        raise ValueError("Invalid source overlap in density-head projector control")
    root = (u * np.sqrt(v)) @ u.T
    inverse = (u / np.sqrt(v)) @ u.T
    x = root @ ((p + p.T) / 2) @ root
    f, w = eigh((x + x.T) / 2)
    n = electrons // 2
    if f[-n] <= 1e-8 or f[-n] - f[-n - 1] <= 1e-8:
        raise ValueError(
            "Density-head occupied subspace is rank deficient or ambiguous"
        )
    c = inverse @ w[:, -n:]
    fixed = 2 * (w[:, -n:] @ w[:, -n:].T)
    return c, dict(
        raw_source_electrons=float(np.trace(x)),
        raw_occupation_min=float(f.min()),
        raw_occupation_max=float(f.max()),
        source_metric_projector_relative_change=float(
            np.linalg.norm(fixed - x) / np.linalg.norm(x)
        ),
        occupation_policy="Explicit rank-N zero-temperature projector; not raw learned density",
    )


def project_occupied(c_source, s_target, cross, *, threshold=1e-5):
    """Project orbitals and orthonormalize the occupied subspace in target S."""
    v, u = eigh((s_target + s_target.T) / 2)
    retained = v > threshold
    if retained.sum() < c_source.shape[1]:
        raise ValueError("Too few retained target directions for occupied space")
    inverse_sqrt = u[:, retained] / np.sqrt(v[retained])
    a = inverse_sqrt.T @ cross @ c_source
    g = a.T @ a
    gv, gu = eigh((g + g.T) / 2)
    if gv.min() <= 1e-8:
        raise ValueError("Occupied rank lost in cross-overlap projection")
    c = inverse_sqrt @ a @ ((gu / np.sqrt(gv)) @ gu.T)
    error = float(np.max(abs(c.T @ s_target @ c - np.eye(c.shape[1]))))
    if error > 3e-7:
        raise ValueError(f"Target occupied metric orthogonality failure {error}")
    return c, dict(
        target_overlap_min=float(v.min()),
        target_rank_retained=int(retained.sum()),
        target_directions_discarded=int((~retained).sum()),
        raw_projected_electrons=float(2 * np.trace(g)),
        minimum_occupied_gram_eigenvalue=float(gv.min()),
        occupied_metric_error=error,
        final_electrons=float(2 * np.trace(c.T @ s_target @ c)),
        repair="Target-metric occupied-subspace orthonormalization; integer occupations retained",
    )


def gaussian_gamma_overlap(atoms, positions_bohr, cell_bohr, shells_by_species):
    """Independent periodic Gaussian overlap with PySCF, reordered for CP2K."""
    from pyscf.pbc import gto

    basis = {}
    permutation = []
    for atom in set(atoms):
        basis[atom] = [
            [ell, *[[float(a), float(c)] for a, c in zip(alpha, coeff)]]
            for ell, alpha, coeff in shells_by_species[atom]
        ]
    for i in range(len(atoms)):
        # PySCF p=(x,y,z); d=(xy,yz,z2,xz,x2-y2), already CP2K's d order.
        permutation.extend(
            (13 * i + np.array([0, 1, 3, 4, 2, 6, 7, 5, 8, 9, 10, 11, 12])).tolist()
        )
    cell = gto.Cell()
    cell.atom = list(zip(atoms, np.asarray(positions_bohr)))
    cell.a = np.asarray(cell_bohr)
    cell.unit = "Bohr"
    cell.basis = basis
    cell.precision = 1e-13
    cell.verbose = 0
    cell.build()
    s = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1))
    if np.iscomplexobj(s) and np.max(abs(s.imag)) > 1e-10:
        raise ValueError("Unexpected complex Gamma overlap")
    return s.real[np.ix_(permutation, permutation)]


def compact_pao_quadrature(table, counts=(2, 2, 1), order=512):
    radius = table.radii.numpy()
    x, w = np.polynomial.legendre.leggauss(order)
    r, weights = (x + 1) * radius[-1] / 2, w * radius[-1] / 2
    values = []
    for ell, count in enumerate(counts):
        raw = table.values_by_l[ell].numpy()[:, :count]
        interpolant = CubicHermiteSpline(
            radius, raw, np.gradient(raw, radius, axis=0, edge_order=2), axis=0
        )
        y = interpolant(np.maximum(r, radius[0]))
        mask = r < radius[0]
        y[mask] = (r[mask, None] / radius[0]) ** ell * raw[0]
        values.append((ell, y))
    return r, weights, values


def cross_tail_cutoff(shells, source_radius, tolerance=1e-12):
    """Conservative Cauchy/supremum bound outside a finite PAO sphere.

    Sum absolute normalized Gaussian primitive amplitudes; cancellations are
    never relied on to exclude an image. Source radial norms are close to one,
    with an additional factor two margin. All selected source shells have 13 AOs.
    """
    volume = 4 * np.pi * source_radius**3 / 3

    def bound(distance):
        minimum = distance - source_radius
        total = 0.0
        for ell, alpha, coeff in shells:
            alpha, coeff = np.asarray(alpha), np.asarray(coeff)
            if minimum <= np.max(np.sqrt(ell / (2 * alpha))):
                return np.inf
            norm = np.sqrt(2 * (2 * alpha) ** (ell + 1.5) / math.gamma(ell + 1.5))
            radial = np.sum(
                abs(coeff) * norm * minimum**ell * np.exp(-alpha * minimum**2)
            )
            total += (2 * ell + 1) ** 2 / (4 * np.pi) * radial**2
        return 2 * np.sqrt(13 * volume * total)

    cutoff = source_radius + 1.0
    while bound(cutoff) > tolerance:
        cutoff += 0.5
        if cutoff > 180:
            raise ValueError("Gaussian tail exceeds the registered integration budget")
    return cutoff, float(bound(cutoff))


def _cross_chunk(arguments):
    d, shells, quadrature = arguments
    with threadpool_limits(limits=1):
        return gaussian_pao_cross_overlap_numpy(d, shells, *quadrature)


def periodic_cross_overlap(
    atoms,
    positions_bohr,
    cell_bohr,
    shells_by_species,
    quadratures,
    source_radii,
    *,
    workers=1,
    progress=None,
):
    """Sum finite PAO/Gaussian image overlaps at Gamma, with bounded tails."""
    n = len(atoms)
    result = np.zeros((13 * n, 13 * n))
    records = []
    inverse = np.linalg.inv(cell_bohr)
    for target in sorted(set(atoms)):
        ti = np.flatnonzero(np.array(atoms) == target)
        for source in sorted(set(atoms)):
            sj = np.flatnonzero(np.array(atoms) == source)
            shells = shells_by_species[target]
            q = quadratures[source]
            cutoff, bound = cross_tail_cutoff(shells, source_radii[source])
            limits = np.ceil(cutoff * np.linalg.norm(inverse, axis=0) + 1).astype(int)
            displacements, rows, cols = [], [], []
            for cell in itertools.product(*(range(-v, v + 1) for v in limits)):
                d = (
                    positions_bohr[sj][None, :, :]
                    + np.asarray(cell) @ cell_bohr
                    - positions_bohr[ti][:, None, :]
                )
                mask = np.sum(d * d, axis=-1) <= cutoff**2
                ii, jj = np.nonzero(mask)
                if len(ii):
                    displacements.append(d[ii, jj])
                    rows.append(ti[ii])
                    cols.append(sj[jj])
            d = np.concatenate(displacements)
            ii, jj = np.concatenate(rows), np.concatenate(cols)
            if progress:
                progress(
                    f"Cross {target}/{source}: {len(d)} center images, {len(q[0])} radial nodes, cutoff {cutoff:.1f} bohr"
                )
            chunks = [
                (d[start : start + 512], shells, q) for start in range(0, len(d), 512)
            ]
            if workers > 1:
                pool = ProcessPoolExecutor(
                    max_workers=workers, mp_context=get_context("spawn")
                )
                values = pool.map(_cross_chunk, chunks)
            else:
                pool = None
                values = map(_cross_chunk, chunks)
            try:
                tensor = result.reshape(n, 13, n, 13).transpose(0, 2, 1, 3)
                start = 0
                for value in values:
                    end = start + len(value)
                    np.add.at(tensor, (ii[start:end], jj[start:end]), value)
                    start = end
            finally:
                if pool is not None:
                    pool.shutdown()
            records.append(
                dict(
                    target=target,
                    source=source,
                    center_images=len(d),
                    cutoff_bohr=cutoff,
                    per_image_frobenius_tail_bound=bound,
                )
            )
    return result, records


def printed_gamma_overlap(path, nao):
    """Read one CP2K Gamma overlap table, accepting variable print columns."""
    output = np.full((nao, nao), np.nan)
    columns = None
    active = False
    seen = np.zeros((nao, nao), dtype=bool)
    filled = 0
    with path.open() as stream:
        for line in stream:
            if line.strip() == "OVERLAP MATRIX":
                if active:
                    raise ValueError("Expected one Gamma overlap, found multiple")
                active = True
                continue
            if not active:
                continue
            fields = line.split()
            if fields and all(v.isdigit() for v in fields):
                columns = np.array(fields, dtype=int) - 1
            elif (
                columns is not None
                and len(fields) == 4 + len(columns)
                and fields[0].isdigit()
                and fields[1].isdigit()
            ):
                row = int(fields[0]) - 1
                if 0 <= row < nao and np.all((columns >= 0) & (columns < nao)):
                    output[row, columns] = [float(x) for x in fields[4:]]
                    filled += int(np.count_nonzero(~seen[row, columns]))
                    seen[row, columns] = True
                    if filled == nao * nao:
                        return output
    raise ValueError("Incomplete native overlap table")
