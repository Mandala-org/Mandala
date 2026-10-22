"""Periodic bond frames and bond-rooted neighborhood relations."""

from __future__ import annotations

import math
import torch


def bond_frames(edge_index, vectors, *, tolerance=1e-6, neighbor_selection=None):
    """First bond and shortest independent neighbor define an oriented frame.

    Stored edge order resolves distance ties exactly through torch.sort. A
    collinear environment is rejected rather than assigning an arbitrary gauge.
    """
    frames = []
    for edge, source in enumerate(edge_index[0].tolist()):
        neighbors = torch.where(edge_index[0] == source)[0]
        if neighbor_selection is not None:
            neighbors = neighbors[torch.isin(neighbors, neighbor_selection)]
        neighbors = neighbors[torch.sort(vectors[neighbors].norm(dim=-1)).indices]
        nonzero = neighbors[vectors[neighbors].norm(dim=-1) > 1e-8]
        if not len(nonzero):
            raise ValueError(f"Cannot construct bond frame for isolated atom {source}")
        first = vectors[edge]
        if torch.allclose(first.detach(), torch.zeros_like(first)):
            first = vectors[nonzero[0]]
        second = None
        for neighbor in nonzero:
            candidate = vectors[neighbor]
            normal = torch.linalg.cross(first, candidate)
            if normal.norm() > tolerance:
                second = normal / normal.norm()
                break
        if second is None:
            raise ValueError(f"No linearly independent chemical bond for atom {source}")
        axis = first / first.norm()
        frames.append(torch.stack([axis, second, torch.linalg.cross(axis, second)], -1))
    return torch.stack(frames) if frames else vectors.new_empty((0, 3, 3))


def cartesian_tesseral_harmonics(vectors, num_l):
    """Same harmonics without angular-coordinate singularities at bond poles."""
    radius = vectors.square().sum(-1).clamp_min(1e-24).sqrt()
    x = vectors[:, 0] / radius
    azimuth = torch.complex(vectors[:, 1], vectors[:, 2]) / radius
    columns = {}
    diagonal = torch.ones_like(azimuth)
    for order in range(num_l):
        if order:
            diagonal = diagonal * -(2 * order - 1) * azimuth
        old, value = diagonal, diagonal
        columns[(order, order)] = diagonal
        if order + 1 < num_l:
            value = x * (2 * order + 1) * diagonal
            columns[(order + 1, order)] = value
        for degree in range(order + 2, num_l):
            old, value = value, (
                (2 * degree - 1) * x * value - (degree + order - 1) * old
            ) / (degree - order)
            columns[(degree, order)] = value
    result = []
    for degree in range(num_l):
        for order in range(-degree, degree + 1):
            m = abs(order)
            norm = math.sqrt((2 * degree + 1) / (4 * math.pi))
            if m:
                norm *= math.sqrt(
                    2 * math.factorial(degree - m) / math.factorial(degree + m)
                )
            value = columns[(degree, m)]
            result.append(norm * (value.imag if order < 0 else value.real))
    return torch.stack(result, -1)


def tesseral_harmonics(vectors, num_l):
    """Integral-normalized real harmonics, Condon–Shortley phase, polar x axis."""
    if vectors.requires_grad:
        return cartesian_tesseral_harmonics(vectors, num_l)
    theta = torch.atan2(
        torch.sqrt(vectors[:, 1].square() + vectors[:, 2].square()), vectors[:, 0]
    )
    phi = torch.atan2(vectors[:, 2], vectors[:, 1])
    x = theta.cos()
    result = []
    for degree in range(num_l):
        for order in range(-degree, degree + 1):
            m = abs(order)
            diagonal = torch.ones_like(x)
            for k in range(1, m + 1):
                diagonal = (
                    diagonal * -(2 * k - 1) * (1 - x.square()).clamp_min(0).sqrt()
                )
            old = diagonal
            value = diagonal if degree == m else x * (2 * m + 1) * diagonal
            for ell in range(m + 2, degree + 1):
                old, value = (
                    value,
                    ((2 * ell - 1) * x * value - (ell + m - 1) * old) / (ell - m),
                )
            norm = math.sqrt((2 * degree + 1) / (4 * math.pi))
            if order:
                norm *= math.sqrt(
                    2 * math.factorial(degree - m) / math.factorial(degree + m)
                )
                value = value * ((m * phi).cos() if order > 0 else (m * phi).sin())
            result.append(norm * value)
    return torch.stack(result, -1)


def bond_neighborhood_relations(edge_index, *, neighbor_selection=None):
    """Prepare bond-rooted index relations; no geometry or harmonics involved."""
    atoms, edges, segments = [], [], []
    for output_edge, endpoints in enumerate(edge_index.T):
        for endpoint, center in enumerate(endpoints):
            selected = torch.where(edge_index[0] == center)[0]
            if neighbor_selection is not None:
                selected = selected[torch.isin(selected, neighbor_selection)]
            atoms.append(
                torch.stack([center.expand(len(selected)), edge_index[1, selected]], -1)
            )
            edges.append(selected)
            segments.append(torch.full_like(selected, 2 * output_edge + endpoint))
    return {
        "sub_atom_idx": torch.cat(atoms) if atoms else edge_index.new_empty((0, 2)),
        "sub_edge_idx": torch.cat(edges) if edges else edge_index.new_empty(0),
        "sub_index": torch.cat(segments) if segments else edge_index.new_empty(0),
    }


def bond_neighborhoods(edge_index, vectors, frames, *, num_l=5):
    """Explicit directed relations for both endpoints of each predicted bond."""
    atoms, edges, angles, segments = [], [], [], []
    for output_edge, endpoints in enumerate(edge_index.T):
        for endpoint, center in enumerate(endpoints):
            selected = torch.where(edge_index[0] == center)[0]
            atoms.append(
                torch.stack([center.expand(len(selected)), edge_index[1, selected]], -1)
            )
            edges.append(selected)
            # Original local-coordinate convention uses ri - (rj + R).
            local = -vectors[selected] @ frames[output_edge]
            angles.append(tesseral_harmonics(local, num_l))
            segments.append(torch.full_like(selected, 2 * output_edge + endpoint))
    if not atoms:
        return {
            "sub_atom_idx": edge_index.new_empty((0, 2)),
            "sub_edge_idx": edge_index.new_empty(0),
            "sub_edge_ang": vectors.new_empty((0, num_l**2)),
            "sub_index": edge_index.new_empty(0),
        }
    return dict(
        zip(
            ("sub_atom_idx", "sub_edge_idx", "sub_edge_ang", "sub_index"),
            (
                torch.cat(atoms),
                torch.cat(edges),
                torch.cat(angles),
                torch.cat(segments),
            ),
        )
    )
