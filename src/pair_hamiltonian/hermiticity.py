"""Post-training Hermitian projection for directed Hamiltonian predictions."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from pair_hamiltonian.output_schema import FullBlockIrrepTransform


def project_onsite_irreps(
    transform: FullBlockIrrepTransform,
    species: str,
    raw_prediction: torch.Tensor,
) -> torch.Tensor:
    """Project raw ``(i,i,0)`` predictions; never call this during training."""
    reversed_prediction = transform.reverse((species, species), raw_prediction).to(
        raw_prediction
    )
    return 0.5 * (raw_prediction + reversed_prediction)


def project_directed_irreps(
    transform: FullBlockIrrepTransform,
    pair_names: Sequence[str],
    raw_prediction: torch.Tensor,
    pair_types: torch.Tensor,
    inverse: torch.Tensor,
) -> torch.Tensor:
    """Project complete directed reverse pairs onto global Hermiticity.

    For edge ``e=(i,j,L)`` with inverse ``r=(j,i,-L)``, this computes
    ``0.5 * (raw[e] + transpose(raw[r]))`` in the irrep basis of ``e``.
    Both raw predictions must already exist; no direction is generated here.
    """
    if raw_prediction.ndim != 2:
        raise ValueError("raw_prediction must have shape (directed_edges, components)")
    edge_count = raw_prediction.shape[0]
    if pair_types.shape != (edge_count,) or inverse.shape != (edge_count,):
        raise ValueError("pair_types and inverse must have one entry per directed edge")
    pair_types = pair_types.to(device=raw_prediction.device, dtype=torch.long)
    inverse = inverse.to(device=raw_prediction.device, dtype=torch.long)
    expected = torch.arange(edge_count, device=raw_prediction.device)
    if (
        torch.any(inverse < 0)
        or torch.any(inverse >= edge_count)
        or not torch.equal(inverse.index_select(0, inverse), expected)
    ):
        raise ValueError("inverse must be an in-range involution")

    pair_to_index = {name: index for index, name in enumerate(pair_names)}
    if len(pair_to_index) != len(pair_names):
        raise ValueError("pair_names must be unique")
    projected = torch.empty_like(raw_prediction)
    covered = torch.zeros(edge_count, dtype=torch.bool, device=raw_prediction.device)
    for pair_index, pair_name in enumerate(pair_names):
        selected = torch.nonzero(pair_types == pair_index, as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        pair = tuple(pair_name.split("-", maxsplit=1))
        reverse_pair = (pair[1], pair[0])
        reverse_name = "-".join(reverse_pair)
        if reverse_name not in pair_to_index:
            raise ValueError(f"pair_names is missing reverse pair {reverse_name}")
        reverse_indices = inverse.index_select(0, selected)
        if not torch.all(
            pair_types.index_select(0, reverse_indices) == pair_to_index[reverse_name]
        ):
            raise ValueError(f"inverse edges do not have reverse type for {pair_name}")
        reverse_in_forward_basis = transform.reverse(
            reverse_pair, raw_prediction.index_select(0, reverse_indices)
        ).to(raw_prediction)
        projected.index_copy_(
            0,
            selected,
            0.5 * (raw_prediction.index_select(0, selected) + reverse_in_forward_basis),
        )
        covered[selected] = True
    if not torch.all(covered):
        raise ValueError("pair_types contains an index absent from pair_names")
    return projected


def directed_hermiticity_relative_error(
    transform: FullBlockIrrepTransform,
    pair_names: Sequence[str],
    prediction: torch.Tensor,
    pair_types: torch.Tensor,
    inverse: torch.Tensor,
) -> float:
    """Return RMS reverse inconsistency divided by RMS prediction magnitude."""
    projected = project_directed_irreps(
        transform, pair_names, prediction, pair_types, inverse
    )
    numerator = torch.linalg.vector_norm(prediction - projected)
    denominator = torch.linalg.vector_norm(prediction).clamp_min(
        torch.finfo(prediction.dtype).tiny
    )
    return float((numerator / denominator).item())
