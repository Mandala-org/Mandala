"""Norms with an explicit derivative convention at zero-length vectors."""

import torch


def zero_safe_norm(values: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
    """Euclidean norm with zero first/second derivatives at the origin.

    The Euclidean norm is not differentiable at zero. PyTorch chooses a zero
    first derivative there, but differentiating that result again produces
    NaNs. Zero-length self edges and identically zero irrep channels occur
    routinely in force training. Evaluate their inactive norm branch at a
    nonzero constant so higher-order autograd respects the same zero convention.
    Nonzero vectors retain the exact Euclidean norm and its derivatives.
    """
    zero = (values == 0).all(dim=dim, keepdim=True)
    safe_values = torch.where(zero, torch.ones_like(values), values)
    norm = torch.linalg.vector_norm(safe_values, dim=dim, keepdim=True)
    norm = torch.where(zero, torch.zeros_like(norm), norm)
    return norm if keepdim else norm.squeeze(dim)
