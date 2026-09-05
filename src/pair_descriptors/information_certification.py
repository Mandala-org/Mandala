"""Geometry-only numerical certification utilities for atomic descriptors."""

from __future__ import annotations

from dataclasses import dataclass
import math

from e3nn import o3
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
import torch


@dataclass(frozen=True, slots=True)
class InverseResult:
    normalized_descriptor_rms: float
    assigned_cartesian_rmsd_angstrom: float
    pair_distance_rmsd_angstrom: float
    recovered_coordinates: tuple[tuple[float, float, float], ...]


def expanded_irrep_scales(
    irreps, channel_rms: list[float], *, device=None
) -> torch.Tensor:
    """Expand one equivariance-preserving scalar scale per irrep copy."""
    values = []
    index = 0
    for multiplicity, irrep in irreps:
        for _copy in range(multiplicity):
            values.extend([channel_rms[index]] * irrep.dim)
            index += 1
    if index != len(channel_rms):
        raise ValueError("Normalization channel count does not match irreps")
    return torch.tensor(values, dtype=torch.float64, device=device)


def descriptor_jacobian(
    descriptor,
    coordinates: torch.Tensor,
    species: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Forward-mode Jacobian of the normalized descriptor, shape ``(K,3N)``."""
    columns = []
    for coordinate in range(coordinates.numel()):
        tangent = torch.zeros_like(coordinates)
        tangent.reshape(-1)[coordinate] = 1.0
        _, column = torch.autograd.functional.jvp(
            lambda value: descriptor(value, species)[0] / scales,
            coordinates,
            tangent,
            create_graph=False,
            strict=False,
        )
        columns.append(column)
    return torch.stack(columns, dim=1)


def species_assigned_rmsd(
    candidate: torch.Tensor, target: torch.Tensor, species: torch.Tensor
) -> float:
    """Cartesian RMSD after independent Hungarian matching within each species."""
    squared = 0.0
    count = 0
    for atomic_number in torch.unique(species):
        keep = torch.nonzero(species == atomic_number, as_tuple=False).flatten()
        cost = (
            torch.cdist(candidate[keep], target[keep]).square().detach().cpu().numpy()
        )
        rows, columns = linear_sum_assignment(cost)
        squared += float(cost[rows, columns].sum())
        count += len(rows)
    return math.sqrt(squared / max(count, 1))


def pair_distance_rmsd(candidate: torch.Tensor, target: torch.Tensor) -> float:
    """Permutation-independent sorted pair-distance discrepancy."""
    first = torch.sort(torch.pdist(candidate))[0]
    second = torch.sort(torch.pdist(target))[0]
    return (
        float(torch.sqrt(torch.mean((first - second).square()))) if len(first) else 0.0
    )


def geometry_signature(
    coordinates: torch.Tensor, species: torch.Tensor
) -> torch.Tensor:
    """Differentiable almost-everywhere permutation-invariant geometry signature."""
    parts = []
    radii = torch.linalg.vector_norm(coordinates, dim=-1)
    unique = torch.unique(species, sorted=True)
    for atomic_number in unique:
        keep = species == atomic_number
        parts.append(torch.sort(radii[keep])[0])
    for left_pos, left_species in enumerate(unique):
        left = coordinates[species == left_species]
        for right_species in unique[left_pos:]:
            right = coordinates[species == right_species]
            distances = torch.cdist(left, right).flatten()
            if left_species == right_species:
                size = len(left)
                mask = torch.triu(
                    torch.ones(size, size, dtype=torch.bool, device=coordinates.device),
                    diagonal=1,
                ).flatten()
                distances = distances[mask]
            parts.append(torch.sort(distances)[0])
    return torch.cat([part for part in parts if part.numel()])


def _batched_descriptor(descriptor, coordinates: torch.Tensor, species: torch.Tensor):
    starts, neighbors, _ = coordinates.shape
    centers = torch.arange(starts, device=coordinates.device).repeat_interleave(
        neighbors
    )
    return descriptor(
        coordinates.reshape(-1, 3),
        species.repeat(starts),
        centers,
        num_centers=starts,
    )


def _project_inside(coordinates: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        norm = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True)
        coordinates.mul_(torch.clamp(radius / norm.clamp_min(1e-12), max=1.0))


def constructive_l0_l1_initializer(
    descriptor,
    target_descriptor: torch.Tensor,
    species: torch.Tensor,
    scales: torch.Tensor,
    channels: list[dict[str, object]],
    *,
    starts: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Recover radii from l=0 channels, then directions from l=1 channels.

    This is a finite numerical realization of the radial-moment/Vandermonde
    route.  It requires at least as many independent l=1 radial channels as
    same-species points; otherwise the caller should use generic multistart.
    """
    if target_descriptor.device.type != "cpu":
        raise ValueError("Constructive initializer currently runs on CPU")
    generator = np.random.default_rng(seed)
    recovered = torch.empty((len(species), 3), dtype=torch.float64)
    basis = o3.spherical_harmonics(
        1,
        torch.eye(3, dtype=torch.float64),
        normalize=True,
        normalization="component",
    )
    inverse_basis = torch.linalg.inv(basis)
    total_residual = 0.0
    for atomic_number_tensor in torch.unique(species, sorted=True):
        atomic_number = int(atomic_number_tensor)
        locations = torch.nonzero(
            species == atomic_number_tensor, as_tuple=False
        ).flatten()
        count = len(locations)
        scalar_channels = sorted(
            [
                channel
                for channel in channels
                if int(channel["species"]) == atomic_number and int(channel["l"]) == 0
            ],
            key=lambda channel: int(
                channel.get("radial_index", channel.get("frequency_index", 0))
            ),
        )
        vector_channels = sorted(
            [
                channel
                for channel in channels
                if int(channel["species"]) == atomic_number and int(channel["l"]) == 1
            ],
            key=lambda channel: int(
                channel.get("radial_index", channel.get("frequency_index", 0))
            ),
        )
        if len(scalar_channels) < count or len(vector_channels) < count:
            raise ValueError(
                f"Insufficient l=0/l=1 channels for {count} atoms of Z={atomic_number}"
            )
        scalar_indices = torch.tensor(
            [int(channel["start"]) for channel in scalar_channels], dtype=torch.long
        )
        scalar_target = target_descriptor[scalar_indices]
        scalar_scale = scales[scalar_indices]
        local_species = torch.full((count,), atomic_number, dtype=torch.long)

        def residual(radii_array: np.ndarray) -> np.ndarray:
            radii = torch.from_numpy(np.sort(radii_array)).to(torch.float64)
            trial = torch.zeros((count, 3), dtype=torch.float64)
            trial[:, 0] = radii
            with torch.no_grad():
                values = descriptor(trial, local_species)[0][scalar_indices]
            return ((values - scalar_target) / scalar_scale).numpy()

        best = None
        for _attempt in range(starts):
            initial = np.sort(
                generator.uniform(
                    0.08 * descriptor.cutoff, 0.82 * descriptor.cutoff, count
                )
            )
            candidate = least_squares(
                residual,
                initial,
                bounds=(0.03 * descriptor.cutoff, 0.85 * descriptor.cutoff),
                max_nfev=500,
                ftol=1e-13,
                xtol=1e-13,
                gtol=1e-13,
            )
            score = float(np.sqrt(np.mean(residual(candidate.x) ** 2)))
            if best is None or score < best[0]:
                best = (score, np.sort(candidate.x))
        if best is None:  # pragma: no cover - starts is validated by callers
            raise RuntimeError("No radial reconstruction attempt was made")
        total_residual = max(total_residual, best[0])
        radii = torch.from_numpy(best[1]).to(torch.float64)
        trial = torch.zeros((count, 3), dtype=torch.float64)
        trial[:, 0] = radii
        contributions = descriptor.contribution(local_species, trial)
        known_y = basis[0]
        radial_matrix = torch.empty((len(vector_channels), count), dtype=torch.float64)
        target_y = torch.empty((len(vector_channels), 3), dtype=torch.float64)
        for row, channel in enumerate(vector_channels):
            start, stop = int(channel["start"]), int(channel["stop"])
            target_y[row] = target_descriptor[start:stop]
            radial_matrix[row] = (contributions[:, start:stop] @ known_y) / torch.dot(
                known_y, known_y
            )
        recovered_y = torch.linalg.lstsq(radial_matrix, target_y).solution
        unit = recovered_y @ inverse_basis
        unit /= torch.linalg.vector_norm(unit, dim=-1, keepdim=True).clamp_min(1e-12)
        recovered[locations] = radii[:, None] * unit
    return recovered, total_residual


def reconstruct_multistart(
    descriptor,
    target_coordinates: torch.Tensor,
    species: torch.Tensor,
    scales: torch.Tensor,
    *,
    starts: int,
    steps: int,
    polish_steps: int,
    learning_rate: float,
    seed: int,
    stage_masks: tuple[torch.Tensor, ...] | None = None,
    polish_candidates: int = 4,
    initial_coordinates: torch.Tensor | None = None,
    radial_fraction: float = 0.85,
) -> InverseResult:
    """Recover a fixed-count/species neighborhood using continuation in ℓ."""
    generator = torch.Generator(device=target_coordinates.device).manual_seed(seed)
    coordinates = torch.randn(
        starts,
        len(target_coordinates),
        3,
        generator=generator,
        dtype=torch.float64,
        device=target_coordinates.device,
    )
    coordinates *= descriptor.cutoff * 0.45
    _project_inside(coordinates, descriptor.cutoff * radial_fraction)
    if initial_coordinates is not None:
        if initial_coordinates.shape != coordinates.shape[1:]:
            raise ValueError("initial_coordinates shape does not match environment")
        coordinates[0] = initial_coordinates
    coordinates.requires_grad_(True)
    target = descriptor(target_coordinates, species)[0].detach()
    if stage_masks is None:
        stage_masks = (torch.ones_like(scales, dtype=torch.bool),)
    stage_masks = tuple(mask.to(device=scales.device) for mask in stage_masks)
    stage_steps = max(1, steps // len(stage_masks))
    for mask in stage_masks:
        optimizer = torch.optim.Adam([coordinates], lr=learning_rate)
        for _step in range(stage_steps):
            optimizer.zero_grad(set_to_none=True)
            values = _batched_descriptor(descriptor, coordinates, species)
            per_start = (
                ((values[:, mask] - target[mask]) / scales[mask]).square().mean(dim=-1)
            )
            radii = torch.linalg.vector_norm(coordinates, dim=-1)
            boundary = (
                torch.relu(radii - descriptor.cutoff * radial_fraction)
                .square()
                .mean(dim=-1)
            )
            loss = (per_start + 100.0 * boundary).sum()
            loss.backward()
            optimizer.step()
            _project_inside(coordinates, descriptor.cutoff * radial_fraction)
    if polish_steps:
        with torch.no_grad():
            values = _batched_descriptor(descriptor, coordinates, species)
            residuals = ((values - target) / scales).square().mean(dim=-1)
            candidates = torch.argsort(residuals)[: min(polish_candidates, starts)]
        for candidate in candidates.tolist():
            point = coordinates[candidate : candidate + 1].detach().clone()
            point.requires_grad_(True)
            optimizer_lbfgs = torch.optim.LBFGS(
                [point],
                lr=0.8,
                max_iter=polish_steps,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer_lbfgs.zero_grad(set_to_none=True)
                values = _batched_descriptor(descriptor, point, species)[0]
                loss = (((values - target) / scales).square()).mean()
                radii = torch.linalg.vector_norm(point, dim=-1)
                loss = (
                    loss
                    + 100.0
                    * torch.relu(radii - descriptor.cutoff * radial_fraction)
                    .square()
                    .mean()
                )
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)
            _project_inside(point, descriptor.cutoff * radial_fraction)
            with torch.no_grad():
                coordinates[candidate].copy_(point[0])
    with torch.no_grad():
        values = _batched_descriptor(descriptor, coordinates, species)
        residuals = torch.sqrt(((values - target) / scales).square().mean(dim=-1))
        best = int(torch.argmin(residuals))
        recovered = coordinates[best].detach()
    return InverseResult(
        normalized_descriptor_rms=float(residuals[best]),
        assigned_cartesian_rmsd_angstrom=species_assigned_rmsd(
            recovered, target_coordinates, species
        ),
        pair_distance_rmsd_angstrom=pair_distance_rmsd(recovered, target_coordinates),
        recovered_coordinates=tuple(
            tuple(float(value) for value in row) for row in recovered.cpu()
        ),
    )


def adversarial_collision_search(
    descriptor,
    target_coordinates: torch.Tensor,
    species: torch.Tensor,
    scales: torch.Tensor,
    *,
    starts: int,
    steps: int,
    learning_rate: float,
    minimum_geometry_rms: float,
    seed: int,
) -> dict[str, float]:
    """Search for descriptor matches constrained away in geometry-signature space."""
    generator = torch.Generator(device=target_coordinates.device).manual_seed(seed)
    coordinates = target_coordinates.repeat(starts, 1, 1)
    coordinates += torch.randn(
        coordinates.shape,
        generator=generator,
        dtype=coordinates.dtype,
        device=coordinates.device,
    ) * (0.12 * descriptor.cutoff)
    _project_inside(coordinates, descriptor.cutoff * 0.85)
    coordinates.requires_grad_(True)
    target_descriptor = descriptor(target_coordinates, species)[0].detach()
    target_signature = geometry_signature(target_coordinates, species).detach()
    optimizer = torch.optim.Adam([coordinates], lr=learning_rate)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        values = _batched_descriptor(descriptor, coordinates, species)
        descriptor_loss = ((values - target_descriptor) / scales).square().mean(dim=-1)
        geometry_rms = torch.stack(
            [
                torch.sqrt(
                    torch.mean(
                        (geometry_signature(value, species) - target_signature).square()
                    )
                )
                for value in coordinates
            ]
        )
        constraint = torch.relu(minimum_geometry_rms - geometry_rms).square()
        (descriptor_loss + 10.0 * constraint).sum().backward()
        optimizer.step()
        _project_inside(coordinates, descriptor.cutoff * 0.85)
    with torch.no_grad():
        values = _batched_descriptor(descriptor, coordinates, species)
        residuals = torch.sqrt(
            ((values - target_descriptor) / scales).square().mean(dim=-1)
        )
        geometry_rms = torch.stack(
            [
                torch.sqrt(
                    torch.mean(
                        (geometry_signature(value, species) - target_signature).square()
                    )
                )
                for value in coordinates
            ]
        )
        feasible = geometry_rms >= minimum_geometry_rms
        score = torch.where(
            feasible, residuals, torch.full_like(residuals, float("inf"))
        )
        best = int(torch.argmin(score if feasible.any() else residuals))
    return {
        "normalized_descriptor_rms": float(residuals[best]),
        "geometry_signature_rms_angstrom": float(geometry_rms[best]),
        "assigned_cartesian_rmsd_angstrom": species_assigned_rmsd(
            coordinates[best], target_coordinates, species
        ),
        "pair_distance_rmsd_angstrom": pair_distance_rmsd(
            coordinates[best], target_coordinates
        ),
    }


def null_direction_continuation(
    descriptor,
    target_coordinates: torch.Tensor,
    species: torch.Tensor,
    scales: torch.Tensor,
    jacobian: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    """Step along the weakest Jacobian direction and project toward its level set."""
    _u, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
    direction = vh[-1].reshape_as(target_coordinates)
    coordinates = (
        (
            target_coordinates
            + 0.05 * descriptor.cutoff * direction / torch.linalg.vector_norm(direction)
        )
        .detach()
        .requires_grad_(True)
    )
    target = descriptor(target_coordinates, species)[0].detach()
    optimizer = torch.optim.Adam([coordinates], lr=learning_rate)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (
            ((descriptor(coordinates, species)[0] - target) / scales).square()
        ).mean()
        loss.backward()
        optimizer.step()
        _project_inside(coordinates, descriptor.cutoff * 0.85)
    residual = torch.sqrt(
        (((descriptor(coordinates, species)[0] - target) / scales).square()).mean()
    )
    signature = torch.sqrt(
        torch.mean(
            (
                geometry_signature(coordinates, species)
                - geometry_signature(target_coordinates, species)
            ).square()
        )
    )
    return {
        "initial_smallest_singular_value": float(singular[-1]),
        "normalized_descriptor_rms": float(residual),
        "geometry_signature_rms_angstrom": float(signature),
    }
