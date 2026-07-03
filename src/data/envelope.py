from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class PairEnvelopeTable:
    family: str
    pair_keys: tuple[str, ...]
    parameters: torch.Tensor
    reference_x_max: torch.Tensor | None = None
    r0_parameter_index: int | None = None

    @property
    def r0(self) -> torch.Tensor:
        if self.r0_parameter_index is None:
            raise ValueError(
                f"Envelope family {self.family!r} does not define an r0 parameter."
            )
        return self.parameters[:, self.r0_parameter_index]

    @property
    def pair_to_index(self) -> dict[str, int]:
        return {pair: idx for idx, pair in enumerate(self.pair_keys)}


def _resolve_pair_key(pair: str, pairs: dict[str, object]) -> str:
    if pair in pairs:
        return pair
    left, right = pair.split("-", maxsplit=1)
    reversed_pair = f"{right}-{left}"
    if reversed_pair in pairs:
        return reversed_pair
    raise ValueError(f"Envelope artifact is missing parameters for pair {pair!r}.")


def _family_parameter_count(family: str) -> int:
    if family == "slater_soft_cutoff":
        return 5
    if family == "slater_exp_quad_soft_wall":
        return 6
    raise ValueError(
        f"Unsupported envelope family {family!r}. "
        "Supported families: 'slater_soft_cutoff', 'slater_exp_quad_soft_wall'."
    )


def _family_r0_parameter_index(family: str) -> int | None:
    if family == "slater_soft_cutoff":
        return 3
    if family == "slater_exp_quad_soft_wall":
        return None
    raise ValueError(
        f"Unsupported envelope family {family!r}. "
        "Supported families: 'slater_soft_cutoff', 'slater_exp_quad_soft_wall'."
    )


def load_pair_envelope_table(
    path: str | Path,
    *,
    pair_order: tuple[str, ...] | list[str] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PairEnvelopeTable:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Envelope artifact not found: {path}")

    payload = json.loads(path.read_text())
    family = payload.get("family")
    if not isinstance(family, str):
        raise ValueError(f"Envelope artifact {path} is missing a string family field.")
    n_params = _family_parameter_count(family)
    r0_parameter_index = _family_r0_parameter_index(family)

    pairs = payload.get("pairs")
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError(f"Envelope artifact {path} does not contain any pair data.")

    if pair_order is None:
        pair_keys = tuple(sorted(pairs.keys()))
    else:
        pair_keys = tuple(pair_order)
    params = []
    reference_x_max = []
    for pair in pair_keys:
        entry_key = _resolve_pair_key(pair, pairs)
        entry = pairs[entry_key]
        if not isinstance(entry, dict):
            raise ValueError(
                f"Envelope entry for pair {entry_key!r} must be a mapping."
            )
        theta = entry.get("theta")
        if not isinstance(theta, list) or len(theta) != n_params:
            raise ValueError(
                f"Envelope entry for pair {entry_key!r} must contain {n_params} "
                f"theta values for family {family!r}."
            )
        params.append(theta)
        reference_x_max.append(float(entry.get("x_max", "nan")))

    parameters = torch.tensor(params, dtype=dtype, device=device)
    reference_x_max_tensor = torch.tensor(
        reference_x_max,
        dtype=dtype,
        device=device,
    )
    return PairEnvelopeTable(
        family=family,
        pair_keys=pair_keys,
        parameters=parameters,
        reference_x_max=reference_x_max_tensor,
        r0_parameter_index=r0_parameter_index,
    )


def load_slater_soft_cutoff_envelope_table(
    path: str | Path,
    *,
    pair_order: tuple[str, ...] | list[str] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> PairEnvelopeTable:
    table = load_pair_envelope_table(
        path,
        pair_order=pair_order,
        dtype=dtype,
        device=device,
    )
    if table.family != "slater_soft_cutoff":
        raise ValueError(
            f"Envelope artifact {path} has family={table.family!r}, expected "
            "'slater_soft_cutoff'."
        )
    return table


def evaluate_slater_soft_cutoff(
    edge_lengths: torch.Tensor,
    pair_params: torch.Tensor,
) -> torch.Tensor:
    if pair_params.ndim != 2 or pair_params.shape[-1] != 5:
        raise ValueError(
            f"pair_params must have shape (E, 5), got {tuple(pair_params.shape)}"
        )
    edge_lengths = edge_lengths.to(device=pair_params.device, dtype=pair_params.dtype)
    amp = torch.exp(pair_params[:, 0])
    rate = torch.exp(pair_params[:, 1])
    nu = pair_params[:, 2]
    r0 = pair_params[:, 3]
    tau = torch.exp(pair_params[:, 4])
    z = torch.clamp((edge_lengths - r0) / tau, min=-60.0, max=60.0)
    return (
        amp
        * torch.pow(1.0 + edge_lengths, nu)
        * torch.exp(-rate * edge_lengths)
        / (1.0 + torch.exp(z))
    )


def evaluate_slater_exp_quad_soft_wall(
    edge_lengths: torch.Tensor,
    pair_params: torch.Tensor,
    reference_x_max: torch.Tensor,
) -> torch.Tensor:
    if pair_params.ndim != 2 or pair_params.shape[-1] != 6:
        raise ValueError(
            f"pair_params must have shape (E, 6), got {tuple(pair_params.shape)}"
        )
    if reference_x_max.shape != edge_lengths.shape:
        raise ValueError(
            "reference_x_max must match edge_lengths shape, got "
            f"{tuple(reference_x_max.shape)} and {tuple(edge_lengths.shape)}"
        )
    edge_lengths = edge_lengths.to(device=pair_params.device, dtype=pair_params.dtype)
    reference_x_max = reference_x_max.to(
        device=pair_params.device,
        dtype=pair_params.dtype,
    )
    amp = torch.exp(pair_params[:, 0])
    b = torch.exp(pair_params[:, 1])
    c = torch.exp(pair_params[:, 2])
    nu = pair_params[:, 3]
    if torch.isnan(reference_x_max).any():
        raise ValueError(
            "slater_exp_quad_soft_wall envelope requires per-pair x_max values in "
            "the envelope artifact."
        )
    rc = reference_x_max + torch.exp(pair_params[:, 4])
    gamma = torch.exp(pair_params[:, 5])
    wall = torch.clamp(rc - edge_lengths, min=1.0e-9)
    return (
        amp
        * torch.pow(1.0 + edge_lengths, nu)
        * torch.exp(-b * edge_lengths - c * edge_lengths * edge_lengths - gamma / wall)
    )


def build_edge_envelope(
    edge_lengths: torch.Tensor,
    edge_type_idx: torch.Tensor,
    envelope_table: PairEnvelopeTable,
) -> torch.Tensor:
    if edge_lengths.shape != edge_type_idx.shape:
        raise ValueError(
            "edge_lengths and edge_type_idx must have identical shapes, got "
            f"{tuple(edge_lengths.shape)} and {tuple(edge_type_idx.shape)}"
        )
    pair_params = envelope_table.parameters.index_select(0, edge_type_idx)
    if envelope_table.family == "slater_soft_cutoff":
        return evaluate_slater_soft_cutoff(edge_lengths, pair_params)
    if envelope_table.family == "slater_exp_quad_soft_wall":
        if envelope_table.reference_x_max is None:
            raise ValueError(
                "slater_exp_quad_soft_wall envelope requires reference_x_max data."
            )
        pair_reference_x_max = envelope_table.reference_x_max.index_select(
            0, edge_type_idx
        )
        return evaluate_slater_exp_quad_soft_wall(
            edge_lengths,
            pair_params,
            pair_reference_x_max,
        )
    raise ValueError(
        f"Unsupported envelope family {envelope_table.family!r}. "
        "Supported families: 'slater_soft_cutoff', 'slater_exp_quad_soft_wall'."
    )


def build_edge_r0_lookup(
    edge_type_idx: torch.Tensor,
    envelope_table: PairEnvelopeTable,
) -> torch.Tensor:
    if envelope_table.r0_parameter_index is None:
        raise ValueError(
            f"Envelope family {envelope_table.family!r} does not support r0 lookup."
        )
    return envelope_table.r0.index_select(0, edge_type_idx)


def scale_block_matrix_by_edge_values(
    matrix,
    edge_values: torch.Tensor,
    edge_partitions: dict[str, dict[str, torch.Tensor]],
    *,
    inverse: bool = False,
    eps: float = EPS,
    allow_prefix_trim: bool = False,
):
    from data.block_matrix import BlockMatrix

    if not isinstance(matrix, BlockMatrix):
        raise TypeError(f"matrix must be BlockMatrix, got {type(matrix)!r}")
    pair_blocks: dict[str, torch.Tensor] = {}
    for key, blocks in matrix.pair_blocks.items():
        parts = edge_partitions.get(key)
        if parts is None:
            raise ValueError(f"Missing edge partition for key {key!r}")
        global_idx = parts["global_idx"]
        if allow_prefix_trim and global_idx.numel() >= blocks.shape[0]:
            global_idx = global_idx[: blocks.shape[0]]
        if global_idx.numel() != blocks.shape[0]:
            raise ValueError(
                f"Edge partition length mismatch for key {key!r}: "
                f"partition={int(global_idx.numel())} blocks={int(blocks.shape[0])}"
            )
        scale = edge_values.index_select(0, global_idx)
        if inverse:
            scale = scale.clamp_min(eps).reciprocal()
        pair_blocks[key] = blocks * scale[:, None, None]
    return BlockMatrix(
        atoms=matrix.atoms,
        atom_counts=matrix.atom_counts,
        pair_blocks=pair_blocks,
        pair_edges=matrix.pair_edges,
        lookup=matrix.lookup,
        orbital_cfg=matrix.orbital_cfg,
        basis=matrix.basis,
    )
