from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class SlaterSoftCutoffEnvelopeTable:
    pair_keys: tuple[str, ...]
    parameters: torch.Tensor  # shape (P, 5): log_amp, log_rate, nu, r0, log_tau

    @property
    def r0(self) -> torch.Tensor:
        return self.parameters[:, 3]

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


def load_slater_soft_cutoff_envelope_table(
    path: str | Path,
    *,
    pair_order: tuple[str, ...] | list[str] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> SlaterSoftCutoffEnvelopeTable:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Envelope artifact not found: {path}")

    payload = json.loads(path.read_text())
    family = payload.get("family")
    if family != "slater_soft_cutoff":
        raise ValueError(
            f"Envelope artifact {path} has family={family!r}, expected "
            "'slater_soft_cutoff'."
        )
    pairs = payload.get("pairs")
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError(f"Envelope artifact {path} does not contain any pair data.")

    if pair_order is None:
        pair_keys = tuple(sorted(pairs.keys()))
    else:
        pair_keys = tuple(pair_order)
    params = []
    for pair in pair_keys:
        entry_key = _resolve_pair_key(pair, pairs)
        entry = pairs[entry_key]
        if not isinstance(entry, dict):
            raise ValueError(
                f"Envelope entry for pair {entry_key!r} must be a mapping."
            )
        theta = entry.get("theta")
        if not isinstance(theta, list) or len(theta) != 5:
            raise ValueError(
                f"Envelope entry for pair {entry_key!r} must contain 5 theta values."
            )
        params.append(theta)

    parameters = torch.tensor(params, dtype=dtype, device=device)
    return SlaterSoftCutoffEnvelopeTable(pair_keys=pair_keys, parameters=parameters)


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


def build_edge_envelope(
    edge_lengths: torch.Tensor,
    edge_type_idx: torch.Tensor,
    envelope_table: SlaterSoftCutoffEnvelopeTable,
) -> torch.Tensor:
    if edge_lengths.shape != edge_type_idx.shape:
        raise ValueError(
            "edge_lengths and edge_type_idx must have identical shapes, got "
            f"{tuple(edge_lengths.shape)} and {tuple(edge_type_idx.shape)}"
        )
    pair_params = envelope_table.parameters.index_select(0, edge_type_idx)
    return evaluate_slater_soft_cutoff(edge_lengths, pair_params)


def build_edge_r0_lookup(
    edge_type_idx: torch.Tensor,
    envelope_table: SlaterSoftCutoffEnvelopeTable,
) -> torch.Tensor:
    return envelope_table.r0.index_select(0, edge_type_idx)


def scale_block_matrix_by_edge_values(
    matrix,
    edge_values: torch.Tensor,
    edge_partitions: dict[str, dict[str, torch.Tensor]],
    *,
    inverse: bool = False,
    eps: float = EPS,
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
