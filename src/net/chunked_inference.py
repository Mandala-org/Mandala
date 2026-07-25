"""Exact edge-streamed inference for large periodic graphs.

The training forward path intentionally remains monolithic.  This module is
used only under ``torch.inference_mode()`` and bounds the transient memory of
the equivariant tensor products by processing edges in contiguous chunks while
keeping the graph's global node set and canonical edge order intact.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TYPE_CHECKING

import torch
from torch_scatter import scatter

from data.block_matrix import BlockMatrix

if TYPE_CHECKING:
    from net.e3gnn import E3GNN


ProgressCallback = Callable[[str, int, int], None]


class EdgeStore:
    """Persistent edge states backed by GPU or pinned host memory."""

    def __init__(
        self,
        *,
        num_edges: int,
        feature_dim: int,
        dtype: torch.dtype,
        compute_device: torch.device,
        storage_device: str | torch.device,
    ) -> None:
        self.num_edges = num_edges
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.compute_device = compute_device
        self.storage_device = torch.device(storage_device)
        if (
            self.storage_device.type == compute_device.type
            and self.storage_device.index is None
        ):
            self.storage_device = compute_device
        if self.storage_device.type != "cpu" and self.storage_device != compute_device:
            raise ValueError(
                "edge_store_device must be 'cpu' or exactly the model compute "
                "device, got "
                f"{self.storage_device}."
            )
        pin_memory = self.storage_device.type == "cpu" and compute_device.type == "cuda"
        self.tensor = torch.empty(
            (num_edges, feature_dim),
            dtype=dtype,
            device=self.storage_device,
            pin_memory=pin_memory,
        )

    def get_slice(self, part: slice) -> torch.Tensor:
        value = self.tensor[part]
        if value.device == self.compute_device:
            return value
        return value.to(self.compute_device, non_blocking=value.is_pinned())

    def set_slice(self, part: slice, value: torch.Tensor) -> None:
        if self.tensor.device == value.device:
            self.tensor[part] = value
            return
        self.tensor[part].copy_(
            value.to(self.tensor.device, non_blocking=self.tensor.is_pinned())
        )

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        if self.tensor.device == self.compute_device:
            return self.tensor.index_select(0, indices)
        value = self.tensor.index_select(0, indices.to(device="cpu"))
        return value.to(self.compute_device, non_blocking=value.is_pinned())


def edge_slices(num_edges: int, chunk_size: int) -> Iterator[slice]:
    """Yield contiguous, non-empty edge slices with explicit validation."""
    if chunk_size <= 0:
        raise ValueError(f"edge_chunk_size must be > 0, got {chunk_size}")
    for start in range(0, num_edges, chunk_size):
        yield slice(start, min(start + chunk_size, num_edges))


def _notify(
    progress: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
) -> None:
    if progress is not None:
        progress(stage, current, total)


def _edge_chunk(x: dict[str, Any], part: slice) -> dict[str, torch.Tensor]:
    """Take a consistent edge-major chunk from the prepared model input."""
    return {
        "edge_index": x["edge_index"][:, part],
        "edge_sh": x["edge_sh"][part],
        "edge_length_emb": x["edge_length_emb"][part],
        "edge_one_hot": x["edge_one_hot"][part],
        "edge_type_idx": x["edge_type_idx"][part],
    }


def _edge_messages(
    *,
    block,
    node_pre: torch.Tensor,
    edge: torch.Tensor,
    chunk: dict[str, torch.Tensor],
) -> torch.Tensor:
    return block.node_upd.compute_edge_messages(
        node_pre=node_pre,
        edge=edge,
        edge_index=chunk["edge_index"],
        edge_sh=chunk["edge_sh"],
        edge_length_emb=chunk["edge_length_emb"],
        edge_type_idx=chunk["edge_type_idx"],
    )


def _stream_node_update(
    *,
    block,
    node: torch.Tensor,
    edge_store: EdgeStore,
    x: dict[str, Any],
    edge_chunk_size: int,
    progress: ProgressCallback | None,
    layer_index: int,
) -> torch.Tensor:
    """Compute an exact global node update while streaming edge activations."""
    num_edges = edge_store.num_edges
    num_nodes = int(node.shape[0])
    chunks = tuple(edge_slices(num_edges, edge_chunk_size))
    node_old, node_pre, node_self_connection = block.node_upd.prepare_node_update(
        node, x["node_one_hot"]
    )
    mode = block.node_upd.message_agg_mode

    if mode in {"sum", "average"}:
        aggregated = node.new_zeros((num_nodes, block.node_upd.conv.irreps_out.dim))
        counts = node.new_zeros((num_nodes, 1)) if mode == "average" else None
        for index, part in enumerate(chunks, start=1):
            chunk = _edge_chunk(x, part)
            messages = _edge_messages(
                block=block,
                node_pre=node_pre,
                edge=edge_store.get_slice(part),
                chunk=chunk,
            )
            dst = chunk["edge_index"][1]
            aggregated.index_add_(0, dst, messages)
            if counts is not None:
                counts.index_add_(
                    0, dst, torch.ones_like(dst, dtype=node.dtype)[:, None]
                )
            _notify(progress, f"layer{layer_index}:node:{mode}", index, len(chunks))
        if counts is not None:
            aggregated = aggregated / counts.clamp_min(1.0)
        return block.node_upd.finalize_node_update(
            aggregated=aggregated,
            node_old=node_old,
            node_self_connection=node_self_connection,
        )

    # Attention needs a global destination-wise softmax.  Pass A obtains its
    # exact stable max; pass B recomputes chunk-local messages and accumulates
    # the normalized numerator and denominator.
    query = block.node_upd.query_proj(node_pre).reshape(
        num_nodes, block.node_upd.attn_num_heads, block.node_upd.attn_head_dim
    )
    max_per_dst = node.new_full(
        (num_nodes, block.node_upd.attn_num_heads), float("-inf")
    )
    scale = float(block.node_upd.attn_head_dim) ** 0.5
    for index, part in enumerate(chunks, start=1):
        chunk = _edge_chunk(x, part)
        messages = _edge_messages(
            block=block,
            node_pre=node_pre,
            edge=edge_store.get_slice(part),
            chunk=chunk,
        )
        dst = chunk["edge_index"][1]
        key = block.node_upd.key_proj(
            torch.cat([chunk["edge_length_emb"], messages], dim=-1)
        ).reshape(-1, block.node_upd.attn_num_heads, block.node_upd.attn_head_dim)
        scores = (query.index_select(0, dst) * key).sum(dim=-1) / scale
        chunk_max = scatter(scores, dst, dim=0, dim_size=num_nodes, reduce="max")
        max_per_dst = torch.maximum(max_per_dst, chunk_max)
        _notify(progress, f"layer{layer_index}:node:attention_max", index, len(chunks))

    denom = node.new_zeros((num_nodes, block.node_upd.attn_num_heads))
    numerator = node.new_zeros(
        (num_nodes, block.node_upd.attn_num_heads, block.node_upd.conv.irreps_out.dim)
    )
    for index, part in enumerate(chunks, start=1):
        chunk = _edge_chunk(x, part)
        messages = _edge_messages(
            block=block,
            node_pre=node_pre,
            edge=edge_store.get_slice(part),
            chunk=chunk,
        )
        dst = chunk["edge_index"][1]
        key = block.node_upd.key_proj(
            torch.cat([chunk["edge_length_emb"], messages], dim=-1)
        ).reshape(-1, block.node_upd.attn_num_heads, block.node_upd.attn_head_dim)
        scores = (query.index_select(0, dst) * key).sum(dim=-1) / scale
        exp_scores = torch.exp(scores - max_per_dst.index_select(0, dst))
        values = torch.stack(
            [projection(messages) for projection in block.node_upd.value_projs], dim=1
        )
        denom.index_add_(0, dst, exp_scores)
        numerator.index_add_(0, dst, values * exp_scores.unsqueeze(-1))
        _notify(progress, f"layer{layer_index}:node:attention_sum", index, len(chunks))
    aggregated = (numerator / denom.clamp_min(1e-12).unsqueeze(-1)).mean(dim=1)
    return block.node_upd.finalize_node_update(
        aggregated=aggregated,
        node_old=node_old,
        node_self_connection=node_self_connection,
    )


def _stream_edge_update(
    *,
    block,
    node: torch.Tensor,
    edge_store: EdgeStore,
    x: dict[str, Any],
    edge_chunk_size: int,
    progress: ProgressCallback | None,
    layer_index: int,
) -> None:
    """Update edge storage in place after its layer's node update is complete."""
    chunks = tuple(edge_slices(edge_store.num_edges, edge_chunk_size))
    for index, part in enumerate(chunks, start=1):
        chunk = _edge_chunk(x, part)
        updated = block.edge_upd.forward_chunk(
            node=node,
            edge=edge_store.get_slice(part),
            edge_index=chunk["edge_index"],
            edge_sh=chunk["edge_sh"],
            edge_length_emb=chunk["edge_length_emb"],
            edge_one_hot=chunk["edge_one_hot"],
            edge_type_idx=chunk["edge_type_idx"],
        )
        edge_store.set_slice(part, updated)
        _notify(progress, f"layer{layer_index}:edge", index, len(chunks))


def _stream_head_to_blocks(
    *,
    model: "E3GNN",
    head,
    node: torch.Tensor,
    edge_store: EdgeStore,
    x: dict[str, Any],
    edge_chunk_size: int,
    output_device: torch.device,
    physical: bool,
    name: str,
    progress: ProgressCallback | None,
) -> BlockMatrix:
    """Run a head pair-by-pair and write physical blocks directly to output."""
    pair_blocks: dict[str, torch.Tensor] = {}
    pair_edges: dict[str, torch.Tensor] = {}
    for key in head.pair_keys:
        if key not in x["pred_pair_edges_static"] or key not in x["edge_partitions"]:
            continue
        parts = x["edge_partitions"][key]
        global_idx = parts["global_idx"]
        key_edges = x["pred_pair_edges_static"][key]
        block_dims = model.mapper.block_dims(key)
        output = torch.empty(
            (int(global_idx.numel()), *block_dims),
            device=output_device,
            dtype=edge_store.dtype,
        )
        chunks = tuple(edge_slices(int(global_idx.numel()), edge_chunk_size))
        for index, part in enumerate(chunks, start=1):
            pair_global_idx = global_idx[part]
            pair_edge_feat = edge_store.gather(pair_global_idx)
            vectors = head.forward_pair_chunk(
                key=key,
                node_feat=node,
                key_edge_feat=pair_edge_feat,
                key_edges=key_edges[:, part],
            )
            blocks = model.mapper.vectors_to_blocks(key, vectors)
            if (
                physical
                and name in {"hamiltonian", "overlap"}
                and model._matrix_envelope_mode() != "off"
            ):
                scale = x["edge_envelope"].index_select(0, pair_global_idx)
                blocks = blocks * scale[:, None, None]
            output[part] = blocks.to(device=output_device)
            _notify(progress, f"head:{name}:{key}", index, len(chunks))
        pair_blocks[key] = output
        pair_edges[key] = key_edges.to(device=output_device)
    return BlockMatrix(
        atoms=x["atoms_tuple"],
        atom_counts=x["atom_counts"],
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=x["pred_lookup_static"],
        orbital_cfg=model.mapper.orbital_cfg,
        basis="e3nn",
    )


def predict_matrices_chunked(
    model: "E3GNN",
    x: dict[str, Any],
    *,
    edge_chunk_size: int,
    edge_store_device: str | torch.device | None = None,
    output_device: str | torch.device = "cpu",
    physical: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, BlockMatrix]:
    """Run exact inference with bounded edge-local activation memory.

    By default the persistent edge state stays on the model device.  Passing
    ``edge_store_device='cpu'`` keeps it in pinned host RAM and transfers only
    chunks, trading throughput for a smaller CUDA residency.
    """
    if torch.is_grad_enabled():
        raise RuntimeError(
            "predict_matrices_chunked must run under inference_mode/no_grad."
        )
    if model.training:
        raise RuntimeError("predict_matrices_chunked requires model.eval().")
    if model._should_recompute_edge_features(x):
        model._populate_edge_features(x)
    required = {"edge_length_emb", "edge_sh", "edge_one_hot", "edge_type_idx"}
    missing = sorted(required.difference(x))
    if missing:
        raise ValueError(
            f"Chunked inference requires precomputed edge features: {missing}"
        )

    num_edges = int(x["edge_index"].shape[1])
    if num_edges == 0:
        raise ValueError("Chunked inference does not support graphs with zero edges.")
    output_device = torch.device(output_device)
    activation_mags = None
    node = model.node_enc(x["node_type_idx"], activation_mags=activation_mags)
    edge_store = EdgeStore(
        num_edges=num_edges,
        feature_dim=model.edge_enc.irreps_out.dim,
        dtype=node.dtype,
        compute_device=node.device,
        storage_device=node.device if edge_store_device is None else edge_store_device,
    )
    chunks = tuple(edge_slices(num_edges, edge_chunk_size))
    for index, part in enumerate(chunks, start=1):
        encoded = model.edge_enc(
            x["edge_type_idx"][part],
            x["edge_length_emb"][part],
            x["edge_sh"][part],
            activation_mags=activation_mags,
        )
        edge_store.set_slice(part, encoded)
        _notify(progress, "edge_encoder", index, len(chunks))

    for layer_index, block in enumerate(model.mp_blocks):
        node = _stream_node_update(
            block=block,
            node=node,
            edge_store=edge_store,
            x=x,
            edge_chunk_size=edge_chunk_size,
            progress=progress,
            layer_index=layer_index,
        )
        _stream_edge_update(
            block=block,
            node=node,
            edge_store=edge_store,
            x=x,
            edge_chunk_size=edge_chunk_size,
            progress=progress,
            layer_index=layer_index,
        )

    return {
        name: _stream_head_to_blocks(
            model=model,
            head=head,
            node=node,
            edge_store=edge_store,
            x=x,
            edge_chunk_size=edge_chunk_size,
            output_device=output_device,
            physical=physical,
            name=name,
            progress=progress,
        )
        for name, head in model.heads.items()
    }
