"""
Common classes and functions for minimal E(3)-GNN study.

Shared between training and analysis scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct
from e3nn.nn import Gate
from torch_scatter import scatter
from torch_geometric.utils import degree
import wandb


class e3LayerNorm(nn.Module):
    """E(3)-equivariant layer normalization (from DeepH-E3)."""

    def __init__(self, irreps_in, eps=1e-5, affine=True, normalization="component"):
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        self.eps = eps
        self.normalization = normalization

        if affine:
            ib, iw = 0, 0
            weight_slices, bias_slices = [], []
            for mul, ir in irreps_in:
                if ir.is_scalar():  # bias only to 0e
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones([iw]))
            self.bias = nn.Parameter(torch.zeros([ib]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        print(f"    [e3LayerNorm] Irreps: {self.irreps_in}, affine={affine}")

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        if batch is None:
            batch = torch.full([x.shape[0]], 0, dtype=torch.int64, device=x.device)

        batch_size = int(batch.max()) + 1
        batch_degree = (
            degree(batch, batch_size, dtype=torch.int64).clamp_(min=1).to(dtype=x.dtype)
        )

        out = []
        ix = 0
        for index, (mul, ir) in enumerate(self.irreps_in):
            field = x[:, ix : ix + mul * ir.dim].reshape(-1, mul, ir.dim)

            # Subtract mean for scalars
            if ir.l == 0:
                mean = (
                    scatter(
                        field, batch, dim=0, dim_size=batch_size, reduce="add"
                    ).mean(dim=1, keepdim=True)
                    / batch_degree[:, None, None]
                )
                field = field - mean[batch]

            # Normalize
            norm = scatter(
                field.abs().pow(2), batch, dim=0, dim_size=batch_size, reduce="mean"
            ).mean(dim=[1, 2], keepdim=True)
            if self.normalization == "norm":
                norm = norm * ir.dim
            field = field / (norm.sqrt()[batch] + self.eps)

            # Affine transformation
            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]
            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[index]]
                field = field + bias[None, :, None]

            out.append(field.reshape(-1, mul * ir.dim))
            ix += mul * ir.dim

        return torch.cat(out, dim=-1)


def log_activation_magnitudes(
    features, irreps, name="", wandb_prefix="", log_to_wandb=False
):
    """Log per-irrep activation magnitudes."""
    stats = []
    ix = 0
    for mul, ir in irreps:
        field = features[:, ix : ix + mul * ir.dim]
        norms = torch.norm(field.reshape(-1, ir.dim), dim=-1)
        mean_mag = norms.mean().item()
        median_mag = norms.median().item()
        stats.append(f"{ir}: {mean_mag:.4f}")

        # Log to wandb if requested
        if log_to_wandb and wandb_prefix:
            wandb.log(
                {
                    f"activations/{wandb_prefix}/{ir}_mean": mean_mag,
                    f"activations/{wandb_prefix}/{ir}_median": median_mag,
                }
            )

        ix += mul * ir.dim

    print(f"      [{name}] Magnitudes per irrep: {', '.join(stats)}")


class MinimalNodeEncoder(nn.Module):
    """Encode element type to scalar features."""

    def __init__(self, num_elements, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_elements, hidden_dim)
        self.irreps_out = Irreps(f"{hidden_dim}x0e")
        print(
            f"    [NodeEncoder] Input: {num_elements} elements → Output: {self.irreps_out}"
        )

    def forward(self, node_type_idx):
        out = self.embedding(node_type_idx)
        print(
            f"      [NodeEncoder.forward] Input shape: {node_type_idx.shape} → Output: {out.shape} (irreps: {self.irreps_out})"
        )
        return out


class MinimalEdgeEncoder(nn.Module):
    """Encode edge distance + edge type + spherical harmonics with Gate nonlinearity."""

    def __init__(self, n_radial, num_edge_types, hidden_irreps, sh_irreps):
        super().__init__()
        self.num_edge_types = num_edge_types

        # Linear projection from radial basis + edge type one-hot to scalars
        from e3nn.o3 import Irrep

        scalar_dim = sum(mul for mul, ir in hidden_irreps if ir == Irrep("0e"))
        self.radial_proj = nn.Linear(n_radial + num_edge_types, scalar_dim)

        # Gate nonlinearity setup
        irreps_scalars = Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l == 0])
        irreps_gated = Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l > 0])
        irreps_gates = Irreps([(mul, "0e") for mul, _ in irreps_gated])

        # TP output must produce both scalars, gates, and gated features
        irreps_tp_out = irreps_scalars + irreps_gates + irreps_gated

        # Tensor product: scalars ⊗ SH → TP output
        self.tp = FullyConnectedTensorProduct(
            Irreps(f"{scalar_dim}x0e"),
            sh_irreps,
            irreps_tp_out,
            internal_weights=True,
            shared_weights=True,
        )

        # Gate nonlinearity
        self.gate = Gate(
            irreps_scalars,
            [F.silu] * len(irreps_scalars),
            irreps_gates,
            [torch.sigmoid] * len(irreps_gates),
            irreps_gated,
        )
        self.irreps_out = self.gate.irreps_out

        # Layer norm
        self.norm = e3LayerNorm(self.irreps_out)

        print(
            f"    [EdgeEncoder] Irreps in: radial({n_radial}) + edge_type({num_edge_types}) → scalars({scalar_dim}x0e)"
        )
        print(f"                  TP: {scalar_dim}x0e ⊗ {sh_irreps} → {irreps_tp_out}")
        print(f"                  Gate: {irreps_tp_out} → {self.irreps_out}")
        print(f"                  Norm: {self.irreps_out}")

    def forward(
        self, edge_length_emb, edge_type_idx, edge_sh, batch_edge, log_to_wandb=False
    ):
        # Create edge type one-hot
        edge_type_onehot = F.one_hot(
            edge_type_idx, num_classes=self.num_edge_types
        ).float()

        # Concatenate radial and edge type features
        combined = torch.cat([edge_length_emb, edge_type_onehot], dim=-1)

        # Project to scalars
        radial_feat = self.radial_proj(combined)  # (E, scalar_dim)

        # Tensor product with spherical harmonics
        tp_out = self.tp(radial_feat, edge_sh)

        # Gate nonlinearity
        edge_feat = self.gate(tp_out)

        # Layer norm
        edge_feat = self.norm(edge_feat, batch_edge)

        print(
            f"      [EdgeEncoder.forward] Radial: {edge_length_emb.shape}, EdgeType: {edge_type_onehot.shape} → Combined: {combined.shape}"
        )
        print(
            f"                            → Scalars: {radial_feat.shape} (irreps: {self.tp.irreps_in1})"
        )
        print(
            f"                            ⊗ SH: {edge_sh.shape} (irreps: {self.tp.irreps_in2})"
        )
        print(
            f"                            → TP out: {tp_out.shape} (irreps: {self.tp.irreps_out})"
        )
        print(
            f"                            → Gate out: {edge_feat.shape} (irreps: {self.irreps_out})"
        )
        log_activation_magnitudes(
            edge_feat,
            self.irreps_out,
            "EdgeEncoder activations",
            "EdgeEncoder",
            log_to_wandb,
        )
        return edge_feat


class MinimalMessageBlock(nn.Module):
    """Message passing layer with edge update + node update with Gate and LayerNorm."""

    def __init__(self, node_irreps, edge_irreps, hidden_irreps, sh_irreps, layer_idx=0):
        super().__init__()
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps
        self.hidden_irreps = hidden_irreps
        self.layer_idx = layer_idx

        # Edge update: concat(src_node, dst_node, edge) ⊗ SH → TP output
        concat_irreps = node_irreps + node_irreps + edge_irreps

        # Gate setup for edge update
        irreps_scalars = Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l == 0])
        irreps_gated = Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l > 0])
        irreps_gates = Irreps([(mul, "0e") for mul, _ in irreps_gated])
        irreps_tp_out = irreps_scalars + irreps_gates + irreps_gated

        self.edge_update_tp = FullyConnectedTensorProduct(
            concat_irreps,
            sh_irreps,
            irreps_tp_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.edge_gate = Gate(
            irreps_scalars,
            [F.silu] * len(irreps_scalars),
            irreps_gates,
            [torch.sigmoid] * len(irreps_gates),
            irreps_gated,
        )
        self.edge_norm = e3LayerNorm(self.edge_gate.irreps_out)

        # Node update: aggregate messages + self-connection → TP output
        irreps_node_tp_out = irreps_scalars + irreps_gates + irreps_gated
        self.node_update_lin = Linear(hidden_irreps + node_irreps, irreps_node_tp_out)

        self.node_gate = Gate(
            irreps_scalars,
            [F.silu] * len(irreps_scalars),
            irreps_gates,
            [torch.sigmoid] * len(irreps_gates),
            irreps_gated,
        )
        self.node_norm = e3LayerNorm(self.node_gate.irreps_out)

        print(f"    [MessageBlock] Irreps:")
        print(
            f"      Edge update: concat({node_irreps}, {node_irreps}, {edge_irreps}) ⊗ {sh_irreps}"
        )
        print(f"                   → TP: {irreps_tp_out}")
        print(f"                   → Gate: {self.edge_gate.irreps_out}")
        print(
            f"      Node update: concat(messages {hidden_irreps}, self {node_irreps})"
        )
        print(f"                   → Linear: {irreps_node_tp_out}")
        print(f"                   → Gate: {self.node_gate.irreps_out}")

    def forward(
        self,
        node_feat,
        edge_feat,
        edge_index,
        edge_sh,
        batch_node,
        batch_edge,
        log_to_wandb=False,
    ):
        N = node_feat.shape[0]

        print(
            f"      [MessageBlock.forward] Input: nodes {node_feat.shape} (irreps: {self.node_irreps}), edges {edge_feat.shape} (irreps: {self.edge_irreps})"
        )

        # Edge update: concatenate src node, dst node, and edge features
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        src_node = node_feat[src_idx]
        dst_node = node_feat[dst_idx]

        edge_concat = torch.cat([src_node, dst_node, edge_feat], dim=-1)
        print(
            f"                            Edge concat: src {src_node.shape} + dst {dst_node.shape} + edge {edge_feat.shape} → {edge_concat.shape}"
        )

        # Apply TP with spherical harmonics
        edge_tp = self.edge_update_tp(edge_concat, edge_sh)
        edge_feat_new = self.edge_gate(edge_tp)
        edge_feat_new = self.edge_norm(edge_feat_new, batch_edge)

        print(
            f"                            Edge TP: {edge_concat.shape} ⊗ {edge_sh.shape} → {edge_tp.shape}"
        )
        print(
            f"                            Edge Gate+Norm: {edge_feat_new.shape} (irreps: {self.edge_gate.irreps_out})"
        )
        log_activation_magnitudes(
            edge_feat_new,
            self.edge_gate.irreps_out,
            "Edge activations",
            f"Layer{self.layer_idx+1}_Edge",
            log_to_wandb,
        )

        # Node update: aggregate edge messages + self-connection
        messages = torch.zeros(N, edge_feat_new.shape[1], device=edge_feat_new.device)
        messages.index_add_(0, dst_idx, edge_feat_new)
        print(f"                            Messages aggregated: {messages.shape}")

        # Concatenate with self-connection
        node_concat = torch.cat([messages, node_feat], dim=-1)
        node_linear = self.node_update_lin(node_concat)
        node_feat_new = self.node_gate(node_linear)
        node_feat_new = self.node_norm(node_feat_new, batch_node)

        print(
            f"                            Node Linear: {node_concat.shape} → {node_linear.shape}"
        )
        print(
            f"                            Node Gate+Norm: {node_feat_new.shape} (irreps: {self.node_gate.irreps_out})"
        )
        log_activation_magnitudes(
            node_feat_new,
            self.node_gate.irreps_out,
            "Node activations",
            f"Layer{self.layer_idx+1}_Node",
            log_to_wandb,
        )

        return node_feat_new, edge_feat_new


class MinimalHead(nn.Module):
    """Predict irrep vectors for each edge type."""

    def __init__(self, hidden_irreps, mapper):
        super().__init__()
        self.mapper = mapper

        # One linear projection per edge type
        self.projections = nn.ModuleDict()
        for edge_type in mapper.edge_types:
            pair_irreps = mapper.get_pair_irreps(edge_type)
            self.projections[edge_type] = Linear(hidden_irreps, pair_irreps)
            print(f"    [Head] {edge_type}: {hidden_irreps} → {pair_irreps}")

    def forward(self, edge_feat, edge_type_idx, edge_index, edge_shift):
        """
        Returns: dict[pair_key] → {"vectors": tensor, "edges": tensor}
        """
        outputs = {}

        for type_str, proj in self.projections.items():
            type_idx = self.mapper.edge_type2idx[type_str]
            mask = edge_type_idx == type_idx

            if mask.any():
                selected_feat = edge_feat[mask]
                pred_vectors = proj(selected_feat)

                # Build 5D edge tensor (sx, sy, sz, i, j)
                selected_edges = torch.cat(
                    [
                        edge_shift[:, mask],  # (3, E')
                        edge_index[:, mask],  # (2, E')
                    ],
                    dim=0,
                )  # (5, E')

                outputs[type_str] = {
                    "vectors": pred_vectors,
                    "edges": selected_edges,
                }
                print(
                    f"      [Head.forward] {type_str}: {mask.sum().item()} edges → vectors {pred_vectors.shape}"
                )

        return outputs


class MinimalNetwork(nn.Module):
    """Complete minimal network."""

    def __init__(
        self,
        num_elements,
        n_radial,
        num_edge_types,
        hidden_irreps,
        sh_irreps,
        num_layers,
        mapper,
    ):
        super().__init__()

        # Count scalar irreps correctly
        from e3nn.o3 import Irrep

        scalar_dim = sum(mul for mul, ir in hidden_irreps if ir == Irrep("0e"))

        self.node_enc = MinimalNodeEncoder(num_elements, scalar_dim)
        self.edge_enc = MinimalEdgeEncoder(
            n_radial, num_edge_types, hidden_irreps, sh_irreps
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_layers):
            node_irreps_in = self.node_enc.irreps_out if i == 0 else hidden_irreps
            edge_irreps_in = hidden_irreps
            self.mp_layers.append(
                MinimalMessageBlock(
                    node_irreps_in,
                    edge_irreps_in,
                    hidden_irreps,
                    sh_irreps,
                    layer_idx=i,
                )
            )

        self.head = MinimalHead(hidden_irreps, mapper)

        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        node_type_idx,
        edge_type_idx,
        edge_index,
        edge_shift,
        edge_length_emb,
        edge_sh,
        batch_node,
        batch_edge,
        log_to_wandb=False,
    ):
        print("    [Forward] Starting forward pass...")

        # Encode
        node_feat = self.node_enc(node_type_idx)
        edge_feat = self.edge_enc(
            edge_length_emb, edge_type_idx, edge_sh, batch_edge, log_to_wandb
        )

        # Message passing (both nodes and edges get updated)
        for i, mp_layer in enumerate(self.mp_layers):
            print(f"    [Forward] Message passing layer {i + 1}/{len(self.mp_layers)}")
            node_feat, edge_feat = mp_layer(
                node_feat,
                edge_feat,
                edge_index,
                edge_sh,
                batch_node,
                batch_edge,
                log_to_wandb,
            )

        # Use edge features for head
        head_feat = edge_feat

        # Head
        print(f"    [Forward] Applying head...")
        outputs = self.head(head_feat, edge_type_idx, edge_index, edge_shift)

        return outputs


def compute_mu_H(H_pred, H_gt, S):
    """
    Compute the mu_H correction factor.

    mu_H = sum_{ij} (H_pred_ij - H_gt_ij) * S_ij / sum_{ij} S_ij^2
    """
    numerator = 0.0
    denominator = 0.0

    for key in H_gt.pair_blocks.keys():
        if key in H_pred.pair_blocks and key in S.pair_blocks:
            pred_blocks = H_pred.pair_blocks[key]
            gt_blocks = H_gt.pair_blocks[key]
            s_blocks = S.pair_blocks[key]

            min_n = min(pred_blocks.shape[0], gt_blocks.shape[0], s_blocks.shape[0])

            diff = pred_blocks[:min_n] - gt_blocks[:min_n]
            s_val = s_blocks[:min_n]

            numerator += torch.sum(diff * s_val).item()
            denominator += torch.sum(s_val * s_val).item()

    if denominator > 1e-10:
        return numerator / denominator
    else:
        return 0.0


def compute_detailed_metrics(H_pred, H_gt, S):
    """
    Compute all metrics: MAE, MSE, modified MAE/MSE, mu_H, and correction statistics.

    Returns a dictionary with:
        - mae: Standard mean absolute error
        - mse: Standard mean squared error
        - mae_mod: Modified MAE with mu_H correction
        - mse_mod: Modified MSE with mu_H correction
        - mu_H: The correction factor
        - correction_mae: Mean absolute value of mu_H * S correction
        - correction_mse: Mean squared value of mu_H * S correction
    """
    # Standard MAE and MSE
    mae = 0.0
    mse = 0.0
    total_elements = 0

    for key in H_gt.pair_blocks.keys():
        if key in H_pred.pair_blocks:
            pred_blocks = H_pred.pair_blocks[key]
            gt_blocks = H_gt.pair_blocks[key]
            min_n = min(pred_blocks.shape[0], gt_blocks.shape[0])

            diff = pred_blocks[:min_n] - gt_blocks[:min_n]
            mae += torch.sum(torch.abs(diff)).item()
            mse += torch.sum(diff**2).item()
            total_elements += diff.numel()

    mae /= total_elements
    mse /= total_elements

    # Compute mu_H correction factor
    mu_H = compute_mu_H(H_pred, H_gt, S)

    # Modified MAE/MSE with mu_H correction and correction statistics
    mae_mod = 0.0
    mse_mod = 0.0
    correction_mae = 0.0
    correction_mse = 0.0

    for key in H_gt.pair_blocks.keys():
        if key in H_pred.pair_blocks and key in S.pair_blocks:
            pred_blocks = H_pred.pair_blocks[key]
            gt_blocks = H_gt.pair_blocks[key]
            s_blocks = S.pair_blocks[key]

            min_n = min(pred_blocks.shape[0], gt_blocks.shape[0], s_blocks.shape[0])

            # Correction term
            correction = mu_H * s_blocks[:min_n]

            # Corrected difference
            diff_corrected = pred_blocks[:min_n] - gt_blocks[:min_n] - correction

            mae_mod += torch.sum(torch.abs(diff_corrected)).item()
            mse_mod += torch.sum(diff_corrected**2).item()

            # Statistics of the correction itself
            correction_mae += torch.sum(torch.abs(correction)).item()
            correction_mse += torch.sum(correction**2).item()

    mae_mod /= total_elements
    mse_mod /= total_elements
    correction_mae /= total_elements
    correction_mse /= total_elements

    return {
        "mae": mae,
        "mse": mse,
        "mae_mod": mae_mod,
        "mse_mod": mse_mod,
        "mu_H": mu_H,
        "correction_mae": correction_mae,
        "correction_mse": correction_mse,
    }
