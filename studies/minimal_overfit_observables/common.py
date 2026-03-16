"""
Common classes and functions for minimal E(3)-GNN study.

Shared between training and analysis scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct, TensorSquare
from e3nn.nn import Gate
from torch_scatter import scatter
from torch_geometric.utils import degree
import wandb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io
import imageio


def check_for_nans(tensor, name, input_tensor=None):
    """
    Check if tensor contains NaNs and raise error with diagnostics.

    Args:
        tensor: The tensor to check
        name: Name/location of the operation (e.g., "EdgeEncoder.radial_proj")
        input_tensor: Optional input tensor for additional diagnostics
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total_count = tensor.numel()

        print(f"\n{'='*80}")
        print(f"[ERROR] NaN DETECTED in {name}")
        print(f"{'='*80}")
        print(
            f"NaN count: {nan_count}/{total_count} elements ({100*nan_count/total_count:.2f}%)"
        )
        print(f"Output shape: {tensor.shape}")

        # Stats on non-NaN values
        non_nan_mask = ~torch.isnan(tensor)
        if non_nan_mask.any():
            non_nan_values = tensor[non_nan_mask]
            print(
                f"Non-NaN stats: min={non_nan_values.min().item():.6e}, max={non_nan_values.max().item():.6e}, mean={non_nan_values.mean().item():.6e}"
            )
        else:
            print("All values are NaN!")

        # Input tensor stats if provided
        if input_tensor is not None:
            print(f"\nInput shape: {input_tensor.shape}")
            if torch.isnan(input_tensor).any():
                print(f"[WARN] Input already contains NaNs!")
            else:
                print(
                    f"Input stats: min={input_tensor.min().item():.6e}, max={input_tensor.max().item():.6e}, mean={input_tensor.mean().item():.6e}"
                )

        print(f"{'='*80}\n")
        raise RuntimeError(f"NaN detected in {name}. Training stopped.")


class e3LayerNorm(nn.Module):
    """E(3)-equivariant layer normalization (from DeepH-E3)."""

    def __init__(
        self, irreps_in, eps=1e-5, affine=True, normalization="component", verbose=True
    ):
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

        if verbose:
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

            # Add epsilon before sqrt for numerical stability
            # Use larger epsilon if norm is very small to prevent gradient issues
            safe_norm = torch.sqrt(norm + self.eps)
            field = field / (safe_norm[batch] + self.eps)

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
    features, irreps, name="", wandb_prefix="", log_to_wandb=False, verbose=True
):
    """Log per-irrep activation magnitudes."""
    if not verbose and not (log_to_wandb and wandb_prefix):
        return

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

    if verbose:
        print(f"      [{name}] Magnitudes per irrep: {', '.join(stats)}")


class MinimalNodeEncoder(nn.Module):
    """Encode element type to scalar features."""

    def __init__(self, num_elements, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_elements, hidden_dim)
        self.irreps_out = Irreps(f"{hidden_dim}x0e")
        print(
            f"    [NodeEncoder] Input: {num_elements} elements -> Output: {self.irreps_out}"
        )

    def forward(self, node_type_idx, verbose=True):
        out = self.embedding(node_type_idx)
        check_for_nans(out, "NodeEncoder.embedding")
        if verbose:
            print(
                f"      [NodeEncoder.forward] Input shape: {node_type_idx.shape} -> Output: {out.shape} (irreps: {self.irreps_out})"
            )
        return out


class MinimalEdgeEncoder(nn.Module):
    """Encode edge distance + edge type + spherical harmonics with Gate nonlinearity."""

    def __init__(
        self,
        n_radial,
        num_edge_types,
        hidden_irreps,
        sh_irreps,
        use_sh_tensor_square=False,
        use_e3layernorm=True,
    ):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.use_sh_tensor_square = bool(use_sh_tensor_square)
        self.use_e3layernorm = bool(use_e3layernorm)

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

        if self.use_sh_tensor_square:
            # Keep SH TensorSquare bounded to hidden_irreps to avoid
            # high-l blow-up in the edge encoder preprocessing path.
            sh_ts_out_irreps = Irreps(hidden_irreps)
            self.sh_tensor_square = TensorSquare(sh_irreps, irreps_out=sh_ts_out_irreps)
            tp_irreps_in2 = self.sh_tensor_square.irreps_out
        else:
            self.sh_tensor_square = None
            tp_irreps_in2 = sh_irreps

        # Tensor product: scalars (x) SH -> TP output
        self.tp = FullyConnectedTensorProduct(
            Irreps(f"{scalar_dim}x0e"),
            tp_irreps_in2,
            irreps_tp_out,
            internal_weights=True,
            shared_weights=True,
        )

        # Gate nonlinearity with parity-aware activations (like DeepH-E3)
        # Use silu for even parity (+1), tanh for odd parity (-1)
        act_scalar = {1: F.silu, -1: torch.tanh}
        act_gate = {1: torch.sigmoid, -1: torch.tanh}

        self.gate = Gate(
            irreps_scalars,
            [act_scalar[ir.p] for _, ir in irreps_scalars],
            irreps_gates,
            [act_gate[ir.p] for _, ir in irreps_gates],
            irreps_gated,
        )
        self.irreps_out = self.gate.irreps_out

        # Optional layer norm (verbose=False to avoid duplicate prints)
        if self.use_e3layernorm:
            self.norm = e3LayerNorm(self.irreps_out, verbose=False)
        else:
            self.norm = nn.Identity()

        print(
            f"    [EdgeEncoder] Irreps in: radial({n_radial}) + edge_type({num_edge_types}) -> scalars({scalar_dim}x0e)"
        )
        print(
            f"                  TP: {scalar_dim}x0e (x) {tp_irreps_in2} -> {irreps_tp_out}"
        )
        if self.use_sh_tensor_square:
            print(
                f"                  SH preprocessing: TensorSquare({sh_irreps}) -> {tp_irreps_in2}"
            )
        print(
            f"                  Note: TP creates separate scalar groups for each L, combined by Gate"
        )
        print(f"                  Gate: {irreps_tp_out} -> {self.irreps_out}")
        print(
            "                  Norm: "
            + (
                f"e3LayerNorm({self.irreps_out})"
                if self.use_e3layernorm
                else "disabled"
            )
        )

    def forward(
        self,
        edge_length_emb,
        edge_type_idx,
        edge_sh,
        batch_edge,
        log_to_wandb=False,
        verbose=True,
    ):
        # Create edge type one-hot
        edge_type_onehot = F.one_hot(edge_type_idx, num_classes=self.num_edge_types).to(
            dtype=edge_length_emb.dtype
        )

        # Concatenate radial and edge type features
        combined = torch.cat([edge_length_emb, edge_type_onehot], dim=-1)

        # Project to scalars
        radial_feat = self.radial_proj(combined)  # (E, scalar_dim)
        check_for_nans(radial_feat, "EdgeEncoder.radial_proj", combined)

        if self.sh_tensor_square is not None:
            edge_sh_tp = self.sh_tensor_square(edge_sh)
            check_for_nans(edge_sh_tp, "EdgeEncoder.sh_tensor_square", edge_sh)
        else:
            edge_sh_tp = edge_sh

        # Tensor product with spherical harmonics
        tp_out = self.tp(radial_feat, edge_sh_tp)
        check_for_nans(tp_out, "EdgeEncoder.tp", radial_feat)

        # Gate nonlinearity
        edge_feat = self.gate(tp_out)
        check_for_nans(edge_feat, "EdgeEncoder.gate", tp_out)

        # Optional layer norm
        if self.use_e3layernorm:
            edge_feat = self.norm(edge_feat, batch_edge)
        check_for_nans(edge_feat, "EdgeEncoder.norm", edge_feat)

        if verbose:
            print(
                f"      [EdgeEncoder.forward] Radial: {edge_length_emb.shape}, EdgeType: {edge_type_onehot.shape} -> Combined: {combined.shape}"
            )
            print(
                f"                            -> Scalars: {radial_feat.shape} (irreps: {self.tp.irreps_in1})"
            )
            print(
                f"                            (x) SH features: {edge_sh_tp.shape} (irreps: {self.tp.irreps_in2})"
            )
            print(
                f"                            -> TP out: {tp_out.shape} (irreps: {self.tp.irreps_out})"
            )
            print(
                f"                            -> Gate out: {edge_feat.shape} (irreps: {self.irreps_out})"
            )
        log_activation_magnitudes(
            edge_feat,
            self.irreps_out,
            "EdgeEncoder activations",
            "EdgeEncoder",
            log_to_wandb,
            verbose=verbose,
        )
        return edge_feat


class MinimalMessageBlock(nn.Module):
    """Message passing layer with edge update + node update with Gate and LayerNorm."""

    def __init__(
        self,
        node_irreps,
        edge_irreps,
        hidden_irreps,
        sh_irreps,
        layer_idx=0,
        use_e3layernorm=True,
    ):
        super().__init__()
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps
        self.hidden_irreps = hidden_irreps
        self.layer_idx = layer_idx
        self.use_e3layernorm = bool(use_e3layernorm)

        # Gate setup for both updates
        irreps_scalars = Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l == 0])
        irreps_gated = Irreps([(mul, ir) for mul, ir in hidden_irreps if ir.l > 0])
        irreps_gates = Irreps([(mul, "0e") for mul, _ in irreps_gated])
        irreps_tp_out = irreps_scalars + irreps_gates + irreps_gated

        # Node update: aggregate messages (edge_irreps) + self-connection (node_irreps)
        # Output will be hidden_irreps
        # (No TP needed, just linear projection)

        # Edge update: concat(updated_src_node, updated_dst_node, edge) (x) SH -> TP output
        # After node update, nodes have hidden_irreps
        edge_concat_irreps = hidden_irreps + hidden_irreps + edge_irreps

        self.edge_update_tp = FullyConnectedTensorProduct(
            edge_concat_irreps,
            sh_irreps,
            irreps_tp_out,
            internal_weights=True,
            shared_weights=True,
        )

        # Gate nonlinearity with parity-aware activations
        act_scalar = {1: F.silu, -1: torch.tanh}
        act_gate = {1: torch.sigmoid, -1: torch.tanh}

        self.edge_gate = Gate(
            irreps_scalars,
            [act_scalar[ir.p] for _, ir in irreps_scalars],
            irreps_gates,
            [act_gate[ir.p] for _, ir in irreps_gates],
            irreps_gated,
        )
        if self.use_e3layernorm:
            self.edge_norm = e3LayerNorm(self.edge_gate.irreps_out, verbose=False)
        else:
            self.edge_norm = nn.Identity()

        # Node update: aggregate messages + self-connection -> Linear (not TP)
        irreps_node_tp_out = irreps_scalars + irreps_gates + irreps_gated
        self.node_update_lin = Linear(hidden_irreps + node_irreps, irreps_node_tp_out)

        # Check if Linear can produce all requested output irreps
        input_irreps = Irreps(hidden_irreps) + Irreps(node_irreps)
        output_irreps = Irreps(irreps_node_tp_out)
        for mul_out, ir_out in output_irreps:
            can_produce = any(ir_in == ir_out for _, ir_in in input_irreps)
            if not can_produce:
                print(
                    f"    [WARN] Linear layer cannot produce {ir_out} from input {input_irreps}"
                )

        self.node_gate = Gate(
            irreps_scalars,
            [act_scalar[ir.p] for _, ir in irreps_scalars],
            irreps_gates,
            [act_gate[ir.p] for _, ir in irreps_gates],
            irreps_gated,
        )
        if self.use_e3layernorm:
            self.node_norm = e3LayerNorm(self.node_gate.irreps_out, verbose=False)
        else:
            self.node_norm = nn.Identity()

        print(f"    [MessageBlock] Irreps:")
        print(f"      Node update: concat(messages {edge_irreps}, self {node_irreps})")
        print(f"                   -> Linear: {irreps_node_tp_out}")
        print(f"                   -> Gate: {self.node_gate.irreps_out}")
        print(
            f"      Edge update: concat({hidden_irreps}, {hidden_irreps}, {edge_irreps}) (x) {sh_irreps}"
        )
        print(f"                   -> TP: {irreps_tp_out}")
        print(f"                   -> Gate: {self.edge_gate.irreps_out}")
        print(
            "                   -> Norm: "
            + (
                f"e3LayerNorm({self.edge_gate.irreps_out})"
                if self.use_e3layernorm
                else "disabled"
            )
        )

    def forward(
        self,
        node_feat,
        edge_feat,
        edge_index,
        edge_sh,
        batch_node,
        batch_edge,
        log_to_wandb=False,
        verbose=True,
    ):
        N = node_feat.shape[0]

        if verbose:
            print(
                f"      [MessageBlock.forward] Input: nodes {node_feat.shape} (irreps: {self.node_irreps}), edges {edge_feat.shape} (irreps: {self.edge_irreps})"
            )

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # Node update first (like DeepH-E3): aggregate current edge messages + self-connection
        messages = torch.zeros(N, edge_feat.shape[1], device=edge_feat.device)
        messages.index_add_(0, dst_idx, edge_feat)
        check_for_nans(
            messages, f"MessageBlock[{self.layer_idx}].messages_aggregate", edge_feat
        )
        if verbose:
            print(f"                            Messages aggregated: {messages.shape}")

        # Concatenate with self-connection
        node_concat = torch.cat([messages, node_feat], dim=-1)
        node_linear = self.node_update_lin(node_concat)
        check_for_nans(
            node_linear, f"MessageBlock[{self.layer_idx}].node_linear", node_concat
        )
        node_feat_new = self.node_gate(node_linear)
        check_for_nans(
            node_feat_new, f"MessageBlock[{self.layer_idx}].node_gate", node_linear
        )
        if self.use_e3layernorm:
            node_feat_new = self.node_norm(node_feat_new, batch_node)
        check_for_nans(node_feat_new, f"MessageBlock[{self.layer_idx}].node_norm")

        if verbose:
            print(
                f"                            Node Linear: {node_concat.shape} -> {node_linear.shape}"
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
            verbose=verbose,
        )

        # Edge update second: use updated nodes
        src_node = node_feat_new[src_idx]
        dst_node = node_feat_new[dst_idx]

        edge_concat = torch.cat([src_node, dst_node, edge_feat], dim=-1)
        if verbose:
            print(
                f"                            Edge concat: src {src_node.shape} + dst {dst_node.shape} + edge {edge_feat.shape} -> {edge_concat.shape}"
            )

        # Apply TP with spherical harmonics
        edge_tp = self.edge_update_tp(edge_concat, edge_sh)
        check_for_nans(edge_tp, f"MessageBlock[{self.layer_idx}].edge_tp", edge_concat)
        edge_feat_new = self.edge_gate(edge_tp)
        check_for_nans(
            edge_feat_new, f"MessageBlock[{self.layer_idx}].edge_gate", edge_tp
        )
        if self.use_e3layernorm:
            edge_feat_new = self.edge_norm(edge_feat_new, batch_edge)
        check_for_nans(edge_feat_new, f"MessageBlock[{self.layer_idx}].edge_norm")

        if verbose:
            print(
                f"                            Edge TP: {edge_concat.shape} (x) {edge_sh.shape} -> {edge_tp.shape}"
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
            verbose=verbose,
        )

        return node_feat_new, edge_feat_new


def _build_gate_activation(irreps_hidden):
    irreps_hidden = Irreps(irreps_hidden).simplify()
    irreps_scalars = Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l == 0])
    irreps_gated = Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l > 0])

    if irreps_gated.dim > 0:
        has_even_scalar = any(ir.l == 0 and ir.p == 1 for _, ir in irreps_hidden)
        gate_ir = "0e" if has_even_scalar else "0o"
        irreps_gates = Irreps([(mul, gate_ir) for mul, _ in irreps_gated]).simplify()
    else:
        irreps_gates = Irreps("")

    scalar_acts = [F.silu if ir.p == 1 else torch.tanh for _, ir in irreps_scalars]
    gate_acts = [torch.sigmoid if ir.p == 1 else torch.tanh for _, ir in irreps_gates]

    return Gate(
        irreps_scalars,
        scalar_acts,
        irreps_gates,
        gate_acts,
        irreps_gated,
    )


class E3GateMLP(nn.Module):
    """
    Equivariant MLP made of e3nn Linear layers with Gate nonlinearity between layers.
    """

    def __init__(self, irreps_in, irreps_out, num_layers=1, hidden_irreps=None):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.num_layers = int(num_layers)
        self.hidden_irreps = (
            Irreps(hidden_irreps) if hidden_irreps is not None else self.irreps_in
        )

        self.layers = nn.ModuleList()
        current_irreps = self.irreps_in
        for _ in range(self.num_layers - 1):
            gate = _build_gate_activation(self.hidden_irreps)
            self.layers.append(Linear(current_irreps, gate.irreps_in))
            self.layers.append(gate)
            current_irreps = gate.irreps_out

        self.final_linear = Linear(current_irreps, self.irreps_out)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_linear(x)


class MinimalHead(nn.Module):
    """Predict irrep vectors for each edge type."""

    def __init__(
        self,
        hidden_irreps,
        mapper,
        n_radial,
        head_e3mlp_layers=1,
        magnitude_factorization=False,
        head_mlp_for_scalars=False,
        head_use_tensor_square=False,
        separate_shifted_self=False,
    ):
        super().__init__()
        hidden_irreps = Irreps(hidden_irreps)
        self.mapper = mapper
        if magnitude_factorization:
            raise ValueError(
                "magnitude_factorization is not supported in minimal_overfit_observables."
            )
        if head_mlp_for_scalars:
            raise ValueError(
                "head_mlp_for_scalars is not supported in minimal_overfit_observables."
            )
        self.head_use_tensor_square = bool(head_use_tensor_square)
        self.separate_shifted_self = bool(separate_shifted_self)
        self.n_radial = int(n_radial)
        self.head_e3mlp_layers = int(head_e3mlp_layers)
        if self.head_e3mlp_layers < 1:
            raise ValueError(
                f"head_e3mlp_layers must be >= 1, got {self.head_e3mlp_layers}"
            )
        self.edge_types = list(mapper.edge_types)
        self.need_tensor_square = bool(self.head_use_tensor_square)

        # Separate projections per edge type and diagonal status.
        self.diag_projections = nn.ModuleDict()
        self.shifted_self_projections = nn.ModuleDict()
        self.offdiag_projections = nn.ModuleDict()
        self.output_dims = {}

        if self.need_tensor_square:
            # Keep head TensorSquare bounded to hidden_irreps to avoid
            # unnecessary high-l expansion (e.g., up to l=8 for l_max=4 inputs).
            self.edge_tensor_square = TensorSquare(
                hidden_irreps, irreps_out=hidden_irreps
            )
            ts_out_irreps = self.edge_tensor_square.irreps_out
        else:
            ts_out_irreps = None

        head_input_irreps = (
            ts_out_irreps if self.head_use_tensor_square else hidden_irreps
        )

        for edge_type in self.edge_types:
            pair_irreps = Irreps(mapper.get_pair_irreps(edge_type))
            self.output_dims[edge_type] = pair_irreps.dim
            projection_irreps = pair_irreps
            if projection_irreps.dim > 0:
                self.diag_projections[edge_type] = E3GateMLP(
                    head_input_irreps,
                    projection_irreps,
                    num_layers=self.head_e3mlp_layers,
                    hidden_irreps=head_input_irreps,
                )
                if self.separate_shifted_self:
                    self.shifted_self_projections[edge_type] = E3GateMLP(
                        head_input_irreps,
                        projection_irreps,
                        num_layers=self.head_e3mlp_layers,
                        hidden_irreps=head_input_irreps,
                    )
                self.offdiag_projections[edge_type] = E3GateMLP(
                    head_input_irreps,
                    projection_irreps,
                    num_layers=self.head_e3mlp_layers,
                    hidden_irreps=head_input_irreps,
                )

            if self.separate_shifted_self:
                print(
                    f"    [Head] {edge_type}: "
                    f"diag {head_input_irreps} -> {projection_irreps}, "
                    f"shifted_self {head_input_irreps} -> {projection_irreps}, "
                    f"offdiag {head_input_irreps} -> {projection_irreps}"
                )
            else:
                print(
                    f"    [Head] {edge_type}: "
                    f"diag {head_input_irreps} -> {projection_irreps}, "
                    f"offdiag {head_input_irreps} -> {projection_irreps}"
                )
            print(
                f"            head projections: E3GateMLP layers={self.head_e3mlp_layers}"
            )
            if self.head_use_tensor_square:
                print(
                    f"            head input pre-transform: TensorSquare({hidden_irreps}) -> {ts_out_irreps}"
                )

            # Check if Linear can produce all requested output irreps
            input_irreps = head_input_irreps
            output_irreps = projection_irreps
            missing_irreps = []
            for mul_out, ir_out in output_irreps:
                can_produce = any(ir_in == ir_out for _, ir_in in input_irreps)
                if not can_produce:
                    missing_irreps.append(str(ir_out))
            if missing_irreps:
                print(
                    f"    [WARN] [{edge_type}] Cannot produce irreps {', '.join(set(missing_irreps))} from {hidden_irreps}"
                )

    def forward(
        self,
        edge_feat,
        edge_type_idx,
        edge_index,
        edge_shift,
        edge_length_emb,
        verbose=True,
    ):
        """
        Returns: dict[pair_key] -> {"vectors": tensor, "edges": tensor}
        """
        outputs = {}

        for type_str in self.edge_types:
            type_idx = self.mapper.edge_type2idx[type_str]
            mask = edge_type_idx == type_idx

            if mask.any():
                selected_feat = edge_feat[mask]
                selected_edge_index = edge_index[:, mask]
                selected_edge_shift = edge_shift[:, mask]
                if self.need_tensor_square:
                    ts_feat = self.edge_tensor_square(selected_feat)
                    check_for_nans(
                        ts_feat, f"Head.{type_str}_tensor_square", selected_feat
                    )
                else:
                    ts_feat = None
                head_feat = ts_feat if self.head_use_tensor_square else selected_feat

                # Edge classes:
                # - diag: i == j and zero shift
                # - shifted_self: i == j and non-zero shift (if separate head enabled)
                # - offdiag: all remaining edges
                is_same_atom = selected_edge_index[0] == selected_edge_index[1]
                is_zero_shift = (selected_edge_shift == 0).all(dim=0)
                is_diag = is_same_atom & is_zero_shift
                if self.separate_shifted_self:
                    is_shifted_self = is_same_atom & (~is_zero_shift)
                    is_offdiag = ~is_same_atom
                else:
                    is_shifted_self = torch.zeros_like(is_diag, dtype=torch.bool)
                    is_offdiag = ~is_diag

                pred_vectors = torch.empty(
                    (selected_feat.shape[0], self.output_dims[type_str]),
                    device=selected_feat.device,
                    dtype=selected_feat.dtype,
                )

                if self.output_dims[type_str] == 0:
                    pred_vectors = selected_feat.new_zeros((selected_feat.shape[0], 0))
                else:
                    if is_diag.any():
                        diag_vectors = self.diag_projections[type_str](
                            head_feat[is_diag]
                        )
                        check_for_nans(
                            diag_vectors,
                            f"Head.{type_str}_diag_projection",
                            head_feat[is_diag],
                        )
                        pred_vectors[is_diag] = diag_vectors

                    if is_shifted_self.any():
                        shifted_self_vectors = self.shifted_self_projections[type_str](
                            head_feat[is_shifted_self]
                        )
                        check_for_nans(
                            shifted_self_vectors,
                            f"Head.{type_str}_shifted_self_projection",
                            head_feat[is_shifted_self],
                        )
                        pred_vectors[is_shifted_self] = shifted_self_vectors

                    if is_offdiag.any():
                        offdiag_vectors = self.offdiag_projections[type_str](
                            head_feat[is_offdiag]
                        )
                        check_for_nans(
                            offdiag_vectors,
                            f"Head.{type_str}_offdiag_projection",
                            head_feat[is_offdiag],
                        )
                        pred_vectors[is_offdiag] = offdiag_vectors

                # Build 5D edge tensor (sx, sy, sz, i, j)
                selected_edges = torch.cat(
                    [
                        selected_edge_shift,  # (3, E')
                        selected_edge_index,  # (2, E')
                    ],
                    dim=0,
                )  # (5, E')

                outputs[type_str] = {
                    "vectors": pred_vectors,
                    "edges": selected_edges,
                }
                if self.separate_shifted_self:
                    if verbose:
                        print(
                            f"      [Head.forward] {type_str}: {mask.sum().item()} edges "
                            f"({int(is_diag.sum().item())} diag, "
                            f"{int(is_shifted_self.sum().item())} shifted_self, "
                            f"{int(is_offdiag.sum().item())} offdiag) "
                            f"-> vectors {pred_vectors.shape}"
                        )
                else:
                    if verbose:
                        print(
                            f"      [Head.forward] {type_str}: {mask.sum().item()} edges "
                            f"({int(is_diag.sum().item())} diag, {int(is_offdiag.sum().item())} offdiag) "
                            f"-> vectors {pred_vectors.shape}"
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
        matrix_targets=None,
        head_e3mlp_layers=1,
        edge_encoder_use_sh_tensor_square=False,
        magnitude_factorization=False,
        head_mlp_for_scalars=False,
        head_use_tensor_square=False,
        head_use_node_embeddings_for_self_edges=False,
        separate_shifted_self=False,
        use_e3layernorm=True,
        verbose=True,
    ):
        super().__init__()
        if magnitude_factorization:
            raise ValueError(
                "magnitude_factorization is not supported in minimal_overfit_observables."
            )
        if head_mlp_for_scalars:
            raise ValueError(
                "head_mlp_for_scalars is not supported in minimal_overfit_observables."
            )
        if head_use_node_embeddings_for_self_edges and num_layers < 1:
            raise ValueError(
                "head_use_node_embeddings_for_self_edges requires num_layers >= 1 "
                "so node and edge embeddings share hidden_irreps."
            )
        self.verbose = verbose
        self.matrix_targets = (
            list(matrix_targets) if matrix_targets is not None else ["hamiltonian"]
        )
        if len(self.matrix_targets) == 0:
            raise ValueError("matrix_targets cannot be empty")
        self.multi_target_mode = matrix_targets is not None
        self.head_use_node_embeddings_for_self_edges = bool(
            head_use_node_embeddings_for_self_edges
        )

        # Count scalar irreps correctly
        from e3nn.o3 import Irrep

        scalar_dim = sum(mul for mul, ir in hidden_irreps if ir == Irrep("0e"))
        self.node_enc = MinimalNodeEncoder(num_elements, scalar_dim)
        self.edge_enc = MinimalEdgeEncoder(
            n_radial,
            num_edge_types,
            hidden_irreps,
            sh_irreps,
            use_sh_tensor_square=edge_encoder_use_sh_tensor_square,
            use_e3layernorm=use_e3layernorm,
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
                    use_e3layernorm=use_e3layernorm,
                )
            )

        self.heads = nn.ModuleDict(
            {
                matrix_name: MinimalHead(
                    hidden_irreps,
                    mapper,
                    n_radial=n_radial,
                    head_e3mlp_layers=head_e3mlp_layers,
                    magnitude_factorization=magnitude_factorization,
                    head_mlp_for_scalars=head_mlp_for_scalars,
                    head_use_tensor_square=head_use_tensor_square,
                    separate_shifted_self=separate_shifted_self,
                )
                for matrix_name in self.matrix_targets
            }
        )

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
        verbose=None,
    ):
        if verbose is None:
            verbose = self.verbose

        if verbose:
            print("    [Forward] Starting forward pass...")

        # Encode
        node_feat = self.node_enc(node_type_idx, verbose=verbose)
        edge_feat = self.edge_enc(
            edge_length_emb,
            edge_type_idx,
            edge_sh,
            batch_edge,
            log_to_wandb,
            verbose=verbose,
        )

        # Message passing (both nodes and edges get updated)
        for i, mp_layer in enumerate(self.mp_layers):
            if verbose:
                print(
                    f"    [Forward] Message passing layer {i + 1}/{len(self.mp_layers)}"
                )
            node_feat, edge_feat = mp_layer(
                node_feat,
                edge_feat,
                edge_index,
                edge_sh,
                batch_node,
                batch_edge,
                log_to_wandb,
                verbose=verbose,
            )

        # Optionally mirror the main model's onsite path by swapping in node
        # embeddings for zero-shift self-edges while keeping edge embeddings
        # for shifted-self and off-diagonal edges.
        head_feat = edge_feat
        if self.head_use_node_embeddings_for_self_edges:
            is_self_edge = (edge_index[0] == edge_index[1]) & (edge_shift == 0).all(
                dim=0
            )
            if is_self_edge.any():
                head_feat = edge_feat.clone()
                head_feat[is_self_edge] = node_feat[edge_index[0, is_self_edge]]

        # Head
        if verbose:
            print(f"    [Forward] Applying head...")
        outputs_by_target = {
            matrix_name: head(
                head_feat,
                edge_type_idx,
                edge_index,
                edge_shift,
                edge_length_emb,
                verbose=verbose,
            )
            for matrix_name, head in self.heads.items()
        }

        if self.multi_target_mode:
            return outputs_by_target
        return outputs_by_target[self.matrix_targets[0]]


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


def compute_basic_matrix_metrics(M_pred, M_gt):
    """
    Compute basic MAE/MSE for a generic block matrix pair.
    """
    mae = 0.0
    mse = 0.0
    total_elements = 0

    for key in M_gt.pair_blocks.keys():
        if key in M_pred.pair_blocks:
            pred_blocks = M_pred.pair_blocks[key]
            gt_blocks = M_gt.pair_blocks[key]
            min_n = min(pred_blocks.shape[0], gt_blocks.shape[0])
            if min_n <= 0:
                continue

            diff = pred_blocks[:min_n] - gt_blocks[:min_n]
            mae += torch.sum(torch.abs(diff)).item()
            mse += torch.sum(diff**2).item()
            total_elements += diff.numel()

    if total_elements <= 0:
        return {"mae": 0.0, "mse": 0.0}

    return {
        "mae": mae / total_elements,
        "mse": mse / total_elements,
    }


def compute_distance_error_curve(
    H_pred,
    H_gt,
    positions: torch.Tensor,
    box: torch.Tensor | None = None,
    partial_train=None,
    n_bins: int = 16,
):
    """
    Compute distance-binned Hamiltonian error statistics.

    For each bin, returns:
      - absolute L1 and L2 (RMSE)
      - relative L1 and L2 (normalized by GT magnitudes in the same bin)
      - edge and element counts
    """
    filtered_pred = filter_blocks_by_partial_train(H_pred, partial_train)
    filtered_gt = filter_blocks_by_partial_train(H_gt, partial_train)

    dists = []
    abs_l1_sum = []
    abs_l2_sum = []
    gt_l1_sum = []
    gt_l2_sum = []
    elem_count = []
    edge_count = []
    edge_l1_abs = []
    edge_l2_abs = []
    edge_l1_rel = []
    edge_l2_rel = []

    if positions is None:
        raise ValueError("positions must be provided for distance-binned analysis")

    for key in H_gt.pair_blocks.keys():
        if key not in H_pred.pair_blocks:
            continue

        pred_blocks_full = H_pred.pair_blocks[key]
        gt_blocks_full = H_gt.pair_blocks[key]
        gt_edges_full = H_gt.pair_edges[key]  # (5, E)

        _, pred_mask = filtered_pred[key]
        _, gt_mask = filtered_gt[key]

        # robust alignment by 5D edge key
        pred_edge_to_idx = {
            tuple(map(int, edge.tolist())): idx
            for idx, edge in enumerate(H_pred.pair_edges[key].t())
        }

        for gt_idx, edge in enumerate(gt_edges_full.t()):
            if not bool(gt_mask[gt_idx]):
                continue
            edge_key = tuple(map(int, edge.tolist()))
            pred_idx = pred_edge_to_idx.get(edge_key, None)
            if pred_idx is None:
                continue
            if (
                pred_idx >= pred_blocks_full.shape[0]
                or gt_idx >= gt_blocks_full.shape[0]
            ):
                continue
            if pred_idx < pred_mask.shape[0] and not bool(pred_mask[pred_idx]):
                continue

            pred_block = pred_blocks_full[pred_idx]
            gt_block = gt_blocks_full[gt_idx]
            diff = pred_block - gt_block

            sx, sy, sz, i, j = edge_key
            shift = torch.tensor(
                [sx, sy, sz], dtype=positions.dtype, device=positions.device
            )
            if box is not None:
                disp = positions[j] - positions[i] + shift @ box
            else:
                disp = positions[j] - positions[i]
            dist = torch.linalg.norm(disp).item()

            dists.append(dist)
            abs_l1_sum.append(torch.abs(diff).sum().item())
            abs_l2_sum.append((diff**2).sum().item())
            gt_l1_sum.append(torch.abs(gt_block).sum().item())
            gt_l2_sum.append((gt_block**2).sum().item())
            elem_count.append(diff.numel())
            edge_count.append(1)

            abs_sum = torch.abs(diff).sum().item()
            sq_sum = (diff**2).sum().item()
            gt_abs_sum = torch.abs(gt_block).sum().item()
            gt_sq_sum = (gt_block**2).sum().item()
            n_el = diff.numel()

            edge_l1_abs.append(abs_sum / max(n_el, 1))
            edge_l2_abs.append((sq_sum / max(n_el, 1)) ** 0.5)
            edge_l1_rel.append(abs_sum / max(gt_abs_sum, 1e-14))
            edge_l2_rel.append((sq_sum / max(gt_sq_sum, 1e-14)) ** 0.5)

    if len(dists) == 0:
        return None

    dists_t = torch.tensor(dists, dtype=torch.float64)
    d_min = float(dists_t.min().item())
    d_max = float(dists_t.max().item())
    if abs(d_max - d_min) < 1e-12:
        d_max = d_min + 1e-6

    bin_edges = torch.linspace(d_min, d_max, n_bins + 1, dtype=torch.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = torch.bucketize(dists_t, bin_edges[1:], right=False).clamp(max=n_bins - 1)

    l1_abs = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_abs = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_rel = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_rel = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_abs_min = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_abs_max = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_abs_min = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_abs_max = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_rel_min = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_rel_max = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_rel_min = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_rel_max = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    n_edges_per_bin = torch.zeros((n_bins,), dtype=torch.int64)
    n_elems_per_bin = torch.zeros((n_bins,), dtype=torch.int64)

    abs_l1_sum_t = torch.tensor(abs_l1_sum, dtype=torch.float64)
    abs_l2_sum_t = torch.tensor(abs_l2_sum, dtype=torch.float64)
    gt_l1_sum_t = torch.tensor(gt_l1_sum, dtype=torch.float64)
    gt_l2_sum_t = torch.tensor(gt_l2_sum, dtype=torch.float64)
    elem_count_t = torch.tensor(elem_count, dtype=torch.int64)
    edge_count_t = torch.tensor(edge_count, dtype=torch.int64)
    edge_l1_abs_t = torch.tensor(edge_l1_abs, dtype=torch.float64)
    edge_l2_abs_t = torch.tensor(edge_l2_abs, dtype=torch.float64)
    edge_l1_rel_t = torch.tensor(edge_l1_rel, dtype=torch.float64)
    edge_l2_rel_t = torch.tensor(edge_l2_rel, dtype=torch.float64)

    for b in range(n_bins):
        mask = bin_idx == b
        if not torch.any(mask):
            continue

        sum_abs_l1 = abs_l1_sum_t[mask].sum()
        sum_abs_l2 = abs_l2_sum_t[mask].sum()
        sum_gt_l1 = gt_l1_sum_t[mask].sum()
        sum_gt_l2 = gt_l2_sum_t[mask].sum()
        sum_elems = elem_count_t[mask].sum()
        sum_edges = edge_count_t[mask].sum()

        n_edges_per_bin[b] = sum_edges
        n_elems_per_bin[b] = sum_elems

        if sum_elems > 0:
            l1_abs[b] = sum_abs_l1 / sum_elems
            l2_abs[b] = torch.sqrt(sum_abs_l2 / sum_elems)
        if sum_gt_l1 > 1e-14:
            l1_rel[b] = sum_abs_l1 / sum_gt_l1
        if sum_gt_l2 > 1e-14:
            l2_rel[b] = torch.sqrt(sum_abs_l2 / sum_gt_l2)

        l1_abs_min[b] = edge_l1_abs_t[mask].min()
        l1_abs_max[b] = edge_l1_abs_t[mask].max()
        l2_abs_min[b] = edge_l2_abs_t[mask].min()
        l2_abs_max[b] = edge_l2_abs_t[mask].max()
        l1_rel_min[b] = edge_l1_rel_t[mask].min()
        l1_rel_max[b] = edge_l1_rel_t[mask].max()
        l2_rel_min[b] = edge_l2_rel_t[mask].min()
        l2_rel_max[b] = edge_l2_rel_t[mask].max()

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "l1_abs": l1_abs.tolist(),
        "l2_abs": l2_abs.tolist(),
        "l1_rel": l1_rel.tolist(),
        "l2_rel": l2_rel.tolist(),
        "l1_abs_min": l1_abs_min.tolist(),
        "l1_abs_max": l1_abs_max.tolist(),
        "l2_abs_min": l2_abs_min.tolist(),
        "l2_abs_max": l2_abs_max.tolist(),
        "l1_rel_min": l1_rel_min.tolist(),
        "l1_rel_max": l1_rel_max.tolist(),
        "l2_rel_min": l2_rel_min.tolist(),
        "l2_rel_max": l2_rel_max.tolist(),
        "n_edges": n_edges_per_bin.tolist(),
        "n_elements": n_elems_per_bin.tolist(),
        "d_min": d_min,
        "d_max": d_max,
        "n_bins": n_bins,
    }


def save_distance_error_curve_plot(
    curve_data,
    output_path: Path | str,
    title: str = None,
    relative_log_scale: bool = True,
):
    """
    Save a 2x2 plot of distance-binned error curves:
      abs L1, abs L2, rel L1, rel L2.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _interp_nans(x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        """Linearly interpolate NaNs so empty bins do not break plotted lines."""
        y = y_vals.astype(float).copy()
        valid = ~np.isnan(y)
        if valid.sum() == 0:
            return y
        if valid.sum() == 1:
            y[~valid] = y[valid][0]
            return y
        y[~valid] = np.interp(x_vals[~valid], x_vals[valid], y[valid])
        return y

    def _interp_nans_log(
        x_vals: np.ndarray, y_vals: np.ndarray, eps: float = 1e-16
    ) -> np.ndarray:
        """
        Interpolate NaNs in log-space (geometric interpolation).
        Only strictly positive finite values are used as anchors.
        """
        y = y_vals.astype(float).copy()
        valid = np.isfinite(y) & (y > 0.0)
        if valid.sum() == 0:
            return y
        if valid.sum() == 1:
            y[~np.isfinite(y)] = y[valid][0]
            return y

        y_log = np.log10(y[valid])
        missing = ~np.isfinite(y)
        y_interp_log = np.interp(x_vals[missing], x_vals[valid], y_log)
        y[missing] = np.power(10.0, y_interp_log)
        y[(~np.isfinite(y)) | (y <= 0.0)] = eps
        return y

    def _make_log_safe(y_vals: np.ndarray, eps: float = 1e-16) -> np.ndarray:
        """
        Ensure values are strictly positive for log-scale plotting.
        NaNs are preserved.
        """
        y = y_vals.astype(float).copy()
        finite = np.isfinite(y)
        nonpos = finite & (y <= 0.0)
        y[nonpos] = eps
        return y

    x = np.array(curve_data["bin_centers"], dtype=float)
    l1_abs = np.array(curve_data["l1_abs"], dtype=float)
    l2_abs = np.array(curve_data["l2_abs"], dtype=float)
    l1_rel = np.array(curve_data["l1_rel"], dtype=float)
    l2_rel = np.array(curve_data["l2_rel"], dtype=float)
    l1_abs_min = np.array(curve_data["l1_abs_min"], dtype=float)
    l1_abs_max = np.array(curve_data["l1_abs_max"], dtype=float)
    l2_abs_min = np.array(curve_data["l2_abs_min"], dtype=float)
    l2_abs_max = np.array(curve_data["l2_abs_max"], dtype=float)
    l1_rel_min = np.array(curve_data["l1_rel_min"], dtype=float)
    l1_rel_max = np.array(curve_data["l1_rel_max"], dtype=float)
    l2_rel_min = np.array(curve_data["l2_rel_min"], dtype=float)
    l2_rel_max = np.array(curve_data["l2_rel_max"], dtype=float)

    # Interpolate empty-bin NaNs for plotting only (raw metrics stay unchanged).
    l1_abs_i = _interp_nans(x, l1_abs)
    l2_abs_i = _interp_nans(x, l2_abs)
    l1_rel_i = _interp_nans_log(x, l1_rel)
    l2_rel_i = _interp_nans_log(x, l2_rel)
    l1_abs_min_i = _interp_nans(x, l1_abs_min)
    l1_abs_max_i = _interp_nans(x, l1_abs_max)
    l2_abs_min_i = _interp_nans(x, l2_abs_min)
    l2_abs_max_i = _interp_nans(x, l2_abs_max)
    l1_rel_min_i = _interp_nans_log(x, l1_rel_min)
    l1_rel_max_i = _interp_nans_log(x, l1_rel_max)
    l2_rel_min_i = _interp_nans_log(x, l2_rel_min)
    l2_rel_max_i = _interp_nans_log(x, l2_rel_max)
    if relative_log_scale:
        l1_rel_i = _make_log_safe(l1_rel_i)
        l2_rel_i = _make_log_safe(l2_rel_i)
        l1_rel_min_i = _make_log_safe(l1_rel_min_i)
        l1_rel_max_i = _make_log_safe(l1_rel_max_i)
        l2_rel_min_i = _make_log_safe(l2_rel_min_i)
        l2_rel_max_i = _make_log_safe(l2_rel_max_i)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        (
            title
            if title is not None
            else f"Distance Error Curves ({curve_data['n_bins']} bins)"
        ),
        fontsize=14,
    )

    panels = [
        (axes[0, 0], l1_abs_i, l1_abs_min_i, l1_abs_max_i, "Absolute L1"),
        (axes[0, 1], l2_abs_i, l2_abs_min_i, l2_abs_max_i, "Absolute L2 (RMSE)"),
        (axes[1, 0], l1_rel_i, l1_rel_min_i, l1_rel_max_i, "Relative L1"),
        (axes[1, 1], l2_rel_i, l2_rel_min_i, l2_rel_max_i, "Relative L2"),
    ]
    for ax, y, y_min, y_max, ttl in panels:
        ax.plot(x, y, marker="o", linewidth=1.2, markersize=2.2, label="mean")
        ax.plot(x, y_min, linewidth=0.9, linestyle="--", alpha=0.8, label="min")
        ax.plot(x, y_max, linewidth=0.9, linestyle="--", alpha=0.8, label="max")
        ax.fill_between(x, y_min, y_max, alpha=0.15)
        ax.set_title(ttl)
        ax.set_xlabel("Distance (A)")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    if relative_log_scale:
        axes[1, 0].set_yscale("log")
        axes[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# EIGENVALUE / DOS UTILITIES
# =============================================================================


def compute_generalized_eigenvalues(H, S):
    """
    Compute generalized eigenvalues for H x = lambda S x using Cholesky reduction.
    """
    H_dense = H.to_dense().detach().to(torch.float64)
    S_dense = S.to_dense().detach().to(torch.float64)

    # Symmetrize for numerical robustness.
    H_dense = 0.5 * (H_dense + H_dense.T)
    S_dense = 0.5 * (S_dense + S_dense.T)

    L = torch.linalg.cholesky(S_dense)
    tmp = torch.linalg.solve(L, H_dense)
    A = torch.linalg.solve(L, tmp.T).T  # A = L^{-1} H L^{-T}
    A = 0.5 * (A + A.T)
    return torch.linalg.eigvalsh(A)


def compute_dos_from_eigenvalues(
    eigenvalues,
    sigma=0.2,
    bin_width=0.1,
    e_min=None,
    e_max=None,
):
    """
    Compute DOS from eigenvalues via Gaussian broadening.
    """
    if e_min is None or e_max is None:
        eig_min = float(torch.min(eigenvalues).item())
        eig_max = float(torch.max(eigenvalues).item())
        span = max(eig_max - eig_min, 1e-6)
        margin = 0.1 * span + 0.05
        e_min = eig_min - margin
        e_max = eig_max + margin

    grid = torch.arange(
        e_min,
        e_max + bin_width,
        bin_width,
        dtype=eigenvalues.dtype,
        device=eigenvalues.device,
    )
    dos = torch.sum(
        torch.exp(-((grid[:, None] - eigenvalues[None, :]) ** 2) / (2 * sigma**2)),
        dim=1,
    ) / (
        torch.sqrt(
            torch.tensor(2 * torch.pi, dtype=eigenvalues.dtype, device=grid.device)
        )
        * sigma
    )
    return grid, dos


def save_dos_comparison_plot(
    H_pred,
    H_gt,
    S,
    output_path: Path | str,
    *,
    sigma: float = 0.2,
    bin_width: float = 0.1,
    title: str = "DOS Comparison",
):
    """
    Compute eigen/DOS metrics and save a DOS comparison plot.

    Returns:
        Dict with eigen and DOS summary metrics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eig_pred = compute_generalized_eigenvalues(H_pred, S)
    eig_gt = compute_generalized_eigenvalues(H_gt, S)

    abs_err = torch.abs(eig_pred - eig_gt)
    rel_err = abs_err / (torch.abs(eig_gt) + 1e-12)

    eig_min = float(torch.min(torch.min(eig_pred), torch.min(eig_gt)).item())
    eig_max = float(torch.max(torch.max(eig_pred), torch.max(eig_gt)).item())
    span = max(eig_max - eig_min, 1e-6)
    margin = 0.1 * span + 0.05
    e_min = eig_min - margin
    e_max = eig_max + margin

    grid, dos_pred = compute_dos_from_eigenvalues(
        eig_pred,
        sigma=sigma,
        bin_width=bin_width,
        e_min=e_min,
        e_max=e_max,
    )
    _, dos_gt = compute_dos_from_eigenvalues(
        eig_gt,
        sigma=sigma,
        bin_width=bin_width,
        e_min=e_min,
        e_max=e_max,
    )

    dos_diff = dos_pred - dos_gt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = grid.detach().cpu().numpy()
    ax.plot(x, dos_gt.detach().cpu().numpy(), label="DOS GT", linewidth=2.0)
    ax.plot(
        x,
        dos_pred.detach().cpu().numpy(),
        label="DOS Pred",
        linewidth=2.0,
        linestyle="--",
    )
    ax.set_xlabel("Energy")
    ax.set_ylabel("DOS")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "eig_abs_mean": float(abs_err.mean().item()),
        "eig_abs_max": float(abs_err.max().item()),
        "eig_rel_mean": float(rel_err.mean().item()),
        "eig_rel_max": float(rel_err.max().item()),
        "dos_mae": float(torch.mean(torch.abs(dos_diff)).item()),
        "dos_mse": float(torch.mean(dos_diff**2).item()),
        "dos_max_abs": float(torch.max(torch.abs(dos_diff)).item()),
        "dos_grid_min": float(grid[0].item()),
        "dos_grid_max": float(grid[-1].item()),
        "dos_grid_points": int(grid.shape[0]),
        "dos_plot_path": str(output_path),
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def get_diagonal_mask(edges_5d):
    """
    Get mask for diagonal blocks (self-interactions: sx=sy=sz=0, i=j).

    Args:
        edges_5d: (5, num_edges) tensor with [sx, sy, sz, i, j]

    Returns:
        Boolean mask of shape (num_edges,) with True for diagonal blocks
    """
    sx, sy, sz, i, j = edges_5d[0], edges_5d[1], edges_5d[2], edges_5d[3], edges_5d[4]
    is_self_interaction = (sx == 0) & (sy == 0) & (sz == 0)
    is_same_atom = i == j
    return is_self_interaction & is_same_atom


def canonicalize_edge_order(
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    positions: torch.Tensor,
    box: torch.Tensor | None = None,
):
    """
    Canonicalize edge order to match the main pipeline convention:
    1) all self-edges first, sorted by source index
    2) all off-diagonal edges sorted by (distance, sx, sy, sz, src, dst)

    Args:
        edge_index: (2, E) with [src, dst]
        edge_shift: (3, E) with [sx, sy, sz]
        positions: (N, 3)
        box: (3, 3) or None

    Returns:
        edge_index_sorted, edge_shift_sorted, permutation_indices
    """
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_shift.shape[0] != 3:
        raise ValueError("edge_shift must have shape (3, E)")
    if edge_index.shape[1] != edge_shift.shape[1]:
        raise ValueError(
            "edge_index and edge_shift must contain the same number of edges"
        )

    src = edge_index[0]
    dst = edge_index[1]
    sx = edge_shift[0]
    sy = edge_shift[1]
    sz = edge_shift[2]
    indices = torch.arange(edge_index.shape[1], device=edge_index.device)

    is_self = (src == dst) & (sx == 0) & (sy == 0) & (sz == 0)
    diag_indices = indices[is_self]
    offdiag_indices = indices[~is_self]

    # Self-edges first, sorted by atom index for deterministic ordering.
    diag_src = src[diag_indices]
    perm_diag = torch.argsort(diag_src)
    sorted_diag_indices = diag_indices[perm_diag]

    # Off-diagonal edges sorted by distance, then by shift/src/dst tie-breakers.
    if offdiag_indices.numel() > 0:
        shift_float = edge_shift[:, offdiag_indices].T.to(
            dtype=positions.dtype, device=positions.device
        )
        src_od = src[offdiag_indices]
        dst_od = dst[offdiag_indices]

        if box is not None:
            disp = (
                positions[dst_od]
                - positions[src_od]
                + shift_float @ box.to(positions.device)
            )
        else:
            disp = positions[dst_od] - positions[src_od]
        distances = torch.linalg.norm(disp, dim=-1)

        d_cpu = distances.cpu().tolist()
        sx_cpu = sx[offdiag_indices].cpu().tolist()
        sy_cpu = sy[offdiag_indices].cpu().tolist()
        sz_cpu = sz[offdiag_indices].cpu().tolist()
        src_cpu = src_od.cpu().tolist()
        dst_cpu = dst_od.cpu().tolist()
        idx_cpu = offdiag_indices.cpu().tolist()

        sortable = []
        for i in range(len(idx_cpu)):
            sortable.append(
                (
                    d_cpu[i],
                    sx_cpu[i],
                    sy_cpu[i],
                    sz_cpu[i],
                    src_cpu[i],
                    dst_cpu[i],
                    idx_cpu[i],
                )
            )
        sortable.sort()
        sorted_offdiag_indices = torch.tensor(
            [row[-1] for row in sortable],
            dtype=torch.long,
            device=edge_index.device,
        )
    else:
        sorted_offdiag_indices = offdiag_indices

    final_indices = torch.cat([sorted_diag_indices, sorted_offdiag_indices], dim=0)
    return edge_index[:, final_indices], edge_shift[:, final_indices], final_indices


def filter_blocks_by_partial_train(
    block_matrix, partial_train, separate_shifted_self: bool = False
):
    """
    Filter blocks based on partial_train setting.

    Args:
        block_matrix: BlockMatrix object
        partial_train: "diag", "offdiag", "shifted_self", or None
        separate_shifted_self: retained for compatibility; not used.

    Returns:
        Dictionary mapping edge_type -> (filtered_blocks, mask)
    """
    if partial_train is None:
        # Return all blocks with full mask
        return {
            key: (
                blocks,
                torch.ones(blocks.shape[0], dtype=torch.bool, device=blocks.device),
            )
            for key, blocks in block_matrix.pair_blocks.items()
        }

    filtered_data = {}
    for key in block_matrix.pair_blocks.keys():
        blocks = block_matrix.pair_blocks[key]
        edges = block_matrix.pair_edges[key]  # (5, num_edges)

        diag_mask = get_diagonal_mask(edges)

        sx, sy, sz, i, j = edges[0], edges[1], edges[2], edges[3], edges[4]
        shifted_self_mask = (i == j) & ((sx != 0) | (sy != 0) | (sz != 0))

        if partial_train == "diag":
            mask = diag_mask
        elif partial_train == "shifted_self":
            mask = shifted_self_mask
        elif partial_train == "offdiag":
            mask = i != j
        else:
            mask = torch.ones(blocks.shape[0], dtype=torch.bool, device=blocks.device)

        filtered_data[key] = (blocks, mask)

    return filtered_data


def extract_partial_hamiltonian(
    H, S, atoms_list, orbital_cfg, sx, sy, sz, partial_train=None
):
    """
    Extract a partial Hamiltonian matrix for a specific [sx, sy, sz] shift.

    Args:
        H: Hamiltonian BlockMatrix
        S: Overlap BlockMatrix
        atoms_list: List of atom symbols
        orbital_cfg: Orbital configuration
        sx, sy, sz: Cell shift indices
        partial_train: "diag", "offdiag", or None - filters which blocks to extract

    Returns:
        (H_dense, S_dense, total_dim) where H_dense and S_dense are numpy arrays
    """

    # Compute total matrix size
    orbital_dims = [orbital_cfg.element_to_irreps[atom].dim for atom in atoms_list]
    total_dim = sum(orbital_dims)

    # Create dense matrices
    H_dense = np.zeros((total_dim, total_dim))
    S_dense = np.zeros((total_dim, total_dim)) if S is not None else None

    # Build index map: atom_idx -> (start_row, end_row)
    atom_ranges = []
    current_idx = 0
    for dim in orbital_dims:
        atom_ranges.append((current_idx, current_idx + dim))
        current_idx += dim

    # Get diagonal mask for filtering if needed
    filtered_H = filter_blocks_by_partial_train(H, partial_train)
    filtered_S = (
        filter_blocks_by_partial_train(S, partial_train) if S is not None else None
    )

    # Fill in blocks for this specific shift
    for key in H.pair_blocks.keys():
        H_blocks = H.pair_blocks[key]
        H_edges = H.pair_edges[key]
        _, h_mask = filtered_H[key]

        if S is not None and key in S.pair_blocks:
            S_blocks = S.pair_blocks[key]
            _, s_mask = filtered_S[key]
        else:
            S_blocks = None
            s_mask = None

        # Find edges with the specified shift
        for idx in range(H_edges.shape[1]):
            # Skip if filtered out by partial_train
            if not h_mask[idx]:
                continue

            shift_x, shift_y, shift_z, i, j = H_edges[:, idx].tolist()

            if int(shift_x) == sx and int(shift_y) == sy and int(shift_z) == sz:
                # Get block
                block_H = H_blocks[idx].detach().cpu().numpy()

                # Get atom ranges
                i_start, i_end = atom_ranges[int(i)]
                j_start, j_end = atom_ranges[int(j)]

                # Fill in matrix
                H_dense[i_start:i_end, j_start:j_end] = block_H

                # Fill overlap if available
                if S_blocks is not None and (s_mask is None or s_mask[idx]):
                    block_S = S_blocks[idx].detach().cpu().numpy()
                    S_dense[i_start:i_end, j_start:j_end] = block_S

    return H_dense, S_dense, total_dim


def visualize_hamiltonians(
    H_pred,
    H_gt,
    S,
    atoms_list,
    orbital_cfg,
    k_range,
    output_dir,
    dynamic_range=False,
    diff_dynamic_range=False,
    per_panel_dynamic_range=True,
    partial_train=None,
    filename_prefix="hamiltonian",
    percentile=80.0,
    matrix_label: str = "H",
):
    """
    Visualize Hamiltonians for all [sx, sy, sz] combinations in [-k, k]^3.

    For each shift, show:
    - Ground truth H
    - Predicted H
    - Difference (pred - gt)
    - Difference with mu_H correction

    Args:
        H_pred: Predicted Hamiltonian BlockMatrix
        H_gt: Ground truth Hamiltonian BlockMatrix
        S: Overlap BlockMatrix
        atoms_list: List of atom symbols
        orbital_cfg: Orbital configuration
        k_range: Range for [sx, sy, sz] visualization: [-k, k]
        output_dir: Directory to save plots
        dynamic_range: If True, use percentile of abs(H_gt) for color range.
                      If False, use fixed [-1, 1] range.
        diff_dynamic_range: If True, scale diff plots by percentile of abs(diff)
                            and abs(diff_corrected). Defaults to False (use
                            same range as ground truth/prediction).
        per_panel_dynamic_range: If True and dynamic_range is enabled, use one
                                 shared dynamic range for (gt, pred) and a second
                                 shared range for (diff, diff_corrected). Defaults
                                 to False (old behavior).
        partial_train: "diag", "offdiag", or None - filters which blocks to visualize
        filename_prefix: Prefix for the output filename (e.g., "hamiltonian" or "hamiltonian_0e")
        percentile: Percentile value for dynamic color range (default: 80.0)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute optional gauge correction (Hamiltonian-only).
    if S is not None:
        mu_H = compute_mu_H(H_pred, H_gt, S)
    else:
        mu_H = 0.0

    print(f"\nVisualizing Hamiltonians for shifts in [{-k_range}, {k_range}]^3")
    print(f"mu_H correction factor: {mu_H:.6e}")
    if partial_train is not None:
        print(
            f"Partial training mode: {partial_train} (showing {partial_train} blocks only)"
        )
    print()

    # Iterate over all shifts
    shifts_to_plot = []
    for sx in range(-k_range, k_range + 1):
        for sy in range(-k_range, k_range + 1):
            for sz in range(-k_range, k_range + 1):
                shifts_to_plot.append((sx, sy, sz))

    for sx, sy, sz in shifts_to_plot:
        print(f"Processing shift [{sx}, {sy}, {sz}]...")

        # Extract partial Hamiltonians
        H_gt_dense, S_dense, dim = extract_partial_hamiltonian(
            H_gt, S, atoms_list, orbital_cfg, sx, sy, sz, partial_train=partial_train
        )
        H_pred_dense, _, _ = extract_partial_hamiltonian(
            H_pred,
            None,
            atoms_list,
            orbital_cfg,
            sx,
            sy,
            sz,
            partial_train=partial_train,
        )

        # Compute differences
        diff = H_pred_dense - H_gt_dense
        if S_dense is not None:
            diff_corrected = diff - mu_H * S_dense
        else:
            diff_corrected = np.abs(diff)

        # Check if there's any data for this shift
        if np.abs(H_gt_dense).max() < 1e-10 and np.abs(H_pred_dense).max() < 1e-10:
            print(f"  Skipping (no data for this shift)")
            continue

        # Create figure with 2x2 subplots (skip overlap)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Build title with irrep information if present
        if filename_prefix == "hamiltonian":
            title = f"{matrix_label} Analysis: shift = [{sx}, {sy}, {sz}]"
        elif filename_prefix == "hamiltonian_rotated":
            title = f"{matrix_label} Analysis (Rotated): shift = [{sx}, {sy}, {sz}]"
        else:
            # Extract irrep from prefix (e.g., "hamiltonian_0e" -> "0e")
            irrep_name = filename_prefix.replace("hamiltonian_", "")
            title = f"{matrix_label} Analysis - Irrep {irrep_name}: shift = [{sx}, {sy}, {sz}]"

        if partial_train is not None:
            title += f" ({partial_train} blocks only)"
        fig.suptitle(title, fontsize=16)

        # Determine color ranges
        if per_panel_dynamic_range and dynamic_range:
            main_abs = np.concatenate(
                [np.abs(H_gt_dense).ravel(), np.abs(H_pred_dense).ravel()]
            )
            v_main = max(float(np.percentile(main_abs, percentile)), 1e-12)
            vmin_gt, vmax_gt = -v_main, v_main
            vmin_pred, vmax_pred = -v_main, v_main

            if diff_dynamic_range:
                diff_abs = np.concatenate(
                    [np.abs(diff).ravel(), np.abs(diff_corrected).ravel()]
                )
                v_diff_shared = max(float(np.percentile(diff_abs, percentile)), 1e-12)
            else:
                v_diff_shared = v_main

            vmin_diff, vmax_diff = -v_diff_shared, v_diff_shared
            vmin_diff_corr, vmax_diff_corr = -v_diff_shared, v_diff_shared
        else:
            # Determine color range for ground truth/pred
            if dynamic_range:
                v = np.percentile(np.abs(H_gt_dense), percentile)
                vmin_main, vmax_main = -v, v
            else:
                vmin_main, vmax_main = -1, 1

            # Determine color range for diff panels
            if diff_dynamic_range:
                v_diff = np.percentile(np.abs(diff), percentile)
                vmin_diff, vmax_diff = -v_diff, v_diff
                v_diff_corr = np.percentile(np.abs(diff_corrected), percentile)
                vmin_diff_corr, vmax_diff_corr = -v_diff_corr, v_diff_corr
            else:
                vmin_diff, vmax_diff = vmin_main, vmax_main
                vmin_diff_corr, vmax_diff_corr = vmin_main, vmax_main

            vmin_gt, vmax_gt = vmin_main, vmax_main
            vmin_pred, vmax_pred = vmin_main, vmax_main

        # 1. Ground truth (top-left)
        im0 = axes[0, 0].imshow(H_gt_dense, cmap="bwr", vmin=vmin_gt, vmax=vmax_gt)
        axes[0, 0].set_title(
            f"Ground Truth {matrix_label}\nMax: {np.abs(H_gt_dense).max():.3f}"
        )
        axes[0, 0].set_xlabel("Orbital j")
        axes[0, 0].set_ylabel("Orbital i")
        plt.colorbar(im0, ax=axes[0, 0])

        # 2. Predicted (top-right)
        im1 = axes[0, 1].imshow(
            H_pred_dense, cmap="bwr", vmin=vmin_pred, vmax=vmax_pred
        )
        axes[0, 1].set_title(
            f"Predicted {matrix_label}\nMax: {np.abs(H_pred_dense).max():.3f}"
        )
        axes[0, 1].set_xlabel("Orbital j")
        axes[0, 1].set_ylabel("Orbital i")
        plt.colorbar(im1, ax=axes[0, 1])

        # 3. Difference (bottom-left)
        im2 = axes[1, 0].imshow(diff, cmap="bwr", vmin=vmin_diff, vmax=vmax_diff)
        axes[1, 0].set_title(f"Difference (pred - gt)\nMAE: {np.abs(diff).mean():.3e}")
        axes[1, 0].set_xlabel("Orbital j")
        axes[1, 0].set_ylabel("Orbital i")
        plt.colorbar(im2, ax=axes[1, 0])

        # 4. Corrected difference (bottom-right) or abs diff for non-H matrices
        im3 = axes[1, 1].imshow(
            diff_corrected,
            cmap="bwr",
            vmin=vmin_diff_corr,
            vmax=vmax_diff_corr,
        )
        if S_dense is not None:
            axes[1, 1].set_title(
                f"Corrected Diff (mu_H={mu_H:.2e})\nMAE: {np.abs(diff_corrected).mean():.3e}"
            )
        else:
            axes[1, 1].set_title(
                f"Absolute Error |pred-gt|\nMAE: {np.abs(diff).mean():.3e}"
            )
        axes[1, 1].set_xlabel("Orbital j")
        axes[1, 1].set_ylabel("Orbital i")
        plt.colorbar(im3, ax=axes[1, 1])

        # Save figure
        filename = f"{filename_prefix}_sx{sx:+d}_sy{sy:+d}_sz{sz:+d}.png"
        filepath = output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved to {filepath}")


def create_hamiltonian_frame_figure(
    H_pred,
    H_gt,
    S,
    atoms_list,
    orbital_cfg,
    sx=0,
    sy=0,
    sz=0,
    dynamic_range=False,
    diff_dynamic_range=False,
    per_panel_dynamic_range=False,
    partial_train=None,
    percentile=80.0,
    figsize=(12, 12),
    return_buffer=False,
    epoch=None,
    max_atoms=None,
    matrix_label: str = "H",
):
    """
    Create a single matplotlib figure showing Hamiltonian comparison for one shift.

    This function creates the 2x2 subplot visualization without saving to disk.
    Useful for creating video frames or one-off analysis.

    Args:
        H_pred: Predicted Hamiltonian BlockMatrix
        H_gt: Ground truth Hamiltonian BlockMatrix
        S: Overlap BlockMatrix
        atoms_list: List of atom symbols
        orbital_cfg: Orbital configuration
        sx, sy, sz: Cell shift indices (default: [0, 0, 0])
        dynamic_range: If True, use percentile of abs(H_gt) for color range
        diff_dynamic_range: If True, scale diff plots by percentile of abs(diff)
                            and abs(diff_corrected). Defaults to False (use
                            same range as ground truth/prediction).
        per_panel_dynamic_range: If True and dynamic_range is enabled, use one
                                 shared dynamic range for (gt, pred) and a second
                                 shared range for (diff, diff_corrected). Defaults
                                 to False (old behavior).
        partial_train: "diag", "offdiag", or None - filters which blocks to show
        percentile: Percentile value for dynamic color range (default: 80.0)
        figsize: Figure size (default: (12, 12))
        return_buffer: If True, return figure as PNG-encoded bytes instead of figure object
        epoch: Optional epoch number to display on the figure
        max_atoms: Optional int. If set to >0, display only the top-left
                  submatrix corresponding to the first `max_atoms` atoms.

    Returns:
        matplotlib.figure.Figure or bytes: Figure object (or PNG bytes if return_buffer=True)
    """

    # Extract partial Hamiltonians
    H_gt_dense, S_dense, dim = extract_partial_hamiltonian(
        H_gt, S, atoms_list, orbital_cfg, sx, sy, sz, partial_train=partial_train
    )
    H_pred_dense, _, _ = extract_partial_hamiltonian(
        H_pred,
        None,
        atoms_list,
        orbital_cfg,
        sx,
        sy,
        sz,
        partial_train=partial_train,
    )

    if max_atoms is not None and int(max_atoms) > 0:
        atom_count = min(int(max_atoms), len(atoms_list))
        orbital_dims = [orbital_cfg.element_to_irreps[a].dim for a in atoms_list]
        cut_dim = int(sum(orbital_dims[:atom_count]))
        H_gt_dense = H_gt_dense[:cut_dim, :cut_dim]
        H_pred_dense = H_pred_dense[:cut_dim, :cut_dim]
        if S_dense is not None:
            S_dense = S_dense[:cut_dim, :cut_dim]

    # Compute differences
    diff = H_pred_dense - H_gt_dense
    if S_dense is not None:
        mu_H = compute_mu_H(H_pred, H_gt, S)
        diff_corrected = diff - mu_H * S_dense
    else:
        mu_H = 0.0
        diff_corrected = np.abs(diff)

    # Check if there's any data for this shift
    if np.abs(H_gt_dense).max() < 1e-10 and np.abs(H_pred_dense).max() < 1e-10:
        raise ValueError(f"No data for shift [{sx}, {sy}, {sz}]")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Determine color ranges
    if per_panel_dynamic_range and dynamic_range:
        main_abs = np.concatenate(
            [np.abs(H_gt_dense).ravel(), np.abs(H_pred_dense).ravel()]
        )
        v_main = max(float(np.percentile(main_abs, percentile)), 1e-12)
        vmin_gt, vmax_gt = -v_main, v_main
        vmin_pred, vmax_pred = -v_main, v_main

        if diff_dynamic_range:
            diff_abs = np.concatenate(
                [np.abs(diff).ravel(), np.abs(diff_corrected).ravel()]
            )
            v_diff_shared = max(float(np.percentile(diff_abs, percentile)), 1e-12)
        else:
            v_diff_shared = v_main

        vmin_diff, vmax_diff = -v_diff_shared, v_diff_shared
        vmin_diff_corr, vmax_diff_corr = -v_diff_shared, v_diff_shared
    else:
        # Determine color range for ground truth/pred
        if dynamic_range:
            v = np.percentile(np.abs(H_gt_dense), percentile)
            vmin_main, vmax_main = -v, v
        else:
            vmin_main, vmax_main = -1, 1

        # Determine color range for diff panels
        if diff_dynamic_range:
            v_diff = np.percentile(np.abs(diff), percentile)
            vmin_diff, vmax_diff = -v_diff, v_diff
            v_diff_corr = np.percentile(np.abs(diff_corrected), percentile)
            vmin_diff_corr, vmax_diff_corr = -v_diff_corr, v_diff_corr
        else:
            vmin_diff, vmax_diff = vmin_main, vmax_main
            vmin_diff_corr, vmax_diff_corr = vmin_main, vmax_main

        vmin_gt, vmax_gt = vmin_main, vmax_main
        vmin_pred, vmax_pred = vmin_main, vmax_main

    # 1. Ground truth (top-left)
    im0 = axes[0, 0].imshow(H_gt_dense, cmap="bwr", vmin=vmin_gt, vmax=vmax_gt)
    axes[0, 0].set_title(
        f"Ground Truth {matrix_label}\nMax: {np.abs(H_gt_dense).max():.3f}"
    )
    axes[0, 0].set_xlabel("Orbital j")
    axes[0, 0].set_ylabel("Orbital i")
    plt.colorbar(im0, ax=axes[0, 0])

    # 2. Predicted (top-right)
    im1 = axes[0, 1].imshow(H_pred_dense, cmap="bwr", vmin=vmin_pred, vmax=vmax_pred)
    axes[0, 1].set_title(
        f"Predicted {matrix_label}\nMax: {np.abs(H_pred_dense).max():.3f}"
    )
    axes[0, 1].set_xlabel("Orbital j")
    axes[0, 1].set_ylabel("Orbital i")
    plt.colorbar(im1, ax=axes[0, 1])

    # 3. Difference (bottom-left)
    im2 = axes[1, 0].imshow(diff, cmap="bwr", vmin=vmin_diff, vmax=vmax_diff)
    axes[1, 0].set_title(f"Difference (pred - gt)\nMAE: {np.abs(diff).mean():.3e}")
    axes[1, 0].set_xlabel("Orbital j")
    axes[1, 0].set_ylabel("Orbital i")
    plt.colorbar(im2, ax=axes[1, 0])

    # 4. Corrected difference (bottom-right) or abs diff for non-H matrices
    im3 = axes[1, 1].imshow(
        diff_corrected,
        cmap="bwr",
        vmin=vmin_diff_corr,
        vmax=vmax_diff_corr,
    )
    if S_dense is not None:
        axes[1, 1].set_title(
            f"Corrected Diff (mu_H={mu_H:.2e})\nMAE: {np.abs(diff_corrected).mean():.3e}"
        )
    else:
        axes[1, 1].set_title(
            f"Absolute Error |pred-gt|\nMAE: {np.abs(diff).mean():.3e}"
        )
    axes[1, 1].set_xlabel("Orbital j")
    axes[1, 1].set_ylabel("Orbital i")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()

    # Add epoch number overlay if provided
    if epoch is not None:
        fig.text(
            0.02,
            0.98,
            f"Epoch: {epoch}",
            transform=fig.transFigure,
            fontsize=16,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    if return_buffer:
        # Convert figure to PNG bytes
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        png_bytes = buffer.getvalue()
        plt.close(fig)
        return png_bytes
    else:
        return fig


def save_hamiltonian_frame_to_disk(
    H_pred,
    H_gt,
    S,
    atoms_list,
    orbital_cfg,
    output_dir,
    epoch,
    sx=0,
    sy=0,
    sz=0,
    dynamic_range=False,
    diff_dynamic_range=False,
    per_panel_dynamic_range=False,
    partial_train=None,
    percentile=80.0,
    max_atoms=None,
    matrix_label: str = "H",
):
    """
    Save a single Hamiltonian visualization frame to disk for video creation.

    Args:
        H_pred: Predicted Hamiltonian BlockMatrix
        H_gt: Ground truth Hamiltonian BlockMatrix
        S: Overlap BlockMatrix
        atoms_list: List of atom symbols
        orbital_cfg: Orbital configuration
        output_dir: Directory to save frames
        epoch: Epoch number (used in filename)
        sx, sy, sz: Cell shift indices (default: [0, 0, 0])
        dynamic_range: If True, use percentile of abs(H_gt) for color range
        diff_dynamic_range: If True, scale diff plots by percentile of abs(diff)
                            and abs(diff_corrected). Defaults to False (use
                            same range as ground truth/prediction).
        per_panel_dynamic_range: If True and dynamic_range is enabled, use one
                                 shared dynamic range for (gt, pred) and a second
                                 shared range for (diff, diff_corrected). Defaults
                                 to False (old behavior).
        partial_train: "diag", "offdiag", or None - filters which blocks to show
        percentile: Percentile value for dynamic color range (default: 80.0)
        max_atoms: Optional int. If set to >0, display only the top-left
                  submatrix corresponding to the first `max_atoms` atoms.

    Returns:
        Path to saved PNG file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig = create_hamiltonian_frame_figure(
        H_pred,
        H_gt,
        S,
        atoms_list,
        orbital_cfg,
        sx=sx,
        sy=sy,
        sz=sz,
        dynamic_range=dynamic_range,
        diff_dynamic_range=diff_dynamic_range,
        per_panel_dynamic_range=per_panel_dynamic_range,
        partial_train=partial_train,
        percentile=percentile,
        epoch=epoch,
        max_atoms=max_atoms,
        matrix_label=matrix_label,
    )

    # Save figure
    filename = f"frame_epoch_{epoch:06d}.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    return filepath


def compile_frames_to_video(
    frame_dir,
    output_path,
    fps=5,
    pattern="frame_epoch_*.png",
    format="mp4",
):
    """
    Compile PNG frames into an MP4 video using imageio.

    Frames are sorted by filename to ensure correct ordering.

    Args:
        frame_dir: Directory containing PNG frames
        output_path: Path to save MP4 file
        fps: Frames per second for video (default: 5)
        pattern: Glob pattern for frame files (default: "frame_epoch_*.png")

    Returns:
        Path to created video file

    Raises:
        ImportError: If imageio is not installed
        FileNotFoundError: If no frames found in directory
    """
    frame_dir = Path(frame_dir)
    output_path = Path(output_path)

    # Find all matching frames
    frames_list = sorted(frame_dir.glob(pattern))

    if not frames_list:
        raise FileNotFoundError(
            f"No frames found in {frame_dir} matching pattern {pattern}"
        )

    print(f"\nCompiling {len(frames_list)} frames into video...")
    print(f"  First frame: {frames_list[0].name}")
    print(f"  Last frame: {frames_list[-1].name}")
    print(f"  FPS: {fps}")
    print(f"  Output: {output_path}")

    # Read all frames
    frames = [imageio.imread(str(frame_path)) for frame_path in frames_list]

    # Write video
    imageio.mimsave(output_path, frames, fps=fps, format=format)

    print(f"[OK] Video saved to {output_path}")
    return output_path


def filter_irreps_block_data_by_irrep(irreps_block_data, target_irrep, mapper):
    """
    Filter IrrepsBlockData to only include contributions from a specific irrep.
    All other irrep components are zeroed out.

    Args:
        irreps_block_data: IrrepsBlockData object
        target_irrep: e3nn.o3.Irrep to keep (e.g., Irrep("0e"), Irrep("1o"))
        mapper: BlockIrrepMapper object

    Returns:
        Filtered IrrepsBlockData with only target_irrep contributions
    """
    from e3nn.o3 import Irrep
    from data.block_matrix import IrrepsBlockData

    if isinstance(target_irrep, str):
        target_irrep = Irrep(target_irrep)

    filtered_vectors = {}

    for edge_type, vectors in irreps_block_data.pair_vectors.items():
        # Get pair irreps for this edge type
        pair_irreps = mapper.get_pair_irreps(edge_type)

        # Create mask for target irrep
        filtered_vec = torch.zeros_like(vectors)
        start_idx = 0

        for mul, irrep in pair_irreps:
            irrep_dim = irrep.dim * mul

            if irrep == target_irrep:
                # Keep this irrep's contribution
                filtered_vec[:, start_idx : start_idx + irrep_dim] = vectors[
                    :, start_idx : start_idx + irrep_dim
                ]

            start_idx += irrep_dim

        filtered_vectors[edge_type] = filtered_vec

    return IrrepsBlockData(
        atoms=irreps_block_data.atoms,
        atom_counts=irreps_block_data.atom_counts,
        pair_vectors=filtered_vectors,
        pair_edges=irreps_block_data.pair_edges,
        lookup=irreps_block_data.lookup,
        orbital_cfg=irreps_block_data.orbital_cfg,
        basis=irreps_block_data.basis,
    )


def split_hamiltonian_by_irrep(H_matrix, mapper, target_irrep):
    """
    Split a BlockMatrix Hamiltonian to keep only one irrep's contribution.

    Args:
        H_matrix: BlockMatrix Hamiltonian
        mapper: BlockIrrepMapper
        target_irrep: e3nn.o3.Irrep or string like "0e", "1o", "2e"

    Returns:
        BlockMatrix with only the target irrep's contribution
    """
    irreps_data = H_matrix.to_vectors(mapper)
    filtered_irreps = filter_irreps_block_data_by_irrep(
        irreps_data, target_irrep, mapper
    )
    return filtered_irreps.to_blocks(mapper)


def compute_irrep_metrics(pred_H_irreps, target_H_irreps, all_irreps, mapper):
    """
    Compute per-irrep block metrics.

    Args:
        pred_H_irreps: Predicted Hamiltonian as IrrepsBlockData
        target_H_irreps: Target Hamiltonian as IrrepsBlockData
        all_irreps: List of all irreps in the system
        mapper: BlockIrrepMapper object

    Returns:
        Dictionary with keys:
        - {irrep_str}_l1_block_abs
        - {irrep_str}_l1_block_rel
        - {irrep_str}_l2_block_abs
        - {irrep_str}_l2_block_rel
    """

    irrep_metrics = {}
    _REL_EPS = 1e-12

    for irrep in all_irreps:
        # Filter both prediction and target by this irrep
        pred_irrep_filtered = filter_irreps_block_data_by_irrep(
            pred_H_irreps, irrep, mapper
        )
        target_irrep_filtered = filter_irreps_block_data_by_irrep(
            target_H_irreps, irrep, mapper
        )

        pred_irrep_blocks = pred_irrep_filtered.to_blocks(mapper)
        target_irrep_blocks = target_irrep_filtered.to_blocks(mapper)

        # Initialize accumulators (block-level only)
        sum_l1_block = 0.0  # Sum of L1 norms (block-level)
        sum_l1_target_block = 0.0  # Sum of L1 norms of target blocks
        sum_l2_block_sq = 0.0  # Sum of L2 norms squared (block-level)
        sum_l2_target_block_sq = 0.0  # Sum of L2 norms squared of target blocks
        total_blocks = 0  # Count of blocks

        # Iterate through each edge type
        for key in target_irrep_blocks.pair_blocks.keys():
            if key in pred_irrep_blocks.pair_blocks:
                pred_blocks = pred_irrep_blocks.pair_blocks[
                    key
                ]  # (num_edges, dim_i, dim_j)
                targ_blocks_full = target_irrep_blocks.pair_blocks[
                    key
                ]  # (num_edges, dim_i, dim_j)
                min_n = min(pred_blocks.shape[0], targ_blocks_full.shape[0])
                if min_n <= 0:
                    continue

                pred_sel = pred_blocks[:min_n]
                targ_sel = targ_blocks_full[:min_n]

                # Vectorized computation (all blocks at once)
                # Shape: (num_blocks, dim_i, dim_j)
                diff = pred_sel - targ_sel

                # Compute norms per block: shape (num_blocks,)
                l1_error_blocks = torch.sum(torch.abs(diff), dim=(1, 2))
                l2_error_blocks_sq = torch.sum(diff**2, dim=(1, 2))
                l1_target_blocks = torch.sum(torch.abs(targ_sel), dim=(1, 2))
                l2_target_blocks_sq = torch.sum(targ_sel**2, dim=(1, 2))

                # Accumulate block-level metrics
                sum_l1_block += torch.sum(l1_error_blocks).item()
                sum_l1_target_block += torch.sum(l1_target_blocks).item()
                sum_l2_block_sq += torch.sum(l2_error_blocks_sq).item()
                sum_l2_target_block_sq += torch.sum(l2_target_blocks_sq).item()
                total_blocks += pred_sel.shape[0]
        if total_blocks > 0:

            # Compute block-level metrics
            l1_block_abs = sum_l1_block / max(total_blocks, 1)
            l2_block_abs = (sum_l2_block_sq / max(total_blocks, 1)) ** 0.5

            # Compute block-level relative metrics
            l1_block_rel = sum_l1_block / (sum_l1_target_block + _REL_EPS)
            l2_block_rel = (sum_l2_block_sq**0.5) / (
                sum_l2_target_block_sq**0.5 + _REL_EPS
            )

            # Store all metrics
            irrep_str = str(irrep)
            irrep_metrics[f"{irrep_str}_l1_block_abs"] = l1_block_abs
            irrep_metrics[f"{irrep_str}_l1_block_rel"] = l1_block_rel
            irrep_metrics[f"{irrep_str}_l2_block_abs"] = l2_block_abs
            irrep_metrics[f"{irrep_str}_l2_block_rel"] = l2_block_rel

    return irrep_metrics


def get_all_irreps_in_hamiltonian(mapper):
    """
    Get a list of all unique irreps present in the Hamiltonian.

    Args:
        mapper: BlockIrrepMapper

    Returns:
        List of unique Irrep objects
    """

    irreps_set = set()
    for edge_type in mapper.edge_types:
        pair_irreps = mapper.get_pair_irreps(edge_type)
        for mul, irrep in pair_irreps:
            if mul > 0:  # Only include irreps that are actually present
                irreps_set.add(irrep)

    # Sort by l, then by parity
    return sorted(irreps_set, key=lambda ir: (ir.l, ir.p))


def permutation_to_matrix(perm_str, device, dtype=torch.float32):
    """
    Convert permutation string to change of basis matrix.

    Args:
        perm_str: Permutation string like '120' or '201' (must be a valid permutation of '012')
        device: Torch device

    Returns:
        3x3 permutation matrix as torch.Tensor, or identity if permutation is invalid
    """
    try:
        perm = [int(c) for c in perm_str]
        if (
            len(perm) != 3
            or not all(p in [0, 1, 2] for p in perm)
            or len(set(perm)) != 3
        ):
            raise ValueError(
                f"Invalid permutation: {perm_str}. Must be a 3-digit permutation of 012."
            )
        # Create permutation matrix where row i has 1 at column perm[i]
        matrix = torch.zeros(3, 3, dtype=dtype, device=device)
        for i in range(3):
            matrix[i, perm[i]] = 1.0
        return matrix
    except (ValueError, IndexError) as e:
        print(f"  Warning: Invalid permutation string: {e}")
        print(f"  Using identity permutation (no transformation)")
        return torch.eye(3, dtype=dtype, device=device)
