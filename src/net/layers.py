"""
layers.py
~~~~~~~~~

E(3)-equivariant message-passing blocks:

*   uses `torch_scatter.scatter` for edge->node aggregation
*   dropout is `e3nn.nn.Dropout` (acts on *all* irrep coeffs)
*   normalisation + activation selected via `make_nonlinearity`
"""

from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter
from collections import OrderedDict

from e3nn.o3 import Irreps
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Dropout
from e3nn.nn import Gate

from net.common import Config, E3MLP, RadialMLP, SeparateWeightTensorProduct
from net.activations import make_nonlinearity
from net.layer_norm import E3LayerNorm


def _magnitude_splits(
    features: torch.Tensor,
    irreps: Irreps,
) -> dict[str, torch.Tensor]:
    """
    Split features by irrep and compute magnitude per irreducible component.
    Returns a mapping from angular momentum l to a 1D tensor of magnitudes.
    """
    mags: dict[str, torch.Tensor] = OrderedDict()
    # features: (M, D)
    start = 0
    for mul, ir in irreps:
        dim = ir.dim
        size = mul * dim
        # slice for this irrep
        chunk = features[:, start : start + size]
        # reshape to (M * mul, dim)
        if mul > 0 and dim > 0:
            reshaped = chunk.reshape(-1, dim)
            # magnitude across dim
            mag = torch.linalg.norm(reshaped, dim=1)
            mags[f"{mul}x{ir.l}{'e' if ir.p == 1 else 'o'}"] = mag
        start += size
    return mags


def _tp_path_exists(irreps_in1: Irreps, irreps_in2: Irreps, ir_out) -> bool:
    """Check if a tensor product path exists for given irreps."""
    from e3nn.o3 import Irrep

    irreps_in1 = Irreps(irreps_in1)
    irreps_in2 = Irreps(irreps_in2)

    # Handle both Irrep and Irreps
    if isinstance(ir_out, Irrep):
        target_irreps = [ir_out]
    elif isinstance(ir_out, str):
        # Parse string like "1o" as single Irrep
        try:
            target_irreps = [Irrep(ir_out)]
        except (ValueError, TypeError):
            # If it fails, treat as Irreps
            target_irreps = [ir for _, ir in Irreps(ir_out)]
    else:
        # Assume it's Irreps
        target_irreps = [ir for _, ir in Irreps(ir_out)]

    # Check if ANY of the target irreps can be produced
    for target_ir in target_irreps:
        for _, ir1 in irreps_in1:
            for _, ir2 in irreps_in2:
                if target_ir in ir1 * ir2:
                    return True
    return False


# ════════════════════════════════════════════════════════════════════════
# EquiConv - E(3)-equivariant convolution with radial weighting
# ════════════════════════════════════════════════════════════════════════
class EquiConv(nn.Module):
    """
    E(3)-equivariant convolution with radial MLP weighting.

    Similar to DeepH-E3's EquiConv. Combines:
    - Tensor product: irreps_in1 (x) irreps_in2 -> irreps_out
    - Radial MLP: generates weights from edge length embeddings
    - Optional Gate nonlinearity
    - Element-wise multiplication with learned weights

    Args:
        n_radial: Input dimension for radial MLP (edge length embedding size)
        irreps_in1: Input irreps for geometric features (node/edge concatenation)
        irreps_in2: Input irreps for spherical harmonics
        irreps_out: Output irreps
        cfg: Configuration object
        nonlin: Whether to use Gate nonlinearity after TP
    """

    def __init__(
        self,
        n_radial: int,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        cfg: Config,
        nonlin: bool = True,
        info: dict | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.info = info or {}

        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        # Determine if we need a gate nonlinearity
        self.nonlin = None
        if nonlin:
            # Separate scalars and gated irreps
            irreps_scalars = Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_out
                    if ir.l == 0 and _tp_path_exists(irreps_in1, irreps_in2, ir)
                ]
            ).simplify()

            irreps_gated = Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_out
                    if ir.l > 0 and _tp_path_exists(irreps_in1, irreps_in2, ir)
                ]
            ).simplify()

            if irreps_gated.dim > 0:
                # Need gates (scalars) for gated tensors
                if _tp_path_exists(irreps_in1, irreps_in2, "0e"):
                    ir_gate = "0e"
                elif _tp_path_exists(irreps_in1, irreps_in2, "0o"):
                    ir_gate = "0o"
                else:
                    raise ValueError(
                        f"Cannot produce gates for irreps_gated={irreps_gated}"
                    )
                irreps_gates = Irreps(
                    [(mul, ir_gate) for mul, _ in irreps_gated]
                ).simplify()
            else:
                irreps_gates = Irreps([])

            # Create gate nonlinearity
            act_scalars = []
            for _, ir in irreps_scalars:
                if ir.p == 1:  # even parity
                    act_scalars.append(torch.nn.functional.silu)
                else:  # odd parity
                    act_scalars.append(torch.tanh)

            act_gates = []
            for _, ir in irreps_gates:
                if ir.p == 1:  # even parity
                    act_gates.append(torch.sigmoid)
                else:  # odd parity
                    act_gates.append(torch.tanh)

            self.nonlin = Gate(
                irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
            )
            irreps_tp_out = self.nonlin.irreps_in
        else:
            irreps_tp_out = Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_out
                    if _tp_path_exists(irreps_in1, irreps_in2, ir)
                ]
            )

        # Create tensor product
        if cfg.tp_type == "separate_weight":
            self.tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_tp_out)
        elif cfg.tp_type == "fully_connected":
            self.tp = FullyConnectedTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_tp_out,
                internal_weights=True,
                shared_weights=True,
            )
        else:
            raise ValueError(f"Unknown tp_type: {cfg.tp_type}")

        # Determine output irreps
        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = irreps_tp_out

        # Element-wise multiplication weights (learned per-irrep scaling)
        # This is computed from radial features via MLP
        self.radial_mlp = RadialMLP(
            in_dim=n_radial,
            out_dim=self.irreps_out.num_irreps,  # One weight per irrep
            layers=self.cfg.radial_layers,
            act="silu",
            dtype=self.cfg.dtype,
        )

    def forward(
        self,
        fea_in1: torch.Tensor,
        fea_in2: torch.Tensor,
        edge_length_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fea_in1: Geometric features (batch, irreps_in1.dim)
            fea_in2: Spherical harmonics (batch, irreps_in2.dim)
            edge_length_emb: Radial embeddings (batch, n_radial)

        Returns:
            Tensor of shape (batch, irreps_out.dim)
        """
        # Tensor product
        z = self.tp(fea_in1, fea_in2)

        # Handle empty output
        if z.shape[-1] == 0:
            return z

        # Apply gate nonlinearity if present
        if self.nonlin is not None:
            z = self.nonlin(z)

        # Radial weighting (element-wise multiplication)
        weights = self.radial_mlp(edge_length_emb)

        # Apply weights per irrep
        z_weighted = []
        start = 0
        for idx, (mul, ir) in enumerate(self.irreps_out):
            size = mul * ir.dim
            z_slice = z[:, start : start + size]
            # Broadcast weight to all components of this irrep
            w = weights[:, idx : idx + 1]
            z_weighted.append(z_slice * w)
            start += size

        if len(z_weighted) == 0:
            return torch.zeros(z.shape[0], 0, dtype=z.dtype, device=z.device)

        return torch.cat(z_weighted, dim=-1)


# ════════════════════════════════════════════════════════════════════════
# Edge update
# ════════════════════════════════════════════════════════════════════════
class EdgeUpdateBlock(nn.Module):
    """
    Edge update block following DeepH-E3 architecture.

    For each edge (i -> j):
      • Concatenate node[i], node[j], and edge features
      • Apply EquiConv with spherical harmonics and radial weighting
      • Apply linear transformation and nonlinearity
      • Optional self-connection and residual
    """

    def __init__(
        self,
        node_irreps: Irreps,
        edge_irreps: Irreps,
        num_species: int,
        cfg: Config,
        info: dict = None,
        edge_irreps_out: Irreps | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.info = info
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps
        if edge_irreps_out is None:
            edge_irreps_out = edge_irreps
        self.edge_irreps_out = edge_irreps_out

        # Resolve sh_irreps and n_radial from config
        sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
        n_radial = cfg.n_radial

        # Pre-linear transformation
        self.lin_pre = E3MLP(
            edge_irreps,
            edge_irreps,
            edge_irreps,
            1,  # Single layer
            cfg,
            activate_last=False,
        )

        # Concatenated input: node[i] + node[j] + edge
        irreps_in1 = (node_irreps + node_irreps + edge_irreps).simplify()
        irreps_in2 = sh_irreps

        # EquiConv: main tensor product with radial weighting
        self.conv = EquiConv(
            n_radial=n_radial,
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=edge_irreps_out,
            cfg=cfg,
            nonlin=True,  # Use gate nonlinearity
            info=info,
        )

        # Post-linear transformation
        self.lin_post = E3MLP(
            self.conv.irreps_out,
            self.conv.irreps_out,
            self.conv.irreps_out,
            1,
            cfg,
            activate_last=False,
        )

        # Self-connection (edge-type specific)
        self.sc = None
        if cfg.use_self_connection:
            # num_species^2 edge types (all pairs of elements)
            n_edge_types = num_species * num_species
            self.sc = FullyConnectedTensorProduct(
                edge_irreps,
                Irreps(f"{n_edge_types}x0e"),
                self.conv.irreps_out,
                internal_weights=True,
                shared_weights=True,
            )

        # Normalization and activation
        self.norm_act = make_nonlinearity(self.conv.irreps_out, cfg)
        self.layer_norm = E3LayerNorm(self.conv.irreps_out) if cfg.e3layernorm else None

        # Dropout
        if cfg.dropout > 0.0:
            self.dropout = Dropout(self.conv.irreps_out, p=cfg.dropout)
        else:
            self.dropout = None

        internal_variant = cfg.internal_e3mlp_variant or cfg.e3mlp_variant
        if cfg.internal_e3mlp_layers > 0:
            self.refine = E3MLP(
                self.conv.irreps_out,
                edge_irreps_out,
                edge_irreps_out,
                cfg.internal_e3mlp_layers,
                cfg,
                activate_last=False,
                variant=internal_variant,
            )
        else:
            self.refine = nn.Identity()

        self.irreps_out = self.conv.irreps_out

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_length_emb: torch.Tensor,
        edge_one_hot: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            node: Node features (N, node_irreps.dim)
            edge: Edge features (E, edge_irreps.dim)
            edge_index: Edge indices (2, E)
            edge_sh: Spherical harmonics (E, sh_irreps.dim)
            edge_length_emb: Radial embeddings (E, n_radial)
            edge_one_hot: Edge type one-hot (E, num_species^2) for self-connection

        Returns:
            Updated edge features (E, irreps_out.dim)
        """
        edge_old = edge

        # Self-connection
        if self.sc is not None and edge_one_hot is not None:
            edge_self_connection = self.sc(edge, edge_one_hot)

        # Pre-linear
        edge = self.lin_pre(edge)

        # Concatenate node features
        src, dst = edge_index
        fea_in = torch.cat([node[src], node[dst], edge], dim=-1)

        # EquiConv
        edge = self.conv(fea_in, edge_sh, edge_length_emb)

        # Post-linear
        edge = self.lin_post(edge)

        # Add self-connection
        if self.sc is not None and edge_one_hot is not None:
            edge = edge + edge_self_connection

        # Normalization and activation
        edge = self.norm_act(edge)
        if self.layer_norm is not None:
            edge = self.layer_norm(edge)

        # Dropout
        if self.dropout:
            edge = self.dropout(edge)

        edge = self.refine(edge)

        # Residual connection
        if self.cfg.edge_update_residual and edge.shape == edge_old.shape:
            edge = edge + edge_old

        return edge


# ════════════════════════════════════════════════════════════════════════
# Node update
# ════════════════════════════════════════════════════════════════════════
class NodeUpdateBlock(nn.Module):
    """
    Node update block following DeepH-E3 architecture.

    Aggregates edge messages to nodes:
      • Concatenate node[i], node[j], and edge features
      • Apply EquiConv to create edge messages
      • Scatter-aggregate messages to destination nodes
      • Apply transformations and optional self-connection
    """

    def __init__(
        self,
        node_irreps: Irreps,
        edge_irreps: Irreps,
        num_species: int,
        cfg: Config,
        info: dict = None,
        node_irreps_out: Irreps = None,  # Output irreps for nodes (defaults to edge_irreps)
    ):
        super().__init__()
        self.cfg = cfg
        self.info = info
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps

        # Default: output nodes with same irreps as edges
        if node_irreps_out is None:
            node_irreps_out = edge_irreps

        # Resolve sh_irreps and n_radial from config
        sh_irreps = Irreps.spherical_harmonics(cfg.l_max)
        n_radial = cfg.n_radial

        # Pre-linear transformation
        self.lin_pre = E3MLP(
            node_irreps,
            node_irreps,
            node_irreps,
            1,
            cfg,
            activate_last=False,
        )

        # Concatenated input: node[i] + node[j] + edge
        irreps_in1 = (node_irreps + node_irreps + edge_irreps).simplify()
        irreps_in2 = sh_irreps

        # EquiConv: main tensor product with radial weighting
        self.conv = EquiConv(
            n_radial=n_radial,
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=node_irreps_out,
            cfg=cfg,
            nonlin=True,
            info=info,
        )

        # Post-linear transformation
        self.lin_post = E3MLP(
            self.conv.irreps_out,
            self.conv.irreps_out,
            self.conv.irreps_out,
            1,
            cfg,
            activate_last=False,
        )

        # Self-connection (node-type specific)
        self.sc = None
        if cfg.use_self_connection:
            self.sc = FullyConnectedTensorProduct(
                node_irreps,
                Irreps(f"{num_species}x0e"),
                self.conv.irreps_out,
                internal_weights=True,
                shared_weights=True,
            )  # Normalization and activation
        self.norm_act = make_nonlinearity(self.conv.irreps_out, cfg)
        self.layer_norm = E3LayerNorm(self.conv.irreps_out) if cfg.e3layernorm else None

        # Dropout
        if cfg.dropout > 0.0:
            self.dropout = Dropout(self.conv.irreps_out, p=cfg.dropout)
        else:
            self.dropout = None

        internal_variant = cfg.internal_e3mlp_variant or cfg.e3mlp_variant
        if cfg.internal_e3mlp_layers > 0:
            self.refine = E3MLP(
                self.conv.irreps_out,
                node_irreps_out,
                node_irreps_out,
                cfg.internal_e3mlp_layers,
                cfg,
                activate_last=False,
                variant=internal_variant,
            )
        else:
            self.refine = nn.Identity()

        self.irreps_out = self.conv.irreps_out

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_length_emb: torch.Tensor,
        node_one_hot: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            node: Node features (N, node_irreps.dim)
            edge: Edge features (E, edge_irreps.dim)
            edge_index: Edge indices (2, E)
            edge_sh: Spherical harmonics (E, sh_irreps.dim)
            edge_length_emb: Radial embeddings (E, n_radial)
            node_one_hot: Node type one-hot (N, num_species) for self-connection

        Returns:
            Updated node features (N, irreps_out.dim)
        """
        node_old = node

        # Self-connection
        if self.sc is not None and node_one_hot is not None:
            node_self_connection = self.sc(node, node_one_hot)

        # Pre-linear
        node = self.lin_pre(node)

        # Create edge messages
        src, dst = edge_index
        fea_in = torch.cat([node[src], node[dst], edge], dim=-1)

        # EquiConv to create messages
        edge_messages = self.conv(fea_in, edge_sh, edge_length_emb)

        # Aggregate messages to nodes
        node = scatter(
            edge_messages, dst, dim=0, dim_size=node_old.size(0), reduce="sum"
        )

        # Post-linear
        node = self.lin_post(node)

        # Add self-connection
        if self.sc is not None and node_one_hot is not None:
            node = node + node_self_connection

        # Normalization and activation
        node = self.norm_act(node)
        if self.layer_norm is not None:
            node = self.layer_norm(node)

        # Dropout
        if self.dropout:
            node = self.dropout(node)

        node = self.refine(node)

        # Residual connection (only if dimensions match)
        if self.cfg.node_update_residual and node.shape == node_old.shape:
            node = node + node_old

        return node


# ════════════════════════════════════════════════════════════════════════
# Message block
# ════════════════════════════════════════════════════════════════════════
class MessageBlock(nn.Module):
    """
    One message-passing step: node update followed by edge update.

    Combines EdgeUpdateBlock and NodeUpdateBlock following DeepH-E3.

    Args:
      node: Tensor of shape (N, hidden_dim)
      edge: Tensor of shape (E, hidden_dim)
      edge_index: LongTensor of shape (2, E) with source/dest indices
      edge_sh: Spherical harmonics (E, sh_irreps.dim)
      edge_length_emb: Radial embeddings (E, n_radial)
      node_one_hot: Node type one-hot (N, num_species)
      edge_one_hot: Edge type one-hot (E, num_species^2)

    Returns:
      Tuple (node_updated, edge_updated) of same shapes.
    """

    def __init__(
        self,
        node_irreps: Irreps,
        edge_irreps: Irreps,
        num_species: int,
        cfg: Config,
        info: dict = None,
        node_irreps_out: Irreps | None = None,
        edge_irreps_out: Irreps | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.info = info or {}
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps

        self.node_upd = NodeUpdateBlock(
            node_irreps=node_irreps,
            edge_irreps=edge_irreps,
            node_irreps_out=node_irreps_out,
            num_species=num_species,
            cfg=cfg,
            info=info,
        )
        self.edge_upd = EdgeUpdateBlock(
            node_irreps=self.node_upd.irreps_out,
            edge_irreps=edge_irreps,
            edge_irreps_out=edge_irreps_out,
            num_species=num_species,
            cfg=cfg,
            info=info,
        )
        self.node_irreps_out = self.node_upd.irreps_out
        self.edge_irreps_out = self.edge_upd.irreps_out

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_length_emb: torch.Tensor,
        node_one_hot: torch.Tensor = None,
        edge_one_hot: torch.Tensor = None,
        activation_mags: dict = None,
    ):
        """
        Forward pass through edge and node updates.

        Args:
            node: Node features (N, node_irreps.dim)
            edge: Edge features (E, edge_irreps.dim)
            edge_index: Edge indices (2, E)
            edge_sh: Spherical harmonics (E, sh_irreps.dim)
            edge_length_emb: Radial embeddings (E, n_radial)
            node_one_hot: Node type one-hot (N, num_species)
            edge_one_hot: Edge type one-hot (E, num_species^2)
            activation_mags: Optional dict to track activation magnitudes

        Returns:
            Tuple of (updated_node, updated_edge)
        """
        node = self.node_upd(
            node, edge, edge_index, edge_sh, edge_length_emb, node_one_hot
        )

        if activation_mags is not None and self.cfg.log_activation_mag and self.info:
            prefix = f"mag_node_{self.info.get('graph', 'graph')}_layer_{self.info.get('layer', 0)}"
            splits = _magnitude_splits(node, self.node_upd.irreps_out)
            for ir_str, mag in splits.items():
                tag = f"{prefix}_{ir_str}"
                activation_mags[tag] = mag

        edge = self.edge_upd(
            node, edge, edge_index, edge_sh, edge_length_emb, edge_one_hot
        )

        if activation_mags is not None and self.cfg.log_activation_mag and self.info:
            prefix = f"mag_edge_{self.info.get('graph', 'graph')}_layer_{self.info.get('layer', 0)}"
            splits = _magnitude_splits(edge, self.edge_upd.irreps_out)
            for ir_str, mag in splits.items():
                tag = f"{prefix}_{ir_str}"
                activation_mags[tag] = mag

        return node, edge
