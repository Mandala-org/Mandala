"""Composable equivariant transformations with reference parameter coordinates.

The tensor products, normalization and endpoint updates below are adapted from
DeepH-E3, commit ``036bf95b996052ef778c5c38e3c54c2ac0e2b591``:
https://github.com/Xiaoxun-Gong/DeepH-E3
The Gaussian basis originates in SchNetPack.  The equations, parameter creation
order, tensor-product instructions and default initialization are retained so
that mapped weights and training gradients can be compared directly.  These
are transformations, not independent model architectures.

Copyright (c) 2023 Xiaoxun-Gong
Copyright (c) 2018 Kristof Schütt, Michael Gastegger, Pan Kessel, Kim Nicoli
Copyright (c) 2018, the respective SchNetPack contributors

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from e3nn.nn import Extract, Gate
from e3nn.o3 import FullyConnectedTensorProduct, Irrep, Irreps, Linear, TensorProduct


def _tensor_product_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    target = Irrep(ir_out)
    return any(
        target in ir1 * ir2
        for _, ir1 in Irreps(irreps_in1).simplify()
        for _, ir2 in Irreps(irreps_in2).simplify()
    )


def make_tensor_product_gate(
    irreps_in1,
    irreps_in2,
    irreps_out,
    act: Mapping[int, Callable] | None = None,
    act_gates: Mapping[int, Callable] | None = None,
) -> Gate:
    """Select reachable scalars/tensors and construct parity-preserving gates.

    The returned gate's ``irreps_in`` describes the tensor-product output,
    including extra scalar gate channels.  Its ``irreps_out`` can be narrower
    than the requested irreps when a tensor-product path does not exist.
    """
    if act is None:
        act = {1: F.silu, -1: torch.tanh}
    if act_gates is None:
        act_gates = {1: torch.sigmoid, -1: torch.tanh}
    requested = Irreps(irreps_out)
    scalars = Irreps(
        [
            (mul, ir)
            for mul, ir in requested
            if ir.l == 0 and _tensor_product_path_exists(irreps_in1, irreps_in2, ir)
        ]
    ).simplify()
    tensors = Irreps(
        [
            (mul, ir)
            for mul, ir in requested
            if ir.l > 0 and _tensor_product_path_exists(irreps_in1, irreps_in2, ir)
        ]
    ).simplify()
    gate_irrep = None
    if tensors.dim:
        if _tensor_product_path_exists(irreps_in1, irreps_in2, "0e"):
            gate_irrep = "0e"
        elif _tensor_product_path_exists(irreps_in1, irreps_in2, "0o"):
            gate_irrep = "0o"
            warnings.warn("Using odd representations as gates", stacklevel=2)
        else:
            raise ValueError(
                f"{irreps_in1} times {irreps_in2} cannot produce scalar gates "
                f"for {tensors}."
            )
    gates = Irreps([(mul, gate_irrep) for mul, _ in tensors]).simplify()
    return Gate(
        scalars,
        [act[ir.p] for _, ir in scalars],
        gates,
        [act_gates[ir.p] for _, ir in gates],
        tensors,
    )


class GaussianBasis(nn.Module):
    """Unnormalized Gaussian features with one-grid-spacing standard deviation.

    ``distances`` has arbitrary leading shape; the feature axis is appended.
    Parameters use the current default dtype, as in the reference.  There is
    no cutoff envelope or implicit ``sqrt(n_gaussians)`` normalization.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        n_gaussians: int = 50,
        centered: bool = False,
        trainable: bool = False,
    ):
        super().__init__()
        if n_gaussians < 2:
            raise ValueError("GaussianBasis requires at least two centers.")
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.Tensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered
        self.out_dim = n_gaussians

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if self.centered:
            coefficient = -0.5 / self.offsets.pow(2)
            difference = distances[..., None]
        else:
            coefficient = -0.5 / self.width.pow(2)
            difference = distances[..., None] - self.offsets
        return torch.exp(coefficient * difference.pow(2))


class FactorizedTensorProduct(nn.Module):
    """CG tensor product with two learned channel factors per allowed path.

    Each path has weights ``W1[u,w] * W2[v,w]`` rather than an independent
    dense ``W[u,v,w]``.  Keeping both factors is essential for optimizer and
    initialization equivalence.  All allowed paths follow input1, input2,
    output order and use e3nn's ``uvw`` connection convention.
    """

    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        super().__init__()
        if kwargs.pop("internal_weights", False):
            raise ValueError("FactorizedTensorProduct supplies its own weights.")
        if not kwargs.pop("shared_weights", True):
            raise ValueError("FactorizedTensorProduct requires shared weights.")
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        instructions, weights1, weights2 = [], [], []
        for i1, (mul1, ir1) in enumerate(self.irreps_in1):
            for i2, (mul2, ir2) in enumerate(self.irreps_in2):
                for iout, (mulout, irout) in enumerate(self.irreps_out):
                    if irout in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mulout)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mulout)))
                        instructions.append((i1, i2, iout, "uvw", True, 1.0))
        self.tp = TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=True,
            **kwargs,
        )
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        factors = [
            (w1[:, None, :] * w2[None, :, :]).reshape(-1)
            for w1, w2 in zip(self.weights1, self.weights2)
        ]
        weights = torch.cat(factors) if factors else x1.new_empty(0)
        return self.tp(x1, x2, weights)


class SymmetricFactorizedTensorProduct(nn.Module):
    """Self tensor product using only input-irrep pairs ``i1 <= i2``."""

    def __init__(self, irreps_in, irreps_out, **kwargs):
        super().__init__()
        if kwargs.pop("internal_weights", False):
            raise ValueError("SymmetricFactorizedTensorProduct supplies weights.")
        if not kwargs.pop("shared_weights", True):
            raise ValueError(
                "SymmetricFactorizedTensorProduct requires shared weights."
            )
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        instructions, weights1, weights2 = [], [], []
        for i1, (mul1, ir1) in enumerate(self.irreps_in):
            for i2 in range(i1, len(self.irreps_in)):
                mul2, ir2 = self.irreps_in[i2]
                for iout, (mulout, irout) in enumerate(self.irreps_out):
                    if irout in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mulout)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mulout)))
                        instructions.append((i1, i2, iout, "uvw", True, 1.0))
        self.tp = TensorProduct(
            self.irreps_in,
            self.irreps_in,
            self.irreps_out,
            instructions,
            internal_weights=False,
            shared_weights=True,
            **kwargs,
        )
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factors = [
            (w1[:, None, :] * w2[None, :, :]).reshape(-1)
            for w1, w2 in zip(self.weights1, self.weights2)
        ]
        weights = torch.cat(factors) if factors else x.new_empty(0)
        return self.tp(x, x, weights)


class GraphIrrepNorm(nn.Module):
    """Normalize each irrep using rows and channels within each graph.

    ``x`` is ``[rows, irreps_in.dim]`` and ``batch`` assigns every row to a
    graph.  With the reference defaults, means are subtracted for all irreps,
    but variance normalization is applied only to ``l=0``.  The denominator
    is ``sqrt(variance) + eps``.  Only even scalars receive learned biases.
    This is graph normalization, not independent normalization of each atom.
    """

    def __init__(
        self,
        irreps_in,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
        subtract_mean: bool = True,
        divide_norm: bool = False,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = self.irreps_in
        self.eps = eps
        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        if normalization not in {"component", "norm"}:
            raise ValueError("normalization must be 'component' or 'norm'.")
        self.normalization = normalization
        if affine:
            ib, iw = 0, 0
            self.bias_slices, self.weight_slices = [], []
            for mul, ir in self.irreps_in:
                if ir.is_scalar():
                    self.bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    self.bias_slices.append(None)
                self.weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones(iw))
            self.bias = nn.Parameter(torch.zeros(ib))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.weight is not None:
                self.weight.fill_(1)
            if self.bias is not None:
                self.bias.fill_(0)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if not batch.numel():
            return x
        batch_size = int(batch.max()) + 1
        counts = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch))
        counts = counts.clamp_min(1).to(dtype=x.dtype)
        result = []
        offset = 0
        for index, (mul, ir) in enumerate(self.irreps_in):
            field = x[:, offset : offset + mul * ir.dim].reshape(-1, mul, ir.dim)
            if self.subtract_mean or ir.l == 0:
                mean = (
                    scatter(
                        field, batch, dim=0, dim_size=batch_size, reduce="add"
                    ).mean(dim=1, keepdim=True)
                    / counts[:, None, None]
                )
                field = field - mean[batch]
            if self.divide_norm or ir.l == 0:
                variance = scatter(
                    field.abs().pow(2),
                    batch,
                    dim=0,
                    dim_size=batch_size,
                    reduce="mean",
                ).mean(dim=[1, 2], keepdim=True)
                if self.normalization == "norm":
                    variance = variance * ir.dim
                field = field / (variance.sqrt()[batch] + self.eps)
            if self.weight is not None:
                field = field * self.weight[self.weight_slices[index]][None, :, None]
            if self.bias is not None and ir.is_scalar():
                field = field + self.bias[self.bias_slices[index]][None, :, None]
            result.append(field.reshape(-1, mul * ir.dim))
            offset += mul * ir.dim
        return torch.cat(result, dim=-1)


class IrrepChannelScale:
    """Multiply each irrep copy by one invariant coefficient per row."""

    def __init__(self, irreps_in):
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = self.irreps_in
        self.len_weight = self.irreps_in.num_irreps

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        result = []
        offset = weight_offset = 0
        for mul, ir in self.irreps_in:
            field = x[:, offset : offset + mul * ir.dim].reshape(-1, mul, ir.dim)
            field = field * weight[:, weight_offset : weight_offset + mul, None]
            result.append(field.reshape(-1, mul * ir.dim))
            offset += mul * ir.dim
            weight_offset += mul
        return torch.cat(result, dim=-1)


class IrrepResidual(nn.Module):
    """Add old and new states, projecting the old state only if irreps differ."""

    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.sc = (
            None
            if self.irreps_in == self.irreps_out
            else Linear(self.irreps_in, self.irreps_out)
        )

    def forward(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if self.sc is not None:
            old = self.sc(old)
        return old + new


class IrrepPermutation(nn.Module):
    """Sort irrep fields without mixing or rescaling their coordinates."""

    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        sorted_irreps = self.irreps_in.sort()
        self.extr = Extract(
            self.irreps_in,
            [((mul, ir),) for mul, ir in sorted_irreps.irreps],
            [(i,) for i in sorted_irreps.inv],
        )
        self.extr_inv = Extract(
            sorted_irreps.irreps,
            [((mul, ir),) for mul, ir in self.irreps_in],
            [(i,) for i in sorted_irreps.p],
        )
        self.irreps_out = sorted_irreps.irreps.simplify()

    def forward(self, x):
        return torch.cat(self.extr(x), dim=-1)

    def inverse(self, x):
        return torch.cat(self.extr_inv(x), dim=-1)


def _graph_norm(norm, irreps):
    if not norm or norm in {"none", "off"}:
        return None
    if norm not in {"graph_irrep", "e3LayerNorm"}:
        raise ValueError(f"Unknown graph irrep normalization: {norm!r}.")
    return GraphIrrepNorm(irreps)


class GatedRadialTensorProduct(nn.Module):
    """Factorized TP, optional gate, then invariant radial channel scaling.

    Unlike a tensor product whose *path weights* come from the radial MLP,
    this transformation applies the MLP's channel scales *after* its gate.
    Optional normalization acts on the resulting edge rows per graph.
    """

    def __init__(
        self,
        fc_len_in,
        irreps_in1,
        irreps_in2,
        irreps_out,
        norm="",
        nonlin=True,
        act=None,
        act_gates=None,
        radial_hidden: Sequence[int] = (64, 64),
    ):
        super().__init__()
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        requested = Irreps(irreps_out)
        self.nonlin = None
        if nonlin:
            self.nonlin = make_tensor_product_gate(
                self.irreps_in1, self.irreps_in2, requested, act, act_gates
            )
            product_out = self.nonlin.irreps_in
        else:
            product_out = Irreps(
                [
                    (mul, ir)
                    for mul, ir in requested
                    if _tensor_product_path_exists(self.irreps_in1, self.irreps_in2, ir)
                ]
            )
        self.tp = FactorizedTensorProduct(self.irreps_in1, self.irreps_in2, product_out)
        self.cfconv = IrrepChannelScale(
            self.nonlin.irreps_out if self.nonlin is not None else product_out
        )
        self.irreps_out = self.cfconv.irreps_out
        layers = []
        widths = [fc_len_in, *radial_hidden, self.cfconv.len_weight]
        for index, (width_in, width_out) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(width_in, width_out))
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
        self.fc = nn.Sequential(*layers)
        self.norm = _graph_norm(norm, self.irreps_out)

    def forward(self, fea_in1, fea_in2, fea_weight, batch_edge):
        value = self.tp(fea_in1, fea_in2)
        if self.nonlin is not None:
            value = self.nonlin(value)
        value = self.cfconv(value, self.fc(fea_weight))
        if self.norm is not None:
            value = self.norm(value, batch_edge)
        return value


class EndpointTensorProductNodeUpdate(nn.Module):
    """Aggregate endpoint/edge tensor-product messages onto ``edge_index[0]``.

    All nodes/edges are directed and ``batch`` has one graph index per node.
    ``edge_sh`` contains the configured angular features; ``edge_length_embedded``
    contains invariant radial features.  With default ``nonlin=False``, gating
    occurs inside the message convolution before aggregation.  With
    ``nonlin=True``, gating follows aggregation and the species self-connection.
    """

    def __init__(
        self,
        num_species,
        fc_len_in,
        irreps_sh,
        irreps_in_node,
        irreps_out_node,
        irreps_in_edge,
        act=None,
        act_gates=None,
        use_selftp=False,
        use_sc=True,
        concat=True,
        only_ij=False,
        nonlin=False,
        norm="graph_irrep",
        if_sort_irreps=False,
        aggregation="sum",
        residual=True,
    ):
        super().__init__()
        node_in, angular, node_out, edge_in = map(
            Irreps, (irreps_in_node, irreps_sh, irreps_out_node, irreps_in_edge)
        )
        product_in = node_in + node_in + edge_in if concat else node_in
        if concat and if_sort_irreps:
            self.sort = IrrepPermutation(product_in)
            product_in = self.sort.irreps_out
        self.lin_pre = Linear(node_in, node_in, biases=True)
        self.nonlin = None
        if nonlin:
            self.nonlin = make_tensor_product_gate(
                product_in, angular, node_out, act, act_gates
            )
            convolution_out = self.nonlin.irreps_in
        else:
            convolution_out = node_out
        self.conv = GatedRadialTensorProduct(
            fc_len_in,
            product_in,
            angular,
            convolution_out,
            nonlin=not nonlin,
            act=act,
            act_gates=act_gates,
        )
        self.lin_post = Linear(self.conv.irreps_out, self.conv.irreps_out, biases=True)
        self.irreps_out = (
            self.nonlin.irreps_out if self.nonlin is not None else self.conv.irreps_out
        )
        self.sc = (
            FullyConnectedTensorProduct(
                node_in, f"{num_species}x0e", self.conv.irreps_out
            )
            if use_sc
            else None
        )
        self.norm = _graph_norm(norm, self.irreps_out)
        self.skip_connect = IrrepResidual(node_in, self.irreps_out)
        self.self_tp = (
            SymmetricFactorizedTensorProduct(self.irreps_out, self.irreps_out)
            if use_selftp
            else None
        )
        self.irreps_in_node = node_in
        self.irreps_in_edge = edge_in
        self.use_sc = use_sc
        self.concat = concat
        self.only_ij = only_ij
        self.if_sort_irreps = if_sort_irreps
        from net.transforms.composition import IndexedAggregation

        self.aggregation = IndexedAggregation(aggregation, receiver=0)
        self.use_residual = residual

    def forward(
        self,
        node_fea,
        node_one_hot,
        edge_sh,
        edge_fea,
        edge_length_embedded,
        edge_index,
        batch,
        selfloop_edge=None,
        edge_length=None,
    ):
        old = node_fea
        connection = self.sc(node_fea, node_one_hot) if self.sc is not None else None
        nodes = self.lin_pre(node_fea)
        index_i, index_j = edge_index
        if self.concat:
            value = torch.cat([nodes[index_i], nodes[index_j], edge_fea], dim=-1)
            if self.if_sort_irreps:
                value = self.sort(value)
        else:
            value = nodes[index_j]
        messages = self.conv(value, edge_sh, edge_length_embedded, batch[index_i])
        nodes = self.aggregation(messages, edge_index, old)
        if self.only_ij:
            if selfloop_edge is None:
                raise ValueError(
                    "only_ij requires an explicit zero-shift self-edge mask."
                )
            nodes = nodes + scatter(
                messages[~selfloop_edge],
                index_j[~selfloop_edge],
                dim=0,
                dim_size=old.shape[0],
                reduce="add",
            )
        nodes = self.lin_post(nodes)
        if connection is not None:
            nodes = nodes + connection
        if self.nonlin is not None:
            nodes = self.nonlin(nodes)
        if self.norm is not None:
            nodes = self.norm(nodes, batch)
        if self.use_residual:
            nodes = self.skip_connect(old, nodes)
        if self.self_tp is not None:
            nodes = self.self_tp(nodes)
        return nodes


class EndpointTensorProductEdgeUpdate(nn.Module):
    """Update each directed edge from its two node states and previous edge.

    The node input is normally the newly updated state.  This component does
    not symmetrize reverse edges or impose a matrix constraint.  Default
    parameter creation and gating/normalization/residual order match the
    endpoint tensor-product update in the pinned reference.
    """

    def __init__(
        self,
        num_species,
        fc_len_in,
        irreps_sh,
        irreps_in_node,
        irreps_in_edge,
        irreps_out_edge,
        act=None,
        act_gates=None,
        use_selftp=False,
        use_sc=True,
        init_edge=False,
        nonlin=False,
        norm="graph_irrep",
        if_sort_irreps=False,
        residual=True,
    ):
        super().__init__()
        if init_edge and use_sc:
            raise ValueError(
                "init_edge=True cannot use a previous-edge self-connection."
            )
        node_in, angular, edge_in, edge_out = map(
            Irreps, (irreps_in_node, irreps_sh, irreps_in_edge, irreps_out_edge)
        )
        product_in = node_in + node_in + edge_in
        if if_sort_irreps:
            self.sort = IrrepPermutation(product_in)
            product_in = self.sort.irreps_out
        self.lin_pre = Linear(edge_in, edge_in, biases=True)
        self.nonlin = None
        if nonlin:
            self.nonlin = make_tensor_product_gate(
                product_in, angular, edge_out, act, act_gates
            )
            convolution_out = self.nonlin.irreps_in
        else:
            convolution_out = edge_out
        self.conv = GatedRadialTensorProduct(
            fc_len_in,
            product_in,
            angular,
            convolution_out,
            nonlin=not nonlin,
            act=act,
            act_gates=act_gates,
        )
        self.lin_post = Linear(self.conv.irreps_out, self.conv.irreps_out, biases=True)
        self.sc = (
            FullyConnectedTensorProduct(
                edge_in, f"{num_species**2}x0e", self.conv.irreps_out
            )
            if use_sc
            else None
        )
        self.irreps_out = (
            self.nonlin.irreps_out if self.nonlin is not None else self.conv.irreps_out
        )
        self.norm = _graph_norm(norm, self.irreps_out)
        self.skip_connect = IrrepResidual(edge_in, self.irreps_out)
        self.self_tp = (
            SymmetricFactorizedTensorProduct(self.irreps_out, self.irreps_out)
            if use_selftp
            else None
        )
        self.irreps_in_node = node_in
        self.irreps_in_edge = edge_in
        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.use_residual = residual

    def forward(
        self,
        node_fea,
        edge_one_hot,
        edge_sh,
        edge_fea,
        edge_length_embedded,
        edge_index,
        batch,
    ):
        old = edge_fea
        connection = None
        if not self.init_edge:
            if self.sc is not None:
                connection = self.sc(edge_fea, edge_one_hot)
            edge_fea = self.lin_pre(edge_fea)
        index_i, index_j = edge_index
        value = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
        if self.if_sort_irreps:
            value = self.sort(value)
        edges = self.conv(value, edge_sh, edge_length_embedded, batch[index_i])
        edges = self.lin_post(edges)
        if connection is not None:
            edges = edges + connection
        if self.nonlin is not None:
            edges = self.nonlin(edges)
        if self.norm is not None:
            edges = self.norm(edges, batch[index_i])
        if not self.init_edge and self.use_residual:
            edges = self.skip_connect(old, edges)
        if self.self_tp is not None:
            edges = self.self_tp(edges)
        return edges
