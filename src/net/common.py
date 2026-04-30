"""
common.py
=========

Light-weight utilities that are shared by *all* network sub-modules:

* Hyper-parameter dataclass : :class:`Config`
* Hidden-width Irreps constructor : :func:`build_hidden_irreps`
* A flexible N-layer radial MLP   : :class:`RadialMLP`

**No activation / norm helpers live here anymore** - those have been moved
to :pymod:`net.activations` so we avoid a circular import between files.
"""

from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch
from torch import nn
from e3nn.o3 import Irreps, Linear, TensorProduct
from omegaconf import OmegaConf

# ════════════════════════════════════════════════════════════════════════
# 1.  Hyper-parameters
# ════════════════════════════════════════════════════════════════════════
if not OmegaConf.has_resolver("torch_dtype"):
    OmegaConf.register_new_resolver("torch_dtype", lambda x: str(x).split(".")[-1])
if not OmegaConf.has_resolver("torch_device"):
    OmegaConf.register_new_resolver("torch_device", lambda x: str(x))


@dataclass(slots=True)
class Config:
    """
    Network hyperparameters controlling representations, model depth,
    nonlinearity, regularization, radial basis, output heads, and loss weighting.
    """

    # radii (unified cutoff - no small/large graph split)
    cutoff_radius: float = 7.0

    # -------------- representation shape --------------------------------
    l_max: int = 4
    hidden_base_dim: int = 64  # multiplicity at ℓ = 0
    hidden_irreps: str | None = None  # explicit hidden irreps override
    edge_type_emb_dim: int = 32  # edge type embedding size
    emb_use_odd_features: bool = True  # use odd parity
    edge_encoder_style: str = "rich"  # "rich" | "distance"
    edge_encoder_use_sh_tensor_square: bool = False
    e3layernorm: bool = True

    # node_type_emb_dim: int = 32  # node type embedding size

    # -------------- depth / topology ------------------------------------
    num_layers_gnn: int = 2  # unified number of message-passing layers

    # -------------- model variants --------------------------------------
    # TensorProduct type: "separate_weight" | "fully_connected"
    tp_type: str = "separate_weight"

    # use self-connection (element-specific features)
    use_self_connection: bool = True

    edge_update_node_combine: str = "concat"  # "concat" | "sum" | "tensor_product"
    edge_update_residual: bool = True  # use residual connections in edge update

    node_update_message_agg: str = "sum"  # "sum" | "average" | "attention"
    node_update_attention_scalar_dim: int = 64
    node_update_attention_heads: int = 4
    node_update_residual: bool = True  # use residual connections in node update

    head_use_mlp_log_scale: bool = False  # whether to use MLP log scaling in the head

    neck_depth: int = 1
    internal_e3mlp_layers: int = 0
    head_e3mlp_layers: int = 1
    head_use_node_embeddings_for_self_edges: bool = True
    separate_shifted_self: bool = False
    head_use_tensor_square: bool = False
    head_diag_output_scale: float = 1.0
    head_offdiag_output_scale: float = 1.0

    head_log_scale_mlp_n_layers: int = 1

    # -------------- non-linearity & norm --------------------------------
    e3mlp_variant: str = "basic"  # "basic" or study-style variants
    internal_e3mlp_variant: str | None = None
    head_e3mlp_variant: str | None = None
    e3mlp_output_scale: float = 1.0
    e3mlp_weight_init_scale: float = 1.0
    e3mlp_residual_scale: float = 0.25
    e3mlp_pre_norm: bool = False
    e3mlp_norm_eps: float = 1e-8
    e3mlp_film_hidden_dim: int = 128
    activation_odd_scalar: str = "tanh"
    activation_odd_gate: str = "tanh"
    nonlin_kind: str = (
        "normact"  # "normact" | "s2act" | "gate_scalars_mlp" | "gate_magnitudes"
    )
    activation_scalar: str = "leakyrelu"
    activation_gate: str = "softplus"
    s2act_res: int = 128
    norm_kind: str = "component"  # for NormActivation: "component" | "norm"

    # ---------------------- training ------------------------------------
    lr: float = 3e-4
    max_epochs: int = 100
    batch_size: int = 1
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 60
    lr_scheduler_min_lr: float = 1e-8
    lr_scheduler_target: str = "val/loss_total"
    revert_on_spike: bool = True
    revert_monitor: str | None = None
    revert_decay_patience: int = 20
    revert_decay_rate: float = 0.8
    revert_spike_factor: float = 2.0
    # -------------- regularisation --------------------------------------
    dropout: float = 0.0  # dropout on *all* irrep coefficients
    l1_reg_coef: float = 0.0
    l2_reg_coef: float = 0.0
    init_weights_factor: float = 1.0
    grad_clip_val: float | None = 0.5
    accumulate_grad_batches: int = 1  # gradient accumulation steps

    # -------------- radial basis ----------------------------------------
    n_radial: int = 64
    radial_layers: Sequence[int] = field(
        default_factory=lambda: (128,)
    )  # e.g. (128,) -> 2-layer MLP

    # -------------- output head ----------------------------------------
    # --------- additional outputs --------------------------------------
    enable_forces: bool = False
    enable_stress: bool = False
    enable_energy: bool = True
    enable_num_electrons: bool = True

    # -------------- training targets -----------------------------------
    train_target: str = "matrix"  # "irreps" | "matrix"
    partial_train: str | None = None  # None | "diag" | "shifted_self" | "offdiag"
    train_on_forces: bool = False
    train_on_stress: bool = False
    train_on_energy: bool = True
    train_on_num_electrons: bool = True
    matrix_targets: list = field(
        default_factory=lambda: ["hamiltonian", "overlap", "density"]
    )
    train_observables_on_gt: bool = False
    symmetrize_output: bool = True  # symmetrize matrix outputs
    symmetrize_hamiltonian_targets: bool = True
    rescale_density_to_num_electrons: bool = False

    # -------------- loss weighting --------------------------------------
    loss_l1_fraction: float = 0.0  # 0.0 for L2, 1.0 for L1
    loss_coef_observables: float = 1e-5
    loss_coef_forces: float = 0.0
    loss_coef_stress: float = 0.0

    # -------------- logging ---------------------------------------------
    run_name: str = "mandala-run"
    verbosity: int = 1
    bench_verbosity: int = 1
    log_partial_gt_observables: bool = False
    log_per_irrep_metrics: bool = False
    print_per_irrep_metrics: bool = False
    log_per_irrep_images: bool = False
    log_activation_mag: bool = False
    wandb_project: str | None = None
    log_every_n_steps: int = 1
    log_on_step: bool = False  # log metrics on step, not just epoch
    log_on_epoch: bool = True  # log metrics on epoch
    log_data: bool = False
    log_forward: bool = False
    benchmark: bool = True
    log_interval: int = 1
    adaptive_log_interval: bool = False
    video_max_atoms: int | None = 6

    # -------------- misc ------------------------------------------------
    safety_checks: bool = False  # enable strict checks on input data
    dtype: torch.dtype = torch.float32  # default data type for all layers
    device: str = "cpu"  # default device for all layers
    gpus: int = 0  # number of GPUs
    num_workers: int | None = (
        None  # auto: allocated cores - 1, or explicit total workers
    )
    save_dir: str = "checkpoints"  # directory to save model checkpoints
    log_model: bool = False  # whether to log the model to WandB

    # ----------------- caching ------------------------------------------
    snapshot_cache_dir: str | None = None  # raw Snapshot .pt cache, if any
    dataset_device: str | None = None  # keep processed dataset on this device
    shuffle_snapshot_load_order: bool = True  # randomize cache warmup order
    seed: int = 42  # random seed for reproducibility
    precompute_edge_features: bool = True  # precompute edge features
    radial_embedding_scale: str = "none"
    apply_cutoff_to_targets: bool = True
    require_exact_edge_match: bool = True

    # -------------------- hyperopt --------------------------------------
    tune: str | None = None  # hyperparameter tuning (e.g. "ray", "wandb")
    train_on_irrep_parts: bool = False

    # --- DeepH-E3 Specific Config ---
    num_block: int = 3
    r_max: float = 8.0
    num_basis: int = 128
    use_sc: bool = True
    use_sbf: bool = False


def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)

    return dtype


# ════════════════════════════════════════════════════════════════════════
# 2.  Hidden irreps auto-builder
# ════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=None)
def build_hidden_irreps(
    l_max: int, base_dim: int, use_odd_features: bool = True
) -> Irreps:
    """
    Create `Irreps` with multiplicity halved for every ℓ > 0
    and *both* parity channels present if `use_odd_features` is True (default).

    Example  (base_dim=32, l_max=2) ::

        32x0e + 32x0o + 16x1e + 16x1o + 8x2e + 8x2o
    """
    parts: List[str] = []
    for ell in range(l_max + 1):
        mul = max(base_dim // (2**ell), 1)
        parts.append(f"{mul}x{ell}e")
        if use_odd_features:
            parts.append(f"{mul}x{ell}o")
    return Irreps("+".join(parts)).simplify()


def resolve_hidden_irreps(cfg: Config) -> Irreps:
    if cfg.hidden_irreps is not None:
        return Irreps(cfg.hidden_irreps).simplify()
    return build_hidden_irreps(
        cfg.l_max,
        cfg.hidden_base_dim,
        cfg.emb_use_odd_features,
    )


# ════════════════════════════════════════════════════════════════════════
# 3.  Flexible radial MLP
# ════════════════════════════════════════════════════════════════════════
class RadialMLP(nn.Module):
    """
    Generic N-layer feed-forward network applied to the radial
    distance embedding (soft one-hot, Bessel basis, …).

    Parameters
    ----------
    in_dim
        Input size (n_radial from dataset).
    layers
        Sequence of **hidden layer widths**.  The final width
        *out_dim* is appended automatically if not already present.
    out_dim
        Output size produced per edge (number of scalar coefficients).
    act
        Name of scalar activation ("silu", "relu", ...).

    Example
    -------
    >>> mlp = RadialMLP(64, out_dim=32, layers=(128, 64))
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        layers: Sequence[int] = (128,),
        act: str = "silu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        from net.activations import scalar_activation  # local import

        layer_dims = list(layers)
        if not layer_dims or layer_dims[-1] != out_dim:
            layer_dims.append(out_dim)

        seq: List[nn.Module] = []
        prev = in_dim
        for hid in layer_dims:
            seq.append(nn.Linear(prev, hid, dtype=dtype))
            if hid != out_dim:  # no activation after last layer
                seq.append(scalar_activation(act))
            prev = hid
        self.net = nn.Sequential(*seq)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def irreps_div(irreps: Irreps, n: int) -> Irreps:
    """
    Divides the multiplicities of all irreps in an Irreps object by n.
    Raises a ValueError if any multiplicity is not divisible by n.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    new_irreps_list = []
    for mul, ir in irreps:
        if mul % n != 0:
            raise ValueError(
                f"Multiplicity {mul} for irrep {ir} is not divisible by {n}."
            )
        new_irreps_list.append((mul // n, ir))

    return Irreps(new_irreps_list).simplify()


# ════════════════════════════════════════════════════════════════════════
# 4.  Irreps splitting helpers
# ════════════════════════════════════════════════════════════════════════
def split_in_half(
    activations: torch.Tensor, irreps: Irreps
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits activations into two halves along the multiplicity dimension.
    Requires all multiplicities in the Irreps to be even.
    """
    batch_dim = activations.shape[0]
    split1_list, split2_list = [], []
    current_dim = 0

    for mul, ir in irreps:
        if mul % 2 != 0:
            raise ValueError(
                f"Multiplicity {mul} for irrep {ir} is not even, cannot split in half."
            )
        half_mul = mul // 2
        slice_dim = mul * ir.dim
        act_slice = activations[:, current_dim : current_dim + slice_dim]

        reshaped_slice = act_slice.reshape(batch_dim, 2, half_mul, ir.dim)
        split1 = reshaped_slice[:, 0, :, :].reshape(batch_dim, half_mul * ir.dim)
        split2 = reshaped_slice[:, 1, :, :].reshape(batch_dim, half_mul * ir.dim)

        split1_list.append(split1)
        split2_list.append(split2)
        current_dim += slice_dim

    return torch.cat(split1_list, dim=1), torch.cat(split2_list, dim=1)


def split_into_three(
    activations: torch.Tensor, irreps: Irreps
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits activations into three parts with proportions (1/4, 1/4, 1/2).
    Requires all multiplicities in the Irreps to be divisible by 4.
    """
    batch_dim = activations.shape[0]
    split1_list, split2_list, split3_list = [], [], []
    current_dim = 0

    for mul, ir in irreps:
        if mul % 4 != 0:
            raise ValueError(
                f"Multiplicity {mul} for irrep {ir} is not divisible by 4."
            )
        quarter_mul = mul // 4
        half_mul = mul // 2
        slice_dim = mul * ir.dim
        act_slice = activations[:, current_dim : current_dim + slice_dim]

        reshaped_slice = act_slice.reshape(batch_dim, mul, ir.dim)
        s1 = reshaped_slice[:, :quarter_mul, :].reshape(batch_dim, quarter_mul * ir.dim)
        s2 = reshaped_slice[:, quarter_mul : 2 * quarter_mul, :].reshape(
            batch_dim, quarter_mul * ir.dim
        )
        s3 = reshaped_slice[:, 2 * quarter_mul :, :].reshape(
            batch_dim, half_mul * ir.dim
        )

        split1_list.append(s1)
        split2_list.append(s2)
        split3_list.append(s3)
        current_dim += slice_dim

    return (
        torch.cat(split1_list, dim=1),
        torch.cat(split2_list, dim=1),
        torch.cat(split3_list, dim=1),
    )


# ════════════════════════════════════════════════════════════════════════
# 4.  E(3) Equivariant MLP
# ════════════════════════════════════════════════════════════════════════
class E3MLP(nn.Module):
    """
    A generic N-layer E(3)-equivariant feed-forward network.

    This module is a sequence of e3nn.o3.Linear layers alternating with
    equivariant non-linearities.

    Parameters
    ----------
    input_irreps : e3nn.o3.Irreps
        Input irreducible representations.
    hidden_irreps : e3nn.o3.Irreps
        Irreducible representations for the hidden layers. This is ignored
        if num_layers is 1.
    output_irreps : e3nn.o3.Irreps
        Output irreducible representations.
    num_layers : int
        Total number of linear layers in the MLP. Must be >= 1.
    cfg : Config
        Configuration object for activation functions.
    activate_last : bool, optional
        Whether to apply a non-linearity after the final layer.
        Defaults to False.
    """

    def __init__(
        self,
        input_irreps: Irreps,
        hidden_irreps: Irreps,
        output_irreps: Irreps,
        num_layers: int,
        cfg: Config,
        activate_last: bool = False,
        *,
        variant: str | None = None,
        output_scale: float | None = None,
        weight_init_scale: float | None = None,
        residual_scale: float | None = None,
        pre_norm: bool | None = None,
        film_hidden_dim: int | None = None,
        post_scale: float = 1.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("E3MLP must have at least 1 layer.")

        self.irreps_in = input_irreps
        self.irreps_hidden = hidden_irreps
        self.irreps_out = output_irreps

        variant = (variant or cfg.e3mlp_variant).lower()
        if variant == "basic":
            from net.activations import make_nonlinearity  # Local import

            layers = []
            current_irreps = input_irreps

            for _ in range(num_layers - 1):
                layers.append(Linear(current_irreps, hidden_irreps))
                layers.append(make_nonlinearity(hidden_irreps, cfg))
                current_irreps = hidden_irreps

            layers.append(Linear(current_irreps, output_irreps))

            if activate_last:
                layers.append(make_nonlinearity(output_irreps, cfg))

            self.net = nn.Sequential(*layers)
            self.post_scale = float(post_scale)
            return

        from net.e3mlp_variants import (
            EquivariantRMSNorm,
            ScaledLinear,
            VariantConfig,
            make_variant_block,
        )

        variant_cfg = VariantConfig(
            output_scale=(
                cfg.e3mlp_output_scale if output_scale is None else output_scale
            ),
            weight_init_scale=(
                cfg.e3mlp_weight_init_scale
                if weight_init_scale is None
                else weight_init_scale
            ),
            residual_scale=(
                cfg.e3mlp_residual_scale if residual_scale is None else residual_scale
            ),
            scalar_activation=cfg.activation_scalar,
            odd_scalar_activation=cfg.activation_odd_scalar,
            gate_activation=cfg.activation_gate,
            odd_gate_activation=cfg.activation_odd_gate,
            film_hidden_dim=(
                cfg.e3mlp_film_hidden_dim
                if film_hidden_dim is None
                else film_hidden_dim
            ),
            pre_norm=(cfg.e3mlp_pre_norm if pre_norm is None else pre_norm),
            norm_eps=cfg.e3mlp_norm_eps,
        )

        def maybe_prenorm(irreps: Irreps, block: nn.Module) -> nn.Module:
            if not variant_cfg.pre_norm:
                return block
            return nn.Sequential(
                EquivariantRMSNorm(irreps, eps=variant_cfg.norm_eps),
                block,
            )

        layers = []
        current_irreps = input_irreps

        for _ in range(num_layers - 1):
            block = make_variant_block(
                variant,
                current_irreps,
                hidden_irreps,
                variant_cfg,
            )
            layers.append(maybe_prenorm(current_irreps, block))
            current_irreps = hidden_irreps

        final_linear = ScaledLinear(
            current_irreps,
            output_irreps,
            output_scale=variant_cfg.output_scale,
            weight_init_scale=variant_cfg.weight_init_scale,
        )
        layers.append(maybe_prenorm(current_irreps, final_linear))

        if activate_last:
            layers.append(
                maybe_prenorm(
                    output_irreps,
                    make_variant_block(
                        variant,
                        output_irreps,
                        output_irreps,
                        variant_cfg,
                    ),
                )
            )

        self.net = nn.Sequential(*layers)
        self.post_scale = float(post_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.post_scale != 1.0:
            y = y * self.post_scale
        return y


# ════════════════════════════════════════════════════════════════════════
# 5.  SeparateWeightTensorProduct (taken from the DeepH-E3 implementation)
# ════════════════════════════════════════════════════════════════════════
class SeparateWeightTensorProduct(nn.Module):
    """
    Tensor product with separate learnable weights for each input.

    From DeepH-E3: z_i = W'_{ij} x_j (x) W''_{ik} y_k

    This differs from FullyConnectedTensorProduct by having separate
    weight matrices for each input irrep, which can be more expressive.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        **kwargs,
    ):
        super().__init__()

        # Ensure proper usage
        if kwargs.pop("internal_weights", False):
            raise ValueError(
                "SeparateWeightTensorProduct requires internal_weights=False"
            )
        if not kwargs.pop("shared_weights", True):
            raise ValueError("SeparateWeightTensorProduct requires shared_weights=True")

        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        instr_tp = []
        weights1, weights2 = [], []

        for i1, (mul1, ir1) in enumerate(irreps_in1):
            for i2, (mul2, ir2) in enumerate(irreps_in2):
                for i_out, (mul_out, ir3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mul_out)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mul_out)))
                        instr_tp.append((i1, i2, i_out, "uvw", True, 1.0))

        self.tp = TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instr_tp,
            internal_weights=False,
            shared_weights=True,
            **kwargs,
        )

        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)

    def forward(self, x1, x2):
        """
        Compute tensor product with separate weights.

        Args:
            x1: Tensor of shape (batch, irreps_in1.dim)
            x2: Tensor of shape (batch, irreps_in2.dim)

        Returns:
            Tensor of shape (batch, irreps_out.dim)
        """
        if len(self.weights1) == 0:
            # No valid tensor product paths - return empty tensor
            batch_size = x1.shape[0]
            return torch.zeros(batch_size, 0, dtype=x1.dtype, device=x1.device)

        weights = []
        for weight1, weight2 in zip(self.weights1, self.weights2):
            # Outer product of weights: (mul1, mul_out) (x) (mul2, mul_out)
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))
        weights = torch.cat(weights)
        return self.tp(x1, x2, weights)
