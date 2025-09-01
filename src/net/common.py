"""
common.py
=========

Light-weight utilities that are shared by *all* network sub-modules:

* Hyper-parameter dataclass : :class:`Config`
* Hidden-width Irreps constructor : :func:`build_hidden_irreps`
* A flexible N-layer radial MLP   : :class:`RadialMLP`

**No activation / norm helpers live here anymore** – those have been moved
to :pymod:`net.activations` so we avoid a circular import between files.
"""

from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch
from torch import nn
from e3nn.o3 import Irreps, Linear
from omegaconf import OmegaConf

# ════════════════════════════════════════════════════════════════════════
# 1.  Hyper-parameters
# ════════════════════════════════════════════════════════════════════════
OmegaConf.register_new_resolver("torch_dtype", lambda x: str(x).split(".")[-1])
OmegaConf.register_new_resolver("torch_device", lambda x: str(x))


@dataclass(slots=True)
class Config:
    """
    Network hyperparameters controlling representations, model depth,
    nonlinearity, regularization, radial basis, output heads, and loss weighting.
    """

    # radii
    cutoff_gnn: float = 5.0
    cutoff_matrix: float = 8.0

    # -------------- representation shape --------------------------------
    l_max_gnn: int = 2
    l_max_matrix: int = 4
    hidden_base_dim: int = 64  # multiplicity at ℓ = 0
    edge_type_emb_dim: int = 32  # edge type embedding size
    # node_type_emb_dim: int = 32  # node type embedding size

    # -------------- depth / topology ------------------------------------
    num_layers_gnn: int = 2
    num_layers_matrix: int = 1

    # -------------- model variants --------------------------------------
    edge_update_node_combine: str = "concat"  # "sum" | "concat"
    edge_update_linear: str = "post"  # "pre" | "post" | "none"
    edge_update: str = "tensor_product"  # "tensor_product" | "concat" | "replace"
    edge_update_residual: bool = True  # use residual connections in edge update

    node_update_message_agg: str = "attention"  # "attention" | "sum"
    node_update: str = "tensor_product"  # "tensor_product" | "concat" | "replace"
    node_update_residual: bool = True  # use residual connections in node update

    head_use_mlp_log_scale: bool = True  # whether to use MLP log scaling in the head

    # -------------- MLP Layers Configuration -----------------------------
    edge_update_pre_lin_mlp_n_layers: int = 1
    edge_update_post_lin_mlp_n_layers: int = 1
    node_update_pre_lin_mlp_n_layers: int = 1
    node_update_attention_mlp_n_layers: int = 1
    node_update_post_lin_mlp_n_layers: int = 1
    head_trunk_mlp_n_layers: int = 3
    head_last_mlp_n_layers: int = 2
    head_final_proj_mlp_n_layers: int = 1
    head_log_scale_mlp_n_layers: int = 1

    # -------------- non-linearity & norm --------------------------------
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
    smoke_test: bool = False
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.25
    lr_scheduler_patience: int = 10
    lr_scheduler_min_lr: float = 1e-7

    # -------------- regularisation --------------------------------------
    dropout: float = 0.0  # dropout on *all* irrep coefficients
    l1_reg_coef: float = 0.0
    l2_reg_coef: float = 0.0
    grad_clip_val: float | None = None

    # -------------- radial basis ----------------------------------------
    n_radial: int = 64
    radial_layers: Sequence[int] = field(
        default_factory=lambda: (128,)
    )  # e.g. (128,) → 2-layer MLP
    share_radial: bool = True

    # -------------- output head ----------------------------------------
    # --------- additional outputs --------------------------------------
    enable_forces: bool = False
    enable_stress: bool = False
    enable_energy: bool = True
    enable_num_electrons: bool = True

    # -------------- training targets -----------------------------------
    train_target: str = "irreps"  # "irreps" | "matrix"
    scheduler_target: str = "val_loss_matrix"
    train_on_forces: bool = False
    train_on_stress: bool = False
    train_on_energy: bool = True
    train_on_num_electrons: bool = True

    # -------------- loss weighting --------------------------------------
    loss_coef_energy: float = 1e-5
    loss_coef_num_electrons: float = 1e-5
    loss_coef_forces: float = 0.0
    loss_coef_stress: float = 0.0

    # -------------- logging ---------------------------------------------
    run_name: str = "mandala-run"
    verbosity: int = 2
    bench_verbosity: int = 1
    log_activation_mag: bool = False
    wandb_project: str | None = None
    log_every_n_steps: int = 1
    log_on_step: bool = False  # log metrics on step, not just epoch
    log_on_epoch: bool = True  # log metrics on epoch

    # -------------- misc ------------------------------------------------
    pedantic: bool = False  # enable strict checks on input data
    dtype: torch.dtype = torch.float32  # default data type for all layers
    device: str = "cpu"  # default device for all layers
    gpus: int = 0  # number of GPUs
    num_workers: int = 0  # DataLoader workers, 0 for no parallelism
    save_dir: str = "checkpoints"  # directory to save model checkpoints
    log_model: bool = False  # whether to log the model to WandB

    # ----------------- caching ------------------------------------------
    cache_root: str | None = None  # path to cache directory, if any
    seed: int = 42  # random seed for reproducibility

    # -------------------- hyperopt --------------------------------------
    tune: str | None = None  # hyperparameter tuning (e.g. "ray", "wandb")


def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)

    return dtype


# ════════════════════════════════════════════════════════════════════════
# 2.  Hidden irreps auto-builder  (cached – deterministic)
# ════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=None)
def build_hidden_irreps(l_max_gnn: int, base_dim: int) -> Irreps:
    """
    Create `Irreps` with multiplicity halved for every ℓ > 0
    and *both* parity channels present.

    Example  (base_dim=32, l_max_gnn=2) ::

        32x0e + 32x0o + 16x1e + 16x1o + 8x2e + 8x2o
    """
    parts: List[str] = []
    for ell in range(l_max_gnn + 1):
        mul = max(base_dim // (2**ell), 1)
        parts.append(f"{mul}x{ell}e")
        parts.append(f"{mul}x{ell}o")
    return Irreps("+".join(parts)).simplify()


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
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("E3MLP must have at least 1 layer.")

        from net.activations import make_nonlinearity  # Local import

        self.irreps_in = input_irreps
        self.irreps_hidden = hidden_irreps
        self.irreps_out = output_irreps

        layers = []
        current_irreps = input_irreps

        # All layers except the last one map to hidden_irreps
        for _ in range(num_layers - 1):
            layers.append(Linear(current_irreps, hidden_irreps))
            layers.append(make_nonlinearity(hidden_irreps, cfg))
            current_irreps = hidden_irreps

        # The final layer maps to the output_irreps
        layers.append(Linear(current_irreps, output_irreps))

        if activate_last:
            layers.append(make_nonlinearity(output_irreps, cfg))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
