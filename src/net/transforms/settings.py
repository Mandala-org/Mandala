"""Activation and MLP settings shared by the stages that build equivariant MLPs.

Stages expose these as ordinary options with short names; ``layer_settings``
turns them into the settings object the MLP and nonlinearity factories read.
"""

from __future__ import annotations

import torch

from net.registry import Option

SCALAR_ACTIVATIONS = (
    "silu",
    "relu",
    "gelu",
    "tanh",
    "sigmoid",
    "leakyrelu",
    "softplus",
    "softsign",
)
MLP_VARIANTS = (
    "basic",
    "normact",
    "gate",
    "gatemagnitudes",
    "film",
    "resnormact",
    "resgatemagnitudes",
    "bilinear",
)
NONLINEARITIES = ("normact", "s2act", "gate_scalars_mlp", "gate_magnitudes")

ACTIVATION_OPTIONS = (
    Option(
        "scalar_activation",
        "choice",
        "leakyrelu",
        choices=SCALAR_ACTIVATIONS,
        doc="Even scalars",
    ),
    Option(
        "gate_activation",
        "choice",
        "softplus",
        choices=SCALAR_ACTIVATIONS,
        doc="Gates of non-scalars",
    ),
    Option(
        "odd_scalar_activation",
        "choice",
        "tanh",
        choices=SCALAR_ACTIVATIONS,
        doc="Odd scalars (odd functions only)",
    ),
    Option("odd_gate_activation", "choice", "tanh", choices=SCALAR_ACTIVATIONS),
    Option(
        "nonlinearity",
        "choice",
        "normact",
        choices=NONLINEARITIES,
        doc="Activation of the basic variant",
    ),
    Option(
        "norm",
        "choice",
        "component",
        choices=("component", "norm"),
        doc="Normalization of norm activations",
    ),
    Option("s2_resolution", "int", 128, doc="Grid resolution of the S2 activation"),
)
MLP_OPTIONS = (
    Option("output_scale", "float", 1.0, doc="Scale of the last linear layer"),
    Option("init_scale", "float", 1.0, doc="Scale of the initial weights"),
    Option(
        "residual_scale",
        "float",
        0.25,
        doc="Scale of residual branches (res* variants)",
    ),
    Option("pre_norm", "bool", False, doc="RMS-normalize before every layer"),
    Option("film_hidden", "int", 64, doc="Hidden width of FiLM conditioning"),
    Option("norm_eps", "float", 1.0e-8),
)

# Stage option -> configuration field read by the MLP and nonlinearity factories.
CONFIG_FIELDS = {
    "scalar_activation": "activation_scalar",
    "gate_activation": "activation_gate",
    "odd_scalar_activation": "activation_odd_scalar",
    "odd_gate_activation": "activation_odd_gate",
    "nonlinearity": "nonlin_kind",
    "norm": "norm_kind",
    "s2_resolution": "s2act_res",
    "output_scale": "e3mlp_output_scale",
    "init_scale": "e3mlp_weight_init_scale",
    "residual_scale": "e3mlp_residual_scale",
    "pre_norm": "e3mlp_pre_norm",
    "film_hidden": "e3mlp_film_hidden_dim",
    "norm_eps": "e3mlp_norm_eps",
}


def layer_settings(options, *, variant=None):
    """The settings object of an MLP or nonlinearity built from stage options."""
    from net.common import Config

    values = {
        field: options[name] for name, field in CONFIG_FIELDS.items() if name in options
    }
    if variant is not None:
        values["e3mlp_variant"] = variant
    values["dtype"] = torch.get_default_dtype()
    return Config(**values)


def stage_settings(cfg, *, prefix_variant=None):
    """Stage options equivalent to a flat configuration's MLP settings."""
    return {name: getattr(cfg, field) for name, field in CONFIG_FIELDS.items()}
