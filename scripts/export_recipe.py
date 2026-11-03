"""Write the explicit pipeline recipe of a flat MANDALA configuration.

The flat fields of the configuration are lowered to stages and composite
blocks (``net.component_defaults.default_graph``) and written as a recipe
that trains the same model; architecture fields are removed, training and data
settings are kept.

Example::

    PYTHONPATH=src:. python scripts/export_recipe.py configs/reference/mandala_paper.yaml \\
        --species O Si --output configs/reference/mandala_paper_blocks.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import yaml
from omegaconf import OmegaConf

# Flat fields that only describe the network; the recipe replaces them.
ARCHITECTURE_FIELDS = {
    "l_max",
    "hidden_base_dim",
    "hidden_irreps",
    "edge_type_emb_dim",
    "emb_use_odd_features",
    "edge_encoder_style",
    "edge_encoder_use_sh_tensor_square",
    "e3layernorm",
    "num_layers_gnn",
    "tp_type",
    "use_self_connection",
    "edge_update_node_combine",
    "edge_update_residual",
    "node_update_message_agg",
    "node_update_attention_scalar_dim",
    "node_update_attention_heads",
    "node_update_residual",
    "head_use_mlp_log_scale",
    "neck_depth",
    "internal_e3mlp_layers",
    "head_e3mlp_layers",
    "head_use_node_embeddings_for_self_edges",
    "head_use_tensor_square",
    "head_pair_mode",
    "head_diag_output_scale",
    "head_offdiag_output_scale",
    "head_log_scale_mlp_n_layers",
    "e3mlp_variant",
    "internal_e3mlp_variant",
    "head_e3mlp_variant",
    "e3mlp_output_scale",
    "e3mlp_weight_init_scale",
    "e3mlp_residual_scale",
    "e3mlp_pre_norm",
    "e3mlp_norm_eps",
    "e3mlp_film_hidden_dim",
    "activation_odd_scalar",
    "activation_odd_gate",
    "nonlin_kind",
    "activation_scalar",
    "activation_gate",
    "s2act_res",
    "norm_kind",
    "dropout",
    "n_radial",
    "radial_layers",
    "pair_conditioned_radial_mlp",
    "radial_weight_granularity",
    "radial_embedding_scale",
}


def main(argv=None):
    from net.common import Config
    from net.component_defaults import default_graph

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--species", nargs="+", required=True, help="Species in recipe order"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = OmegaConf.to_container(OmegaConf.load(args.config), resolve=False)
    if document.get("pipeline"):
        raise SystemExit("The configuration already has an explicit pipeline")
    names = {field.name for field in fields(Config)}
    cfg = Config.from_mapping(
        {key: value for key, value in document.items() if key in names}
    )
    pipeline = default_graph(cfg, len(args.species))
    recipe = {
        key: value for key, value in document.items() if key not in ARCHITECTURE_FIELDS
    }
    recipe["species"] = {**recipe.get("species", {}), "order": list(args.species)}
    recipe["pipeline"] = pipeline
    header = (
        f"# Explicit recipe of {args.config}, written by scripts/export_recipe.py.\n"
    )
    args.output.write_text(header + yaml.safe_dump(recipe, sort_keys=False, width=100))
    print(
        f"wrote {args.output}: {len(pipeline['stages'])} top-level stages, {len(pipeline['blocks'])} blocks"
    )


if __name__ == "__main__":
    main()
