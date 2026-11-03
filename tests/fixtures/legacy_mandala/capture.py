"""Capture reference outputs of the pre-refactor monolithic MANDALA model.

Run once on the commit before the modular decomposition. The decomposed model
must reproduce these outputs from the mapped weights (see
``tests/physics/test_mandala_decomposition.py``).
"""

from pathlib import Path

import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.structure_inference import build_model_input_from_structure
from net.common import Config
from net.e3gnn import E3GNN

HERE = Path(__file__).resolve().parent
SHELLS = {"H": ["1x0e", "1x1o"], "O": ["1x0e", "1x1o", "1x2e"]}
ATOMS = ("O", "H", "H", "O", "H")
POSITIONS = [
    [0.0, 0.0, 0.0],
    [0.96, 0.1, 0.05],
    [-0.3, 0.93, 0.1],
    [2.1, 1.4, 0.9],
    [2.6, 2.1, 1.3],
]
BOX = [[4.6, 0.2, 0.1], [0.3, 4.9, 0.2], [0.1, 0.4, 5.1]]
BASE = dict(
    cutoff_radius=3.0,
    l_max=2,
    hidden_irreps="8x0e+2x0o+4x1o+2x1e+2x2e+2x2o",
    hidden_base_dim=8,
    n_radial=6,
    radial_layers=(8,),
    edge_type_emb_dim=5,
    num_layers_gnn=2,
    dtype=torch.float64,
    verbosity=0,
    data={"graph_source": "radius", "graph_radius": 3.0},
)
VARIANTS = {
    "paper": dict(
        node_update_message_agg="attention",
        node_update_attention_scalar_dim=8,
        node_update_attention_heads=2,
        e3mlp_variant="film",
        internal_e3mlp_variant="resnormact",
        head_e3mlp_variant="normact",
        e3mlp_film_hidden_dim=6,
        separate_shifted_self=True,
        head_use_node_embeddings_for_self_edges=True,
        matrix_targets=["hamiltonian"],
    ),
    "variant_b": dict(
        node_update_message_agg="sum",
        use_self_connection=True,
        e3layernorm=True,
        edge_update_node_combine="tensor_product",
        node_update_residual=True,
        edge_update_residual=True,
        head_pair_mode="shared_conditioned",
        head_use_mlp_log_scale=True,
        head_use_tensor_square=True,
        pair_conditioned_radial_mlp=True,
        tp_type="fully_connected",
        separate_shifted_self=False,
        head_use_node_embeddings_for_self_edges=False,
        e3mlp_variant="basic",
        neck_depth=2,
        matrix_targets=["hamiltonian", "overlap"],
    ),
    "variant_c": dict(
        node_update_message_agg="average",
        edge_encoder_style="distance",
        edge_update_node_combine="sum",
        internal_e3mlp_layers=0,
        e3mlp_variant="gatemagnitudes",
        head_e3mlp_layers=2,
        nonlin_kind="gate_magnitudes",
        radial_weight_granularity="copy",
        hidden_irreps="8x0e+4x1o+2x2e",
        matrix_targets=["hamiltonian", "density"],
    ),
}


def build(name, seed=11):
    torch.manual_seed(seed)
    cfg = Config(**{**BASE, **VARIANTS[name]})
    orbital_cfg = OrbitalIrrepConfig.from_dict(SHELLS)
    mapper = BlockIrrepMapper(orbital_cfg, dtype=torch.float64)
    model = E3GNN(mapper, cfg).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    return model


def inputs(model):
    return build_model_input_from_structure(
        atoms=ATOMS,
        positions=torch.tensor(POSITIONS, dtype=torch.float64),
        box=torch.tensor(BOX, dtype=torch.float64),
        cfg=model.cfg,
        mapper=model.mapper,
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    for name in VARIANTS:
        model = build(name).eval()
        with torch.no_grad():
            predictions = model(inputs(model))
        outputs = {
            operator: {
                key: value.clone() for key, value in prediction.pair_vectors.items()
            }
            for operator, prediction in predictions.items()
        }
        torch.save(
            {"state_dict": model.state_dict(), "outputs": outputs},
            HERE / f"{name}.pt",
        )
        print(
            name,
            sum(p.numel() for p in model.parameters()),
            {
                op: {k: tuple(v.shape) for k, v in d.items()}
                for op, d in outputs.items()
            },
        )
