"""Small real model configurations shared by scientific contract tests.

Only setup lives here. Expected transformations, losses and derivatives belong
in individual tests and must not be computed by this helper.
"""

from contextlib import contextmanager
from pathlib import Path

from omegaconf import OmegaConf
import torch

from core.block_irrep_mapper import BlockIrrepMapper
from core.orbital_irrep_config import OrbitalIrrepConfig
from net.common import Config
from net.configuration import resolve_configuration
from net.e3gnn import E3GNN

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("deeph", "deeph_e3", "mace_h", "tensor_messages_projected_attention")


@contextmanager
def double_precision(seed=917):
    previous = torch.get_default_dtype()
    with torch.random.fork_rng():
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


def tiny_config(kind, *, species_shells=None, spinful=False, radius=3.2):
    species_shells = species_shells or {"H": [0, 1]}
    family = "reference" if kind in MODELS[:3] else "hybrids"
    config = OmegaConf.load(ROOT / "configs" / family / f"{kind}.yaml")
    if kind == "deeph":
        config.parameters.update(
            node_width=4, edge_width=6, post_edge_width=4, num_l=3, angular_dim=9
        )
    elif kind in ("deeph_e3", "mace_h"):
        config.parameters.update(
            embedding_width=3,
            radial_width=4,
            node_irreps="3x0e+2x1o+1x2e",
            edge_irreps="3x0e+2x1o+1x2e",
            angular_irreps="1x0e+1x1o+1x2e",
        )
        if kind == "mace_h":
            config.parameters.update(
                hidden_irreps="2x0e+2x1o+2x2e",
                density_lmax=2,
                density_radial_width=3,
                correlation=2,
                radial_layers=[4],
            )
    if "codec" not in config:
        config.codec = {}
    config.codec.spinful = spinful
    if kind == MODELS[-1] and spinful:
        # An explicit linear readout completes the fixed SOC codec layout.
        # Channels unreachable from this small hybrid's angular basis are zero;
        # the compiler must never silently discard them or mislabel spinless data.
        stages = config.pipeline.stages
        decode_index = next(i for i, stage in enumerate(stages) if stage.id == "decode")
        stages[decode_index].inputs = ["coupled_complete"]
        stages.insert(
            decode_index,
            dict(
                id="readout_completion",
                role="readout",
                transform="irrep_linear",
                inputs=["coupled"],
                outputs=["coupled_complete"],
                options={"irreps_out": "${layout.coupled_irreps}"},
            ),
        )
    config.data.graph_source = "radius"
    config.data.graph_radius = radius
    config.training.dtype = "float64"
    config.training.max_epochs = 2
    config.training.batch_size = 1
    resolved, codec = resolve_configuration(
        config, species_shells=species_shells, spinful=spinful
    )
    return Config.from_mapping(resolved), codec


def tiny_model(
    kind,
    *,
    species_shells=None,
    spinful=False,
    radius=3.2,
    seed=917,
    fit_statistics=True,
):
    with double_precision(seed):
        cfg, codec = tiny_config(
            kind, species_shells=species_shells, spinful=spinful, radius=radius
        )
        orbital_cfg = OrbitalIrrepConfig.from_dict(
            {
                element: [
                    f"1x{degree}{'e' if degree % 2 == 0 else 'o'}" for degree in degrees
                ]
                for element, degrees in codec.species_shells.items()
            }
        )
        model = E3GNN(BlockIrrepMapper(orbital_cfg, dtype=torch.float64), cfg).double()
        if fit_statistics:
            fit_contract_statistics(model)
        return model


def fit_contract_statistics(model):
    """Nontrivial training-only field statistics; never toggle fitted by hand."""
    from net.training_policies import SpeciesIrrepAffine

    for module in model.modules():
        if isinstance(module, SpeciesIrrepAffine):
            pairs = torch.arange(module.mean.shape[0]).repeat_interleave(5)
            values = torch.arange(len(pairs) * module.irreps.dim, dtype=torch.float64)
            values = (values * 0.37).sin().reshape(len(pairs), module.irreps.dim)
            values = values * torch.linspace(0.4, 1.7, module.irreps.dim)
            module.fit(
                [
                    {
                        "id": "contract-training",
                        "split": "train",
                        "pair_type": pairs,
                        "coupled": values,
                    }
                ],
                training_ids=["contract-training"],
            )


def raw_inputs(model, *, atoms=("H", "H", "H"), positions=None, box=None):
    """Build ordinary public structure-only inputs; no cached frames are forged."""
    from data.structure_inference import build_model_input_from_structure

    if positions is None:
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.2, 0.3, 0.2], [-0.4, 1.3, 0.7]], dtype=torch.float64
        )
    if box is None:
        box = torch.tensor(
            [[5.2, 0.3, 0.2], [0.1, 5.4, 0.4], [0.2, 0.1, 5.6]], dtype=torch.float64
        )
    return build_model_input_from_structure(
        atoms=atoms, positions=positions, box=box, cfg=model.cfg, mapper=model.mapper
    )
