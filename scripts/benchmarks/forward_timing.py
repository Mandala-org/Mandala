"""Time the forward and training passes of a MANDALA model on a synthetic crystal.

Uses only interfaces shared by the monolithic and the modular implementation
(``Config``, ``E3GNN``, structure-only inputs), so the same script times both
source trees: put the tree to measure first on ``PYTHONPATH``.

Example::

    PYTHONPATH=src:. python scripts/benchmarks/forward_timing.py \\
        --config configs/reference/mandala_paper.yaml \\
        --shells '{"O": [0, 0, 1, 1, 2], "Si": [0, 0, 1, 1, 2]}' \\
        --atoms 96 --device cuda --label modular --output timing.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Flat-config fields that describe data, logging or training infrastructure;
# they do not change the network and may refer to unavailable paths.
INFRASTRUCTURE = ("data", "training", "save_dir", "snapshot_cache_dir", "wandb_project")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shells", required=True, help='JSON {"species": [l, ...]}')
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument(
        "--density", type=float, default=0.066, help="Atoms per cubic Angstrom"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Graph radius (default: cutoff_radius)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--no-static-cache",
        action="store_true",
        help="Recompute structure-only stages every forward (as the base tree does)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def synthetic_crystal(shells, atoms, density, seed):
    """A jittered simple-cubic arrangement in a periodic cubic cell."""
    generator = torch.Generator().manual_seed(seed)
    per_side = math.ceil(atoms ** (1 / 3))
    length = (atoms / density) ** (1 / 3)
    spacing = length / per_side
    grid = torch.stack(
        torch.meshgrid(
            *(torch.arange(per_side, dtype=torch.float64),) * 3, indexing="ij"
        ),
        -1,
    ).reshape(-1, 3)[:atoms]
    jitter = (
        0.15
        * spacing
        * (torch.rand(atoms, 3, generator=generator, dtype=torch.float64) - 0.5)
    )
    positions = (grid + 0.5) * spacing + jitter
    species = sorted(shells)
    symbols = tuple(species[i % len(species)] for i in range(atoms))
    return symbols, positions, torch.eye(3, dtype=torch.float64) * length


def build_model(config_path, shells, radius, dtype):
    from core.block_irrep_mapper import BlockIrrepMapper
    from core.orbital_irrep_config import OrbitalIrrepConfig
    from net.common import Config
    from net.configuration import resolve_configuration
    from net.e3gnn import E3GNN

    document = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    data = dict(document.get("data") or {})
    for key in INFRASTRUCTURE:
        document.pop(key, None)
    cutoff = radius or float(document.get("cutoff_radius", Config().cutoff_radius))
    document["data"] = {
        key: value
        for key, value in data.items()
        if key in {"representation", "basis", "supports"}
    }
    document["data"].update(graph_source="radius", graph_radius=cutoff)
    document.update(cutoff_radius=cutoff, verbosity=0, dtype=dtype)
    resolved, _ = resolve_configuration(document, species_shells=shells)
    cfg = Config.from_mapping(resolved)
    parity = {0: "e", 1: "o", 2: "e", 3: "o"}
    orbital = OrbitalIrrepConfig.from_dict(
        {
            element: [f"1x{l}{parity[l]}" for l in degrees]
            for element, degrees in shells.items()
        }
    )
    mapper = BlockIrrepMapper(orbital, dtype=getattr(torch, dtype))
    model = E3GNN(mapper, cfg)
    # Training-set output statistics (MACE-H) cost the same whatever their
    # values; benchmarks use the identity calibration.
    for module in model.modules():
        if hasattr(module, "fitted") and isinstance(module.fitted, torch.Tensor):
            module.fitted.fill_(True)
    return model, cfg, mapper


def model_inputs(model, cfg, mapper, symbols, positions, box, device, dtype):
    from data.structure_inference import build_model_input_from_structure
    from data.sample import move_tree

    x = build_model_input_from_structure(
        atoms=symbols,
        positions=positions.to(dtype),
        box=box.to(dtype),
        cfg=cfg,
        mapper=mapper,
    )
    x = move_tree(x, device)
    try:  # Modular tree: cache structure-only stages as the dataset does.
        from data.graph_features import attach_static_features, complete_graph_metadata
        from data.graph_topology import prepare_graph_topology
        from net.component_defaults import static_feature_pipeline

        prepare_graph_topology(x, cfg.data.get("supports"))
        complete_graph_metadata(
            x, mapper, separate_shifted_self=cfg.separate_shifted_self
        )
        static = static_feature_pipeline(cfg, mapper).to(device)
        attach_static_features(x, static)
        x["_geometry_reference"] = {
            key: x[key].detach().clone() for key in ("positions", "box")
        }
    except ImportError:
        pass
    return x


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timed(function, device, warmup, repeats):
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    return {
        "median_s": statistics.median(samples),
        "p10_s": ordered[max(0, int(0.1 * len(ordered)) - 1)],
        "p90_s": ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)],
        "samples_s": samples,
    }


def main(argv=None):
    args = parse_arguments(argv)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    shells = {key: list(value) for key, value in json.loads(args.shells).items()}
    model, cfg, mapper = build_model(args.config, shells, args.radius, args.dtype)
    model = model.to(args.device, dtype)
    symbols, positions, box = synthetic_crystal(
        shells, args.atoms, args.density, args.seed
    )
    x = model_inputs(model, cfg, mapper, symbols, positions, box, args.device, dtype)
    if args.no_static_cache:
        x.pop("_static_features", None)

    def forward():
        with torch.no_grad():
            model(x)

    def train_step():
        model.zero_grad(set_to_none=True)
        predictions = model(x)
        loss = sum(
            vectors.square().sum()
            for p in predictions.values()
            for vectors in p.pair_vectors.values()
        )
        loss.backward()

    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    result = {
        "label": args.label,
        "config": str(args.config),
        "shells": shells,
        "atoms": args.atoms,
        "edges": int(x["edge_index"].shape[1]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(args.device),
        "device_name": (
            torch.cuda.get_device_name()
            if str(args.device).startswith("cuda")
            else platform.processor()
        ),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "forward": timed(forward, args.device, args.warmup, args.repeats),
        "train_step": timed(train_step, args.device, args.warmup, args.repeats),
    }
    if str(args.device).startswith("cuda"):
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        f"{args.label}: {result['parameters']} parameters, {result['edges']} edges, "
        f"forward {result['forward']['median_s'] * 1e3:.2f} ms, "
        f"train step {result['train_step']['median_s'] * 1e3:.2f} ms"
    )


if __name__ == "__main__":
    main()
