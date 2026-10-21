#!/usr/bin/env python3
"""Regenerate compact source-only model-math oracles (never a MANDALA forward).

Run with PYTHONPATH=src in the repository's recorded development environment.
Upstream trees must match the pinned commits. Geometry tensors are supplied inputs:
these fixtures certify model/training mathematics, not upstream preprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import types

import e3nn
from e3nn.o3 import Irreps
import torch
import torch_geometric

from net.configuration import resolve_configuration


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "deeph": "141faaaba1d5e17248283ef939bcd2ead2d8e677",
    "deeph_e3": "036bf95b996052ef778c5c38e3c54c2ac0e2b591",
    "mace_h": "3d6248b715eef302a4fd674d52bcd69254596a51",
}


def pack(tensor):
    flat = tensor.detach().cpu().reshape(-1)
    value = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
    }
    if tensor.is_complex():
        value.update(real=flat.real.tolist(), imag=flat.imag.tolist())
    else:
        value["values"] = flat.tolist()
    return value


def unpack(value):
    if "real" in value:
        tensor = torch.complex(torch.tensor(value["real"]), torch.tensor(value["imag"]))
        return tensor.to(getattr(torch, value["dtype"])).reshape(value["shape"])
    return torch.tensor(value["values"], dtype=getattr(torch, value["dtype"])).reshape(
        value["shape"]
    )


def parameter_probe(parameter):
    """Every parameter is covered; three coordinates plus two global moments."""
    if parameter is None:
        return None
    flat = parameter.detach().double().reshape(-1)
    if flat.numel() == 0:
        return {
            "shape": list(parameter.shape),
            "values": [],
            "sum": 0.0,
            "squared_sum": 0.0,
        }
    indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
    return {
        "shape": list(parameter.shape),
        "indices": indices,
        "values": flat[indices].tolist(),
        "sum": flat.sum().item(),
        "squared_sum": flat.square().sum().item(),
    }


def configuration(kind, spinful=False):
    if kind == "deeph":
        overrides = {
            "node_width": 4,
            "edge_width": 8,
            "post_edge_width": 4,
            "angular_dim": 4,
            "num_l": 2,
        }
        shells = {"H": [0]}
    else:
        overrides = dict(
            embedding_width=3,
            radial_width=4,
            angular_irreps="1x0e+1x1o+1x2e",
            node_irreps="3x0e+2x1o+1x2e",
            edge_irreps="3x0e+2x1o+1x2e",
        )
        shells = {"Si": [0, 1]}
        if kind == "mace_h":
            overrides.update(
                hidden_irreps="2x0e+2x1o+2x2e",
                density_lmax=2,
                density_radial_width=4,
                radial_layers=[4, 4, 4],
            )
    return resolve_configuration(
        ROOT / "configs/reference" / f"{kind}.yaml",
        species_shells=shells,
        spinful=spinful,
        overrides={"parameters": overrides},
    )[0]


def supplied_inputs(kind):
    if kind == "deeph":
        # Geometry preprocessing has its own differential tests. Freeze its
        # supplied relation tensors here so the model oracle is independently
        # executable even if MANDALA's geometry builder subsequently changes.
        from data.bond_geometry import bond_frames, bond_neighborhoods

        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.1, 0.3]])
    else:
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.3, 0.2], [-0.4, 1.3, 0.7]])
    edges = torch.cartesian_prod(torch.arange(3), torch.arange(3)).T
    vectors = positions[edges[1]] - positions[edges[0]]
    onsite = edges[0] == edges[1]
    inputs = dict(
        node_type_idx=torch.zeros(3, dtype=torch.long),
        node_batch=torch.zeros(3, dtype=torch.long),
        edge_index=edges,
        edge_lengths=vectors.norm(dim=-1),
        edge_vectors=vectors,
    )
    if kind == "deeph":
        inputs.update(
            bond_neighborhoods(edges, vectors, bond_frames(edges, vectors), num_l=2)
        )
    else:
        inputs.update(
            node_one_hot=torch.ones(3, 1),
            edge_type_idx=torch.zeros(9, dtype=torch.long),
            edge_one_hot=torch.ones(9, 1),
            edge_batch=torch.zeros(9, dtype=torch.long),
            selfloop_edge=onsite,
            message_edge_index=edges[:, ~onsite],
            message_vectors=vectors[~onsite],
            message_lengths=vectors[~onsite].norm(dim=-1),
            message_to_output=torch.where(~onsite)[0],
        )
    return inputs, positions


def load_source(kind, root):
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if actual != PINS[kind]:
        raise ValueError(f"{kind}: expected {PINS[kind]}, got {actual}")
    package_dir = (
        root / {"deeph": "deeph", "deeph_e3": "deephe3", "mace_h": "maceh"}[kind]
    )
    if subprocess.run(
        ["git", "diff", "HEAD", "--quiet", "--", str(package_dir)], cwd=root
    ).returncode:
        raise ValueError(f"{kind}: tracked source differs from pinned commit")
    name = f"_mandala_fixture_source_{kind}"
    package = types.ModuleType(name)
    package.__path__ = [str(package_dir)]
    sys.modules[name] = package
    entry = "maceh" if kind == "mace_h" else "model"
    module = importlib.import_module(f"{name}.{entry}")
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package_dir.rglob("*.py"))
    }
    return module, hashes


def source_model(kind, config, source):
    p, layout = config["parameters"], config["layout"]
    if kind == "deeph":
        model = source.HGNN(
            1,
            4,
            8,
            1,
            "GaussianBasis",
            6.0,
            True,
            False,
            True,
            True,
            "LayerNorm",
            "CGConv",
            False,
            False,
            False,
            num_l=2,
        )
        for layer in model.modules():
            if isinstance(layer, source.CGConv):
                layer.propagate = layer._orig_propagate
        return model
    common = dict(
        num_species=1,
        irreps_edge_init=f"{p['embedding_width']}x0e",
        irreps_sh=p["angular_irreps"],
        irreps_post_node=layout["post_node_irreps"],
        irreps_mid_edge=p["edge_irreps"],
        irreps_post_edge=layout["post_irreps"],
        irreps_out_edge=layout["coupled_irreps"],
        r_max=p["cutoff_radius"],
        num_basis=p["radial_width"],
    )
    if kind == "deeph_e3":
        return source.Net(
            **common,
            irreps_embed_node=f"{p['embedding_width']}x0e",
            irreps_mid_node=p["node_irreps"],
            irreps_out_node="1x0e",
            num_block=3,
            use_sbf=False,
            selftp=p["terminal_self_product"],
        )
    return source.Net(
        **common,
        num_blocks=3,
        max_ell=p["density_lmax"],
        hidden_irreps=p["hidden_irreps"],
        avg_num_neighbors=p["avg_num_neighbors"],
        atomic_numbers=[14],
        correlation=p["correlation"],
        radial_MLP=p["radial_layers"],
        mace_norm="e3LayerNorm",
        expand_norm="e3LayerNorm",
        basis_func="Gaussian",
        num_gaussian=p["density_radial_width"],
        shift_scale=True,
    )


class SourceGraph(dict):
    __getattr__ = dict.__getitem__


def generate(kind, root, spinful=False):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32 if kind == "deeph" else torch.float64)
    config = configuration(kind, spinful)
    source, hashes = load_source(kind, root)
    seed = 42 if kind == "deeph" else 337
    torch.manual_seed(seed)
    model = source_model(kind, config, source).double()
    torch.set_default_dtype(torch.float64)
    inputs, positions = supplied_inputs(kind)
    graph = SourceGraph(
        x=inputs["node_type_idx"],
        edge_index=inputs["edge_index"],
        edge_attr=torch.cat(
            [inputs["edge_lengths"][:, None], inputs["edge_vectors"]], -1
        ),
        batch=inputs["node_batch"],
    )
    statistics = None
    if kind == "mace_h":
        width = Irreps(config["layout"]["coupled_irreps"]).dim
        means, scales = (
            torch.linspace(-0.2, 0.2, width),
            torch.linspace(0.3, 1.7, width),
        )
        with torch.no_grad():
            model.mean_tensor.copy_(means[None, None])
            model.std_tensor.copy_(scales[None, None])
        statistics = {
            "mean": pack(means),
            "std": pack(scales),
            "population": "externally supplied frozen synthetic statistics; no dataset fitting",
        }
    if kind == "deeph":
        target, mask = torch.zeros(9, 1), torch.ones(9, 1, dtype=torch.bool)
        source_codec = None
        edge_attr = torch.cat(
            [
                inputs["edge_lengths"][:, None],
                positions[inputs["edge_index"][0]],
                positions[inputs["edge_index"][1]],
            ],
            -1,
        )

        def forward():
            return model(
                inputs["node_type_idx"],
                inputs["edge_index"],
                edge_attr,
                inputs["node_batch"],
                inputs["sub_atom_idx"],
                inputs["sub_edge_idx"],
                inputs["sub_edge_ang"],
                inputs["sub_index"],
            )

    else:
        module = importlib.import_module(f"{source.__package__}.e3modules")
        source_codec = module.e3TensorDecomp(
            Irreps(config["layout"]["coupled_irreps"]),
            [tuple(pair) for pair in config["layout"]["shell_pairs"]],
            torch.float64,
            spinful=spinful,
        )
        desired = torch.randn(
            9,
            Irreps(config["layout"]["coupled_irreps"]).dim,
            generator=torch.Generator().manual_seed(199),
        )
        target = source_codec.get_H(desired).detach()
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[0] = False

        def forward():
            return source_codec.get_H(model(graph)[1])

    options = dict(lr=0.001 if kind == "deeph" else 0.002, betas=(0.9, 0.999), eps=1e-8)
    initial = {name: parameter_probe(p) for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), **options)
    steps = []
    for _ in range(3):
        optimizer.zero_grad()
        output = forward()
        loss = (output - target).abs().square()[mask].mean()
        loss.backward()
        gradients = {
            name: parameter_probe(p.grad) for name, p in model.named_parameters()
        }
        optimizer.step()
        steps.append(
            {
                "output": pack(output),
                "loss": loss.item(),
                "gradients": gradients,
                "updated_parameters": {
                    name: parameter_probe(p) for name, p in model.named_parameters()
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": kind,
        "spinful": spinful,
        "scope": "Tiny-width default-topology model mathematics and three fresh Adam steps on supplied features",
        "source": {"commit": PINS[kind], "sha256": hashes},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "e3nn": e3nn.__version__,
            "torch_geometric": torch_geometric.__version__,
            "device": "cpu",
            "compute_dtype": "float64",
            "threads": 1,
            "initialization_dtype": "float32" if kind == "deeph" else "float64",
            "compatibility": (
                ["CGConv generic PyG propagate dispatcher"] if kind == "deeph" else []
            ),
        },
        "seed": seed,
        "configuration": config,
        "inputs": {k: pack(v) for k, v in inputs.items()},
        "statistics": statistics,
        "optimizer": options,
        "target": pack(target),
        "mask": pack(mask),
        "initial_parameters": initial,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*PINS, "all"], default="all")
    parser.add_argument(
        "--deeph-root", type=Path, default=Path("/tmp/mandala_architecture_refs/deeph")
    )
    parser.add_argument(
        "--deeph-e3-root", type=Path, default=Path("/home/bartek/casus/DeepH-E3")
    )
    parser.add_argument(
        "--mace-h-root", type=Path, default=Path("/tmp/mandala_architecture_refs/maceh")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "tests/fixtures/reference_operators"
    )
    parser.add_argument(
        "--spinful", action="store_true", help="For coupled-irrep presets only"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kinds = list(PINS) if args.model == "all" else [args.model]
    for kind in kinds:
        if args.spinful and kind == "deeph":
            continue
        root = getattr(
            args,
            {
                "deeph": "deeph_root",
                "deeph_e3": "deeph_e3_root",
                "mace_h": "mace_h_root",
            }[kind],
        )
        result = generate(kind, root, args.spinful)
        path = args.output_dir / f"{kind}{'_spinful' if args.spinful else ''}.json"
        path.write_text(
            json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        )
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
