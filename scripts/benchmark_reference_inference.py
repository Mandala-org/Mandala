"""Time native model kernels and MANDALA recipes on common stored support."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import types

from ase.data import atomic_numbers
from omegaconf import OmegaConf
import torch

from core.basis_converter import OpenMXE3NNConverter
from data.block_matrix import IrrepsBlockData
from data.operator_features import build_operator_sample
from data.sample import move_tree
from scripts.generate_reference_fixtures import PINS, SourceGraph
from scripts.train import build_components


def load_upstream(root, kind):
    pkg, module = {
        "deeph": ("deeph", "model"),
        "deeph_e3": ("deephe3", "model"),
        "mace_h": ("maceh", "maceh"),
    }[kind]
    directory = root / kind / pkg
    name = "_speed_upstream_" + pkg
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    return importlib.import_module(f"{name}.{module}")


def upstream_model(root, kind, resolved, layout):
    source = load_upstream(root, kind)
    p = resolved["parameters"]
    species = list(layout.species_shells)
    if kind == "deeph":
        model = source.HGNN(
            len(species),
            p["node_width"],
            p["edge_width"],
            layout.packed_dim,
            "GaussianBasis",
            p["gauss_stop"],
            True,
            False,
            True,
            True,
            "LayerNorm",
            "CGConv",
            False,
            False,
            False,
            num_l=p["num_l"],
        )
        # Compatibility bridge also used by the independent parity fixtures.
        for layer in model.modules():
            if isinstance(layer, source.CGConv):
                layer.propagate = layer._orig_propagate
        return model, None
    l = resolved["layout"]
    shared = dict(
        num_species=len(species),
        irreps_edge_init=f'{p["embedding_width"]}x0e',
        irreps_sh=p["angular_irreps"],
        irreps_post_node=l["post_node_irreps"],
        irreps_mid_edge=p["edge_irreps"],
        irreps_post_edge=l["post_irreps"],
        irreps_out_edge=l["coupled_irreps"],
        r_max=p["cutoff_radius"],
        num_basis=p["radial_width"],
    )
    if kind == "deeph_e3":
        model = source.Net(
            **shared,
            irreps_embed_node=f'{p["embedding_width"]}x0e',
            irreps_mid_node=p["node_irreps"],
            irreps_out_node="1x0e",
            num_block=3,
            use_sbf=False,
            selftp=p["terminal_self_product"],
        )
    else:
        model = source.Net(
            **shared,
            num_blocks=3,
            max_ell=p["density_lmax"],
            hidden_irreps=p["hidden_irreps"],
            avg_num_neighbors=p["avg_num_neighbors"],
            atomic_numbers=[atomic_numbers[s] for s in species],
            correlation=p["correlation"],
            radial_MLP=p["radial_layers"],
            mace_norm="e3LayerNorm",
            expand_norm="e3LayerNorm",
            basis_func="Gaussian",
            num_gaussian=p["density_radial_width"],
            shift_scale=False,
        )
    codec = importlib.import_module(f"{source.__package__}.e3modules").e3TensorDecomp(
        l["coupled_irreps"],
        layout.shell_pairs,
        torch.float32,
        device_torch="cuda",
    )
    return model, codec


def native_input(x, kind):
    src, dst = x["edge_index"]
    pos = x["positions"]
    periodic_dst = pos[dst] + x["edge_shift"].T.to(pos) @ x["box"]
    vec = periodic_dst - pos[src]
    length = vec.norm(dim=-1, keepdim=True)
    if kind == "deeph":
        edge_attr = torch.cat((length, periodic_dst, pos[src]), dim=1)
        return (
            x["node_type_idx"],
            x["edge_index"],
            edge_attr,
            x["node_batch"],
            x["sub_atom_idx"],
            x["sub_edge_idx"],
            x["sub_edge_ang"],
            x["sub_index"],
        )
    return SourceGraph(
        x=x["node_type_idx"],
        edge_index=x["edge_index"],
        edge_attr=torch.cat((length, vec), dim=1),
        batch=x["node_batch"],
    )


def timing(fn, cases, warmup, repeats):
    for i in range(warmup):
        result = fn(cases[i % len(cases)])
        torch.cuda.synchronize()
        del result
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    values = []
    for _ in range(repeats):
        for case in cases:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            result = fn(case)
            torch.cuda.synchronize()
            values.append((time.perf_counter_ns() - start) / 1e6)
            del result
    median = statistics.median(values)
    return {
        "median_ms": median,
        "structures_per_second": 1000 / statistics.mean(values),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_above_ready_mib": (torch.cuda.max_memory_allocated() - base) / 2**20,
        "samples_ms": values,
    }


def benchmark(root, material, kind, native, paths, repeats):
    name = kind if native or kind == "mandala" else kind + "_in_mandala"
    print("START", material, name, flush=True)
    config = OmegaConf.load(root / "configs" / f"{kind}_{material}.yaml")
    config.data.splits = {
        "train": list(config.data.splits.train)[:1],
        "val": paths,
        "test": [],
    }
    config.snapshot_cache_dir = None
    config.symmetrize_hamiltonian_targets = False
    config.symmetrize_validation_output = False
    # No targets are used for inference or fitted statistics.
    torch.manual_seed(42)
    resolved, layout, model, datasets, _ = build_components(config, resume=True)
    dataset = datasets["val"]
    snapshots = [dataset._load_snapshot(*paths) for paths in dataset.snapshot_paths]
    expected = [set(s.hamiltonian.lookup) for s in snapshots]
    if layout is None:
        for s in snapshots:
            s.hamiltonian = s.hamiltonian * 0.0
            s.overlap = s.overlap * 0.0
            s.density = s.density * 0.0
    mapper = model.mapper
    if native:
        del model
        gc.collect()
        model, codec = upstream_model(root / "upstream", kind, resolved, layout)
    model.to("cuda").eval()

    def prep(i):
        s = snapshots[i]
        if layout is not None:
            meta = dict(
                atoms=s.hamiltonian.atoms,
                positions=s.positions.float(),
                box=s.box.float(),
                spinful=False,
                species_shells=layout.species_shells,
            )
            x = build_operator_sample(
                meta,
                {},
                expected[i],
                layout,
                species=list(layout.species_shells),
                target_frame="bond" if kind == "deeph" else "global",
                local_coordinates=kind == "deeph",
                num_l=resolved["parameters"].get("num_l", 5),
                frame_meta=dict(positions=s.positions.double(), box=s.box.double()),
                supports=resolved["data"].get("supports"),
            )["inputs"]
        else:
            matrix, info = dataset.snapshot_paths[i]
            x, _ = dataset._process_snapshot_to_sample(
                s, matrix_path=matrix, info_path=info
            )
        x = move_tree(x, torch.device("cuda"), torch.float32)
        return x, native_input(x, kind) if native else x

    def forward(case):
        _, inp = case
        return (
            (model(*inp) if kind == "deeph" else model(inp)) if native else model(inp)
        )

    def decode(case, value):
        x, _ = case
        if native:
            packed = value if kind == "deeph" else codec.get_H(value[1])
            matrix = IrrepsBlockData.from_packed(
                packed,
                x,
                mapper,
                layout,
                frame="bond" if kind == "deeph" else "global",
                operator="hamiltonian",
            ).to_blocks(mapper)
        else:
            matrix = model.predicted_irreps_to_block_matrices(value, x)["hamiltonian"]
        if matrix.basis == "e3nn":
            matrix = OpenMXE3NNConverter(matrix.orbital_cfg).matrix_to_openmx(matrix)
        if matrix.basis != "openmx":
            raise ValueError("Output is outside common OpenMX basis")
        return matrix

    def end_to_end(i):
        case = prep(i)
        return decode(case, forward(case)).to("cpu")

    with torch.no_grad():
        cases = [prep(i) for i in range(len(paths))]
        for i, case in enumerate(cases):
            matrix = decode(case, forward(case))
            if set(matrix.lookup) != expected[i]:
                raise ValueError(f"Output support mismatch for {name} on {paths[i]}")
            if not all(torch.isfinite(b).all() for b in matrix.pair_blocks.values()):
                raise ValueError("Nonfinite Hamiltonian")
            del matrix
        result = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "directed_edges": [len(s) for s in expected],
            "forward": timing(forward, cases, 5, repeats),
        }
        del cases
        result["end_to_end"] = timing(end_to_end, list(range(len(paths))), 2, repeats)
    print("DONE", material, name, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--material", choices=["graphene", "mos2"], required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Run on a Slurm GPU allocation")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = list(
        OmegaConf.load(
            args.root / "configs" / f"mandala_{args.material}.yaml"
        ).data.splits.val
    )[:3]
    result = dict(
        material=args.material,
        structures=paths,
        seed=42,
        batch_size=1,
        dtype="float32",
        tf32=False,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        node=os.environ["SLURMD_NODENAME"],
        job_id=os.environ["SLURM_JOB_ID"],
        upstream_pins=PINS,
        scope="native model call or shared framework call; end-to-end uses shared in-memory stored-support adapter, decoding and CPU output, no disk I/O or Hermitian projection",
        variants={},
    )
    for kind in ("mandala", "deeph", "deeph_e3", "mace_h"):
        for native in [False] if kind == "mandala" else [True, False]:
            name = kind if native or kind == "mandala" else kind + "_in_mandala"
            try:
                result["variants"][name] = benchmark(
                    args.root, args.material, kind, native, paths, args.repeats
                )
            except Exception as e:
                result["variants"][name] = {"error": repr(e)}
                traceback.print_exc()
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output)
            gc.collect()
            torch.cuda.empty_cache()
    if any("error" in v for v in result["variants"].values()):
        raise RuntimeError("One or more variants failed; partial results preserved")


if __name__ == "__main__":
    main()
