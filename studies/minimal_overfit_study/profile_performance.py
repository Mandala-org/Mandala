"""
Performance Profiling Script for Minimal E(3)-GNN Study
=========================================================

Usage:
    python profile_performance.py [--profile-type {simple,detailed,cuda}]

Measures:
- Forward pass time
- Backward pass time
- Data structure construction overhead
- Memory usage
- GPU utilization (if available)
"""

import sys
from pathlib import Path
import argparse
import torch
import time
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.math import soft_one_hot_linspace
from ase import Atoms
from ase.neighborlist import neighbor_list

from data.snapshot import Snapshot
from core.block_irrep_mapper import BlockIrrepMapper
from net.common import build_hidden_irreps
from common import MinimalNetwork, compute_detailed_metrics

from collections import Counter
from data.block_matrix import IrrepsBlockData
import torch.nn.functional as F


@contextmanager
def timer(name, enabled=True):
    """Context manager for timing code blocks."""
    if not enabled:
        yield
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    print(f"  ⏱️  {name}: {elapsed*1000:.2f} ms")


def load_data_and_build_graph(config, device):
    """Load snapshot and build graph (same as training script)."""

    print("\n[LOADING DATA]")

    with timer("Load snapshot"):
        snapshot = Snapshot.from_openmx(
            matrix_path=config["data_path"],
            info_path=config["info_path"],
            convention="e3nn",
            symmetrize_density=True,
            cutoff_radius=None,
            dtype=torch.float32,
        )

    hamiltonian_e3nn = snapshot.hamiltonian.to(device)
    overlap_e3nn = snapshot.overlap.to(device)
    orbital_cfg = snapshot.hamiltonian.orbital_cfg
    positions = snapshot.positions.to(device)
    box = snapshot.box.to(device) if snapshot.box is not None else None
    atoms_list = list(snapshot.hamiltonian.atoms)

    print(f"  Atoms: {len(atoms_list)}, Elements: {set(atoms_list)}")

    # Create mapper
    with timer("Create BlockIrrepMapper"):
        mapper = BlockIrrepMapper(orbital_cfg, device=device, dtype=torch.float32)

    # Build graph
    print("\n[BUILDING GRAPH]")
    num_atoms = len(atoms_list)

    ase_atoms = Atoms(
        symbols=atoms_list,
        positions=positions.cpu().numpy(),
        cell=box.cpu().numpy() if box is not None else None,
        pbc=box is not None,
    )

    with timer("Compute neighbor list"):
        src, dst, offsets = neighbor_list(
            "ijS", ase_atoms, config["cutoff_radius"], self_interaction=False
        )

    # Add self-edges
    self_src = torch.arange(num_atoms, dtype=torch.long, device=device)
    self_dst = torch.arange(num_atoms, dtype=torch.long, device=device)
    self_offsets = torch.zeros((num_atoms, 3), dtype=torch.long, device=device)

    all_src = torch.cat([self_src, torch.from_numpy(src).to(device)])
    all_dst = torch.cat([self_dst, torch.from_numpy(dst).to(device)])
    all_offsets = torch.cat([self_offsets, torch.from_numpy(offsets).to(device).long()])

    edge_index = torch.stack([all_src, all_dst], dim=0)
    edge_shift = all_offsets.T

    print(
        f"  Total edges: {edge_index.shape[1]} ({num_atoms} self + {edge_index.shape[1] - num_atoms} neighbors)"
    )

    # Compute edge features
    with timer("Compute edge vectors and distances"):
        if box is not None:
            shift_float = edge_shift.T.float()
            edge_vec = (
                positions[edge_index[1]] - positions[edge_index[0]] + shift_float @ box
            )
        else:
            edge_vec = positions[edge_index[1]] - positions[edge_index[0]]

        edge_dist = torch.linalg.norm(edge_vec, dim=1)

    with timer("Compute spherical harmonics"):
        sh_irreps = Irreps.spherical_harmonics(config["l_max"])
        edge_vec_norm = edge_vec.clone()
        non_zero_mask = edge_dist > 1e-6
        edge_vec_norm[non_zero_mask] = edge_vec[non_zero_mask] / edge_dist[
            non_zero_mask
        ].unsqueeze(-1)
        edge_sh = spherical_harmonics(sh_irreps, edge_vec_norm, normalize=False)

    with timer("Compute radial basis functions"):
        edge_length_emb = soft_one_hot_linspace(
            edge_dist,
            start=0.0,
            end=config["cutoff_radius"],
            number=config["n_radial"],
            basis="gaussian",
            cutoff=False,
        )
        edge_length_emb = edge_length_emb * config["n_radial"] ** 0.5

    with timer("Compute edge types"):
        element_to_idx = {elem: idx for idx, elem in enumerate(orbital_cfg.elements())}
        node_type_idx = torch.tensor(
            [element_to_idx[a] for a in atoms_list], device=device
        )

        edge_type_strs = [
            f"{atoms_list[edge_index[0, i].item()]}-{atoms_list[edge_index[1, i].item()]}"
            for i in range(edge_index.shape[1])
        ]
        edge_type_idx = torch.tensor(
            [mapper.edge_type2idx[et] for et in edge_type_strs], device=device
        )

    batch_node = torch.zeros(num_atoms, dtype=torch.long, device=device)
    batch_edge = torch.zeros(edge_index.shape[1], dtype=torch.long, device=device)

    return {
        "snapshot": snapshot,
        "hamiltonian": hamiltonian_e3nn,
        "overlap": overlap_e3nn,
        "mapper": mapper,
        "atoms_list": atoms_list,
        "orbital_cfg": orbital_cfg,
        "node_type_idx": node_type_idx,
        "edge_type_idx": edge_type_idx,
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "edge_length_emb": edge_length_emb,
        "edge_sh": edge_sh,
        "batch_node": batch_node,
        "batch_edge": batch_edge,
        "sh_irreps": sh_irreps,
    }


def profile_simple(network, data, device, num_iterations=100):
    """Simple timing profile."""

    print("\n" + "=" * 80)
    print("SIMPLE PERFORMANCE PROFILE")
    print("=" * 80)

    network.eval()

    # Warmup
    print("\n[WARMUP]")
    with torch.no_grad():
        for _ in range(10):
            _ = network(
                data["node_type_idx"],
                data["edge_type_idx"],
                data["edge_index"],
                data["edge_shift"],
                data["edge_length_emb"],
                data["edge_sh"],
                data["batch_node"],
                data["batch_edge"],
                log_to_wandb=False,
            )

    # Forward pass timing
    print(f"\n[FORWARD PASS] ({num_iterations} iterations)")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            pred_raw = network(
                data["node_type_idx"],
                data["edge_type_idx"],
                data["edge_index"],
                data["edge_shift"],
                data["edge_length_emb"],
                data["edge_sh"],
                data["batch_node"],
                data["batch_edge"],
                log_to_wandb=False,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    forward_time = (time.perf_counter() - start) / num_iterations

    print(f"  Average forward pass: {forward_time*1000:.2f} ms")

    # Backward pass timing
    print(f"\n[BACKWARD PASS] ({num_iterations} iterations)")
    network.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(num_iterations):
        network.zero_grad()

        pred_raw = network(
            data["node_type_idx"],
            data["edge_type_idx"],
            data["edge_index"],
            data["edge_shift"],
            data["edge_length_emb"],
            data["edge_sh"],
            data["batch_node"],
            data["batch_edge"],
            log_to_wandb=False,
        )

        # Dummy loss
        loss = sum(v["vectors"].sum() for v in pred_raw.values())
        loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    backward_time = (time.perf_counter() - start) / num_iterations

    print(f"  Average forward + backward: {backward_time*1000:.2f} ms")
    print(f"  Backward only: {(backward_time - forward_time)*1000:.2f} ms")

    # Memory
    if torch.cuda.is_available():
        print(f"\n[MEMORY]")
        print(f"  Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")

    return {
        "forward_time_ms": forward_time * 1000,
        "backward_time_ms": backward_time * 1000,
    }


def profile_detailed(network, data, device, num_iterations=20):
    """Detailed breakdown of computation time."""

    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE BREAKDOWN")
    print("=" * 80)

    network.eval()

    print(f"\n[DETAILED TIMING] ({num_iterations} iterations average)")

    # IrrepsBlockData construction
    print("\n1. IrrepsBlockData Construction Overhead")

    with torch.no_grad():
        pred_raw = network(
            data["node_type_idx"],
            data["edge_type_idx"],
            data["edge_index"],
            data["edge_shift"],
            data["edge_length_emb"],
            data["edge_sh"],
            data["batch_node"],
            data["batch_edge"],
            log_to_wandb=False,
        )

    total_construction_time = 0.0

    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        # This is the expensive Python loop
        pair_vec_H = {}
        pair_edges_dict = {}
        lookup_dict = {}

        for key, payload in pred_raw.items():
            pair_vec_H[key] = payload["vectors"]
            pair_edges_dict[key] = payload["edges"]

            for idx, edge_5d in enumerate(payload["edges"].t()):
                sx, sy, sz, i, j = edge_5d.tolist()  # GPU → CPU transfer!
                lookup_dict[(int(sx), int(sy), int(sz), int(i), int(j))] = (key, idx)

        pred_H_irreps = IrrepsBlockData(
            atoms=tuple(data["atoms_list"]),
            atom_counts=Counter(data["atoms_list"]),
            pair_vectors=pair_vec_H,
            pair_edges=pair_edges_dict,
            lookup=lookup_dict,
            orbital_cfg=data["orbital_cfg"],
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_construction_time += time.perf_counter() - start

    avg_construction = total_construction_time / num_iterations
    print(f"   Average: {avg_construction*1000:.2f} ms")

    # Irrep to block conversion
    print("\n2. Irrep-to-Block Conversion (to_blocks)")

    total_conversion_time = 0.0

    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        pred_H_matrix = pred_H_irreps.to_blocks(data["mapper"])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_conversion_time += time.perf_counter() - start

    avg_conversion = total_conversion_time / num_iterations
    print(f"   Average: {avg_conversion*1000:.2f} ms")

    # Loss computation
    print("\n3. Loss Computation")

    total_loss_time = 0.0

    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        loss_H = 0.0
        for key in data["hamiltonian"].pair_blocks.keys():
            if key in pred_H_matrix.pair_blocks:
                pred_blocks = pred_H_matrix.pair_blocks[key]
                targ_blocks = data["hamiltonian"].pair_blocks[key]
                min_n = min(pred_blocks.shape[0], targ_blocks.shape[0])
                loss_H += F.mse_loss(pred_blocks[:min_n], targ_blocks[:min_n])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_loss_time += time.perf_counter() - start

    avg_loss = total_loss_time / num_iterations
    print(f"   Average: {avg_loss*1000:.2f} ms")

    # Detailed metrics computation
    print("\n4. Detailed Metrics Computation")

    with timer("compute_detailed_metrics"):
        _ = compute_detailed_metrics(
            pred_H_matrix, data["hamiltonian"], data["overlap"]
        )

    print("\n[BOTTLENECK SUMMARY]")
    total_post_network = avg_construction + avg_conversion + avg_loss
    print(
        f"  IrrepsBlockData construction: {avg_construction*1000:.2f} ms ({avg_construction/total_post_network*100:.1f}%)"
    )
    print(
        f"  Irrep-to-block conversion:    {avg_conversion*1000:.2f} ms ({avg_conversion/total_post_network*100:.1f}%)"
    )
    print(
        f"  Loss computation:             {avg_loss*1000:.2f} ms ({avg_loss/total_post_network*100:.1f}%)"
    )
    print(f"  Total post-network overhead:  {total_post_network*1000:.2f} ms")


def profile_with_pytorch_profiler(network, data, device):
    """Use PyTorch's built-in profiler for detailed analysis."""

    print("\n" + "=" * 80)
    print("PYTORCH PROFILER")
    print("=" * 80)

    try:
        from torch.profiler import profile, ProfilerActivity, schedule

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        print("\n[PROFILING] Running profiler (10 iterations)...")

        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=5, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace("trace.json"),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(10):
                network.zero_grad()

                pred_raw = network(
                    data["node_type_idx"],
                    data["edge_type_idx"],
                    data["edge_index"],
                    data["edge_shift"],
                    data["edge_length_emb"],
                    data["edge_sh"],
                    data["batch_node"],
                    data["batch_edge"],
                    log_to_wandb=False,
                )

                loss = sum(v["vectors"].sum() for v in pred_raw.values())
                loss.backward()

                prof.step()

        print("\n✅ Profile saved to trace.json")
        print("   View in Chrome: chrome://tracing")

        print("\n[TOP OPERATIONS BY CUDA TIME]")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print("\n[TOP OPERATIONS BY CPU TIME]")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    except ImportError:
        print("⚠️  PyTorch profiler not available (requires PyTorch 1.8+)")


def main():
    parser = argparse.ArgumentParser(
        description="Profile performance of minimal E(3)-GNN"
    )
    parser.add_argument(
        "--profile-type",
        choices=["simple", "detailed", "cuda", "all"],
        default="simple",
        help="Type of profiling to run",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(project_root / "data/small/H2O/original/H2O.matrix"),
        help="Path to matrix file",
    )
    parser.add_argument(
        "--info-path",
        type=str,
        default=str(project_root / "data/small/H2O/original/H2O.info.out"),
        help="Path to info file",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=4,
        help="Maximum angular momentum",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of message passing layers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print("E(3)-GNN PERFORMANCE PROFILER")
    print("=" * 80)
    print(f"\nDevice: {device}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"L_max: {args.l_max}")
    print(f"Num layers: {args.num_layers}")

    # Configuration
    config = {
        "data_path": Path(args.data_path),
        "info_path": Path(args.info_path),
        "hidden_dim": args.hidden_dim,
        "l_max": args.l_max,
        "num_layers": args.num_layers,
        "cutoff_radius": 8.0,
        "n_radial": 128,
    }

    # Load data
    data = load_data_and_build_graph(config, device)

    # Build network
    print("\n[BUILDING NETWORK]")

    with timer("Network instantiation"):
        hidden_irreps = build_hidden_irreps(
            l_max=config["l_max"], base_dim=config["hidden_dim"], use_odd_features=True
        )

        num_elements = len(data["orbital_cfg"].elements())
        num_edge_types = num_elements**2

        # Create network with verbose=False to avoid print statements
        network = MinimalNetwork(
            num_elements=num_elements,
            n_radial=config["n_radial"],
            num_edge_types=num_edge_types,
            hidden_irreps=hidden_irreps,
            sh_irreps=data["sh_irreps"],
            num_layers=config["num_layers"],
            mapper=data["mapper"],
            verbose=False,  # Disable prints for profiling
        ).to(device)

    num_params = sum(p.numel() for p in network.parameters())
    print(f"  Parameters: {num_params:,}")

    # Run profiling
    if args.profile_type in ["simple", "all"]:
        profile_simple(network, data, device)

    if args.profile_type in ["detailed", "all"]:
        profile_detailed(network, data, device)

    if args.profile_type in ["cuda", "all"] and torch.cuda.is_available():
        profile_with_pytorch_profiler(network, data, device)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print("\n✅ See PERFORMANCE_OPTIMIZATION_PLAN.md for optimization strategies")


if __name__ == "__main__":
    main()
