"""
summary.py
==========

Utility functions for printing model and dataset summaries.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any
from torch import nn

from data.gnn_dataset import E3GNNDataset
from data.block_matrix import BlockMatrix, IrrepsBlockData


def print_model_summary(model: nn.Module, verbosity: int = 1) -> None:
    """
    Print a summary of the model architecture and parameter count.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to summarize.
    verbosity : int
        If >= 1, print the summary.
    """
    if verbosity < 1:
        return

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Non-trainable params:  {total_params - trainable_params:,}")

    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f"Model size:            {size_mb:.2f} MB")
    print("=" * 80 + "\n")


def analyze_block_matrix_edges(
    block_matrix: BlockMatrix | IrrepsBlockData,
) -> Dict[str, Any]:
    """
    Analyze edge statistics from a BlockMatrix or IrrepsBlockData.

    Returns
    -------
    dict
        Dictionary containing:
        - total_edges: Total number of edges
        - pbc_edges: Number of edges crossing periodic boundaries
        - pbc_percentage: Percentage of edges crossing PBC
        - periodic_images: Average number of periodic images per unique (i,j) pair
        - avg_neighbors: Average number of neighbors per node
        - num_nodes: Number of unique nodes
    """
    total_edges = 0
    pbc_edges = 0

    # Track unique (i, j) pairs and count their periodic images
    unique_pairs: Dict[Tuple[int, int], int] = {}

    # Track all unique node indices
    unique_nodes = set()

    for key, edges in block_matrix.pair_edges.items():
        # edges shape: (5, num_edges) where first 3 are (sx, sy, sz)
        num_edges = edges.shape[1]
        total_edges += num_edges

        # Count PBC edges (where shift vector is not (0, 0, 0))
        shifts = edges[:3, :]  # (3, num_edges)
        is_pbc = (shifts != 0).any(dim=0)
        pbc_edges += is_pbc.sum().item()

        # Count periodic images and collect node indices
        for edge_idx in range(num_edges):
            i, j = edges[3, edge_idx].item(), edges[4, edge_idx].item()
            pair = (min(i, j), max(i, j))
            unique_pairs[pair] = unique_pairs.get(pair, 0) + 1
            unique_nodes.add(i)
            unique_nodes.add(j)

    num_nodes = len(unique_nodes)
    avg_periodic_images = (
        sum(unique_pairs.values()) / len(unique_pairs) if unique_pairs else 0.0
    )
    pbc_percentage = (pbc_edges / total_edges * 100) if total_edges > 0 else 0.0
    avg_neighbors = total_edges / num_nodes if num_nodes > 0 else 0.0

    return {
        "total_edges": total_edges,
        "pbc_edges": pbc_edges,
        "pbc_percentage": pbc_percentage,
        "periodic_images": avg_periodic_images,
        "avg_neighbors": avg_neighbors,
        "num_nodes": num_nodes,
    }


def analyze_graph_edges(x: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze edge statistics from graph features.

    Parameters
    ----------
    x : dict
        Graph input dictionary containing 'edge_index' and 'positions'.

    Returns
    -------
    dict
        Dictionary containing:
        - total_edges: Total number of edges in graph
        - avg_neighbors: Average number of neighbors per node
    """
    edge_index = x.get("edge_index")
    positions = x.get("positions")

    if edge_index is None or positions is None:
        return {"total_edges": 0, "avg_neighbors": 0.0}

    num_edges = edge_index.shape[1]
    num_nodes = positions.shape[0]

    # Calculate average neighbors (considering directed edges)
    avg_neighbors = num_edges / num_nodes if num_nodes > 0 else 0.0

    return {
        "total_edges": num_edges,
        "avg_neighbors": avg_neighbors,
    }


def print_dataset_summary(
    dataset: E3GNNDataset, name: str = "Dataset", verbosity: int = 1
) -> None:
    """
    Print a summary of the dataset statistics.

    Parameters
    ----------
    dataset : E3GNNDataset
        The dataset to analyze.
    name : str
        Name to display in the summary header.
    verbosity : int
        If >= 1, print the summary.
    """
    if verbosity < 1:
        return

    if len(dataset) == 0:
        print(f"\n{name} is empty.\n")
        return

    # Accumulate statistics
    total_nodes = 0
    total_target_edges = 0
    total_pbc_edges = 0
    total_periodic_images = 0.0
    total_target_neighbors = 0.0
    total_graph_edges = 0
    total_graph_neighbors = 0.0

    num_snapshots_with_graph = 0  # Track how many snapshots have precomputed graphs

    for idx in range(len(dataset)):
        x, y = dataset[idx]

        # Node count
        num_nodes = x["positions"].shape[0]
        total_nodes += num_nodes

        # Target edge statistics (from one of the matrices, e.g., density)
        density = y["density"]
        if isinstance(density, (BlockMatrix, IrrepsBlockData)):
            stats = analyze_block_matrix_edges(density)
            total_target_edges += stats["total_edges"]
            total_pbc_edges += stats["pbc_edges"]
            total_periodic_images += stats["periodic_images"]
            total_target_neighbors += stats["avg_neighbors"]

        # Graph edge statistics (if precomputed)
        if "edge_index" in x:
            num_snapshots_with_graph += 1
            graph_stats = analyze_graph_edges(x)
            total_graph_edges += graph_stats["total_edges"]
            total_graph_neighbors += graph_stats["avg_neighbors"]

    # Compute averages
    n = len(dataset)
    avg_nodes = total_nodes / n
    avg_target_edges = total_target_edges / n
    avg_target_neighbors = total_target_neighbors / n
    avg_pbc_percentage = (
        (total_pbc_edges / total_target_edges * 100) if total_target_edges > 0 else 0.0
    )
    avg_periodic_images = total_periodic_images / n

    print("\n" + "=" * 80)
    print(f"{name.upper()} SUMMARY")
    print("=" * 80)
    print(f"Number of snapshots:              {n}")
    print(f"Average nodes per snapshot:       {avg_nodes:.1f}")
    print()
    print("Target Statistics (Block Matrices):")
    print(f"  Avg edges in targets:           {avg_target_edges:.1f}")
    print(f"  Avg neighbors per node:         {avg_target_neighbors:.1f}")
    print(f"  Avg periodic images per pair:   {avg_periodic_images:.2f}")
    print(f"  Edges over PBC:                 {avg_pbc_percentage:.2f}%")
    print()

    if num_snapshots_with_graph > 0:
        avg_graph_edges = total_graph_edges / num_snapshots_with_graph
        avg_graph_neighbors = total_graph_neighbors / num_snapshots_with_graph
        print("Outputs Statistics (Graphs):")
        print(f"  Avg edges in graph:             {avg_graph_edges:.1f}")
        print(f"  Avg neighbors per node:         {avg_graph_neighbors:.1f}")
    else:
        print("Outputs Statistics (Graphs):")
        print("  (Graph features not precomputed)")
    print("=" * 80 + "\n")
