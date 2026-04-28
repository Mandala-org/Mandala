from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from _bootstrap import add_repo_root_to_path

REPO_ROOT = add_repo_root_to_path()

import torch

from core.block_irrep_mapper import BlockIrrepMapper
from data.factory import DatasetFactory
from data.snapshot import Snapshot
from net.common import Config

DATA_ROOT = REPO_ROOT / "data"
DEMO_OUTPUT_ROOT = REPO_ROOT / "demos" / "_outputs"

H2O_MATRIX = DATA_ROOT / "small" / "H2O" / "original" / "H2O.matrix"
H2O_INFO = DATA_ROOT / "small" / "H2O" / "original" / "H2O.info.out"
FHIAIMS_ROOT = DATA_ROOT / "fhi-aims" / "original" / "basis_small"
PYSCF_RESULTS_ROOT = DATA_ROOT / "pyscf_baseline" / "results"


def demo_config(**overrides: Any) -> Config:
    """Small, notebook-friendly default config."""
    cfg = Config(
        cutoff_radius=7.0,
        l_max=2,
        hidden_base_dim=16,
        num_layers_gnn=1,
        n_radial=16,
        radial_layers=(32,),
        neck_depth=1,
        internal_e3mlp_layers=0,
        head_e3mlp_layers=1,
        batch_size=1,
        lr=1e-3,
        max_epochs=3,
        enable_energy=False,
        enable_num_electrons=False,
        enable_forces=False,
        enable_stress=False,
        train_on_energy=False,
        train_on_num_electrons=False,
        train_on_forces=False,
        train_on_stress=False,
        loss_coef_observables=1e-5,
        loss_coef_forces=0.0,
        loss_coef_stress=0.0,
        matrix_targets=["hamiltonian", "overlap", "density"],
        train_target="matrix",
        device="cpu",
        dtype=torch.float32,
        gpus=0,
        num_workers=0,
        verbosity=0,
        bench_verbosity=0,
        log_on_step=False,
        log_on_epoch=False,
        log_data=False,
        log_forward=False,
        log_per_irrep_metrics=False,
        print_per_irrep_metrics=False,
        log_per_irrep_images=False,
        log_activation_mag=False,
        benchmark=False,
        log_interval=1,
        adaptive_log_interval=False,
        safety_checks=True,
        require_exact_edge_match=True,
        apply_cutoff_to_targets=True,
        precompute_edge_features=True,
    )
    return replace(cfg, **overrides)


def load_h2o_snapshot(
    *, cfg: Config | None = None, convention: str = "e3nn"
) -> Snapshot:
    """Load the bundled H2O OpenMX example."""
    return Snapshot.from_openmx(
        matrix_path=H2O_MATRIX,
        info_path=H2O_INFO,
        convention=convention,
        cutoff_radius=None if cfg is None else cfg.cutoff_radius,
        cfg=cfg,
    )


def load_h2o_dataset(
    *,
    cfg: Config | None = None,
    convention: str = "e3nn",
) -> tuple[Any, Any, BlockIrrepMapper]:
    """Build a tiny one-snapshot dataset for demo purposes."""
    cfg = cfg or demo_config()
    fac = DatasetFactory(cfg, convention=convention)
    fac.add_snapshot(H2O_MATRIX, H2O_INFO, purpose="train")
    return fac.create()


def summarize_snapshot(snapshot: Snapshot) -> str:
    """Return a compact, human-readable snapshot summary."""
    atoms = "".join(snapshot.density.atoms)
    keys = ", ".join(sorted(snapshot.density.keys()))
    pieces = [
        f"atoms: {atoms}",
        f"basis: {snapshot.density.basis}",
        f"matrix keys: {keys}",
        f"cutoff_radius: {snapshot.cutoff_radius}",
        f"has_positions: {snapshot.positions is not None}",
        f"has_box: {snapshot.box is not None}",
    ]
    return "\n".join(pieces)


def summarize_block_matrix(matrix) -> str:
    """Return pairwise shapes and edge counts for a BlockMatrix-like object."""
    lines: list[str] = []
    for key in sorted(matrix.pair_blocks.keys()):
        blocks = matrix.pair_blocks[key]
        edges = matrix.pair_edges[key]
        lines.append(f"{key:>5}: blocks={tuple(blocks.shape)}, edges={edges.shape[1]}")
    return "\n".join(lines)


def summarize_dataset_sample(x: dict[str, Any], y: dict[str, Any]) -> str:
    """Return a readable overview of one dataset sample."""
    lines = [
        f"atoms: {''.join(x['atoms'])}",
        f"node count: {len(x['atoms'])}",
        f"edge count: {x['edge_index'].shape[1]}",
        f"self edges: {x['num_self_edges']}",
        f"target edges before cutoff: {x['target_edges_before_cutoff']}",
        f"target edges after cutoff: {x['target_edges_after_cutoff']}",
    ]
    for name in ("hamiltonian", "overlap", "density"):
        target = y[name]
        shape = getattr(target, "pair_blocks", getattr(target, "pair_vectors", None))
        if shape is not None:
            lines.append(f"{name}: keys={sorted(shape.keys())}")
    return "\n".join(lines)


def ensure_output_dir(name: str) -> Path:
    out = DEMO_OUTPUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    return out
