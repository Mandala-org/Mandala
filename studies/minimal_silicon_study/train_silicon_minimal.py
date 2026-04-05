"""
Minimal silicon study (multi-snapshot).

A simplified variant of minimal_overfit_study that:
  - loads multiple silicon snapshots from a directory,
  - builds train/val datasets via DatasetFactory,
  - trains MinimalNetwork on multiple snapshots.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from ase import Atoms
from ase.neighborlist import neighbor_list
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import Irreps, spherical_harmonics
from tqdm.auto import tqdm

# Add project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse observables-study utilities.
MINIMAL_OVERFIT_DIR = Path(__file__).resolve().parents[1] / "minimal_overfit_study"
sys.path.insert(0, str(MINIMAL_OVERFIT_DIR))
OBSERVABLES_STUDY_DIR = (
    Path(__file__).resolve().parents[1] / "minimal_overfit_observables"
)
sys.path.insert(0, str(OBSERVABLES_STUDY_DIR))

from data.factory import DatasetFactory
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix, IrrepsBlockData
from core.block_irrep_mapper import BlockIrrepMapper
from core.sparse_math import (
    build_trace_alignment,
    build_trace_alignment_from_pair_edges,
    trace_matmul_sparse_block_matrix_aligned,
)
from net.common import Config as NetConfig, build_hidden_irreps

from common import (
    MinimalNetwork,
    canonicalize_edge_order,
    compile_frames_to_video,
    compute_basic_matrix_metrics_aligned,
    compute_detailed_metrics_aligned,
    compute_distance_error_curve,
    compute_irrep_metrics,
    get_all_irreps_in_hamiltonian,
    save_distance_error_curve_plot,
    save_dos_comparison_plot,
    save_hamiltonian_frame_to_disk,
    split_hamiltonian_by_irrep,
    visualize_hamiltonians,
)
from strict_checks import strict_edge_alignment_check, strict_reverse_edge_check
from detailed_logging import (
    build_wandb_detailed_metrics_log,
    build_wandb_per_irrep_metrics_log,
    log_config,
    log_cutoff_application,
    log_detailed_training_metrics,
    log_final_metrics,
    log_graph,
    log_mapper_info,
    log_orbital_config,
    log_per_irrep_metrics,
    log_snapshot_info,
    log_strict_checks_passed,
    log_study_complete,
)

DISTANCE_NORM_POWER = 4
DISTANCE_NORM_MIN_ARG = 1e-12
DISTANCE_NORM_MAG_EPS = 1e-300
HARTREE_TO_EV = 27.2113845
UNIT_SCALE_FROM_HARTREE = {
    "hartree": 1.0,
    "ev": HARTREE_TO_EV,
    "mev": HARTREE_TO_EV * 1000.0,
    "100mev": HARTREE_TO_EV * 10.0,
}
UNIT_DISPLAY_NAME = {
    "hartree": "Hartree",
    "ev": "eV",
    "mev": "meV",
    "100mev": "100meV",
}
BOOLEAN_ARG_NAMES = [
    "enable_energy",
    "enable_num_electrons",
    "train_on_energy",
    "train_on_num_electrons",
    "enable_forces",
    "train_on_forces",
    "rescale_density_to_num_electrons",
    "symmetrize_preds",
    "adaptive_log_interval",
    "benchmark",
    "e3layernorm",
    "separate_shifted_self",
    "edge_encoder_use_sh_tensor_square",
    "head_mlp_for_scalars",
    "head_use_tensor_square",
    "head_use_node_embeddings_for_self_edges",
    "apply_cutoff_to_targets",
    "require_exact_edge_match",
    "log_data",
    "log_model",
    "log_forward",
    "verbose_forward",
    "log_per_irrep_metrics",
    "print_per_irrep_metrics",
    "log_per_irrep_images",
    "generate_video",
]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


def parse_matrix_targets(value):
    targets = [t.strip().lower() for t in str(value).split(",") if t.strip()]
    if not targets:
        raise argparse.ArgumentTypeError(
            "Expected at least one target in --matrix-targets."
        )
    return targets


def _coerce_wandb_bool(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot coerce wandb.config['{name}']={value!r} to bool")


def _get_wandb_config_value(name: str):
    """Read config by canonical underscore key or sweep-style hyphen key."""
    if name in wandb.config:
        return wandb.config[name]
    alias = name.replace("_", "-")
    if alias in wandb.config:
        return wandb.config[alias]
    return None


ARCHITECTURE_ARG_NAMES = {
    "training_unit",
    "convention",
    "matrix_targets",
    "orbital_selection",
    "hidden_dim",
    "l_max",
    "hidden_irreps",
    "num_layers",
    "n_radial",
    "dtype",
    "e3layernorm",
    "separate_shifted_self",
    "edge_encoder_use_sh_tensor_square",
    "radial_embedding_scale",
    "head_mlp_for_scalars",
    "head_use_tensor_square",
    "head_use_node_embeddings_for_self_edges",
    "head_e3mlp_layers",
}
DATA_OVERRIDE_ARG_NAMES = {
    "data_path",
    "num_train",
    "num_val",
    "train_temps",
    "val_temp",
    "n_snapshots_per_temp",
    "val_n_snapshots",
    "seed",
    "require_exact_edge_match",
}
FORBIDDEN_RESUME_OVERRIDE_ARG_NAMES = {
    "cutoff_radius",
    "apply_cutoff_to_targets",
}
OBJECTIVE_OVERRIDE_ARG_NAMES = {
    "enable_energy",
    "enable_num_electrons",
    "enable_forces",
    "train_on_energy",
    "train_on_num_electrons",
    "train_on_forces",
    "loss_coef_observables",
    "loss_coef_density_matrix",
    "loss_coef_forces",
    "rescale_density_to_num_electrons",
    "symmetrize_preds",
}
SCHEDULER_OVERRIDE_ARG_NAMES = {"lr", "lr_factor", "lr_patience"}
NON_TRAINING_OVERRIDE_ARG_NAMES = {
    "run_name",
    "checkpoint_dir",
    "snapshot_cache_dir",
    "wandb_project",
    "wandb_entity",
    "device",
    "log_interval",
    "adaptive_log_interval",
    "benchmark",
    "log_data",
    "log_model",
    "log_forward",
    "verbose_forward",
    "log_per_irrep_metrics",
    "print_per_irrep_metrics",
    "log_per_irrep_images",
    "generate_video",
    "video_max_atoms",
    "grad_clip",
    "num_epochs",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal silicon multi-snapshot study")

    # Data split arguments.
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument(
        "--training-unit",
        type=str.lower,
        default="ev",
        choices=["hartree", "ev", "mev", "100mev"],
        help=(
            "Unit used for Hamiltonian training targets and metrics. "
            "Raw OpenMX Hamiltonian is interpreted as Hartree and scaled to this unit."
        ),
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=None,
        help="Global split mode: number of train pairs from discovered list",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=None,
        help="Global split mode: number of val pairs from discovered list (can be 0)",
    )
    parser.add_argument(
        "--train-temps",
        type=str,
        default=None,
        help=(
            "Temperature split mode: comma-separated training temperatures, e.g. "
            "'900,1200,1500'. If omitted, all detected temperatures except --val-temp are used."
        ),
    )
    parser.add_argument(
        "--val-temp",
        type=int,
        default=None,
        help=(
            "Temperature split mode: validation temperature (e.g., 2700). "
            "Can be one of training temperatures; validation snapshots are then "
            "selected disjointly from training snapshots."
        ),
    )
    parser.add_argument(
        "--n-snapshots-per-temp",
        type=int,
        default=None,
        help="Temperature split mode: number of snapshots sampled per training temperature",
    )
    parser.add_argument(
        "--val-n-snapshots",
        type=int,
        default=None,
        help=(
            "Temperature split mode: validation snapshots at --val-temp "
            "(defaults to --n-snapshots-per-temp, can be 0)"
        ),
    )

    # Selected features.
    parser.add_argument("--convention", type=str, default="e3nn")
    parser.add_argument(
        "--matrix-targets",
        type=parse_matrix_targets,
        default=parse_matrix_targets("hamiltonian,overlap,density"),
        help="Comma-separated matrix targets to predict/train (subset of: hamiltonian,overlap,density).",
    )
    parser.add_argument(
        "--enable-energy",
        type=parse_bool,
        default=True,
        help="Enable energy metric computation from predicted matrices (default: True).",
    )
    parser.add_argument(
        "--enable-num-electrons",
        type=parse_bool,
        default=True,
        help="Enable number-of-electrons metric computation from predicted matrices (default: True).",
    )
    parser.add_argument(
        "--train-on-energy",
        type=parse_bool,
        default=True,
        help="Enable energy loss term (default: True).",
    )
    parser.add_argument(
        "--train-on-num-electrons",
        type=parse_bool,
        default=True,
        help="Enable number-of-electrons loss term (default: True).",
    )
    parser.add_argument(
        "--loss-coef-observables",
        type=float,
        default=1e-5,
        help="Loss coefficient for energy/num-electrons terms (default: 1e-5).",
    )
    parser.add_argument(
        "--loss-coef-density-matrix",
        type=float,
        default=1.0,
        help="Additional multiplier applied only to the density matrix block loss (default: 1.0).",
    )
    parser.add_argument(
        "--enable-forces",
        type=parse_bool,
        default=False,
        help="Enable force-related data/paths.",
    )
    parser.add_argument(
        "--train-on-forces",
        type=parse_bool,
        default=False,
        help="Enable force loss term.",
    )
    parser.add_argument(
        "--loss-coef-forces",
        type=float,
        default=0.0,
        help="Loss coefficient for force term (default: 0.0).",
    )
    parser.add_argument(
        "--symmetrize-preds",
        type=parse_bool,
        default=True,
        help=(
            "Symmetrize predicted H/S/D matrices as (M + M^T) / 2 for "
            "matrix metrics/visualizations (default: True). Observable and "
            "force computations use unsymmetrized predictions."
        ),
    )
    parser.add_argument(
        "--rescale-density-to-num-electrons",
        type=parse_bool,
        default=False,
        help=(
            "Post-prediction metrics-only correction: rescale predicted density so "
            "Tr(D_pred S_pred) matches the target number of electrons. "
            "Training losses and observable losses stay on the raw prediction."
        ),
    )
    parser.add_argument(
        "--orbital-selection",
        type=str,
        default=None,
        help=(
            "Optional orbital reduction spec for all snapshots. "
            "Examples: '1s1p' or '{\"Si\":\"2s2p\"}'."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--l-max", type=int, default=4)
    parser.add_argument("--hidden-irreps", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--cutoff-radius", type=float, default=8.0)
    parser.add_argument("--n-radial", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=4000)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Floating-point dtype for data and model parameters (default: float32)",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients across this many train samples before optimizer step.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--adaptive-log-interval", type=parse_bool, default=False)
    parser.add_argument("--benchmark", type=parse_bool, default=False)
    parser.add_argument("--e3layernorm", type=parse_bool, default=True)
    parser.add_argument("--separate-shifted-self", type=parse_bool, default=False)
    parser.add_argument(
        "--edge-encoder-use-sh-tensor-square",
        type=parse_bool,
        default=False,
        help=(
            "Use TensorSquare(spherical harmonics) before the edge encoder tensor product "
            "(default: False)."
        ),
    )
    parser.add_argument(
        "--radial-embedding-scale",
        type=str,
        default="none",
        choices=["none", "sqrt_n_radial"],
        help=(
            "Optional extra scaling for radial embeddings. "
            "'none' matches src/DeepH-style; 'sqrt_n_radial' reproduces legacy minimal behavior."
        ),
    )
    parser.add_argument("--head-mlp-for-scalars", type=parse_bool, default=False)
    parser.add_argument("--head-use-tensor-square", type=parse_bool, default=False)
    parser.add_argument(
        "--head-use-node-embeddings-for-self-edges",
        type=parse_bool,
        default=False,
    )
    parser.add_argument("--head-e3mlp-layers", type=int, default=3)
    parser.add_argument("--apply-cutoff-to-targets", type=parse_bool, default=False)
    parser.add_argument("--require-exact-edge-match", type=parse_bool, default=False)

    parser.add_argument("--log-data", type=parse_bool, default=False)
    parser.add_argument("--log-model", type=parse_bool, default=False)
    parser.add_argument("--log-forward", type=parse_bool, default=False)
    parser.add_argument("--verbose-forward", type=parse_bool, default=False)
    parser.add_argument("--log-per-irrep-metrics", type=parse_bool, default=False)
    parser.add_argument("--print-per-irrep-metrics", type=parse_bool, default=False)
    parser.add_argument("--log-per-irrep-images", type=parse_bool, default=False)
    parser.add_argument("--generate-video", type=parse_bool, default=False)
    parser.add_argument(
        "--video-max-atoms",
        type=int,
        default=6,
        help=(
            "For training-video frame plots, keep only first N atoms in the displayed "
            "matrix. Set <=0 to disable cropping."
        ),
    )

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="studies/minimal_silicon_study/checkpoints",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="mandala-minimal-silicon-study",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--checkpoint-source-project",
        type=str,
        default=None,
        help=(
            "Optional Weights & Biases project used only to resolve "
            "--resume-from-run-id. If omitted, --wandb-project is used."
        ),
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional Weights & Biases entity. If omitted, use WANDB_ENTITY/default account.",
    )
    parser.add_argument(
        "--snapshot-cache-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for raw parsed Snapshot .pt cache. "
            "If omitted, no snapshot cache is used."
        ),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--randomize-seed",
        type=parse_bool,
        default=False,
        help="If true, replace --seed with a fresh random seed at startup.",
    )
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--resume-from-run-id", type=str, default=None)
    parser.add_argument(
        "--fresh-run",
        type=parse_bool,
        default=False,
        help=(
            "If true together with --resume-from-run-id, load the checkpoint resolved "
            "from that run id but start a new independent W&B run."
        ),
    )
    return parser


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_parser()
    return parser, parser.parse_args()


def _get_explicit_cli_overrides(
    parser: argparse.ArgumentParser, argv: list[str]
) -> set[str]:
    option_to_dest = {}
    for action in parser._actions:
        for option_string in action.option_strings:
            option_to_dest[option_string] = action.dest

    explicit: set[str] = set()
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("--"):
            i += 1
            continue
        opt = token.split("=", 1)[0]
        dest = option_to_dest.get(opt)
        if dest is not None:
            explicit.add(dest)
        i += 1
    return explicit


def _normalize_config_value(name: str, value):
    if name == "matrix_targets" and value is not None:
        return list(value)
    if name == "train_temps" and isinstance(value, list):
        return ",".join(str(v) for v in value)
    return value


def _apply_checkpoint_config_overrides(
    args: argparse.Namespace,
    checkpoint_config: dict[str, Any],
    explicit_overrides: set[str],
) -> argparse.Namespace:
    merged = copy.deepcopy(args)
    for name, value in checkpoint_config.items():
        if not hasattr(merged, name):
            continue
        if name in explicit_overrides or name == "resume_from_checkpoint":
            continue
        setattr(merged, name, _normalize_config_value(name, value))
    return merged


def _load_checkpoint(path: str | os.PathLike) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _resolve_resume_checkpoint_path(path: str | os.PathLike) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        latest = candidate / "latest_checkpoint.pt"
        best = candidate / "best_model.pt"
        final = candidate / "final_model.pt"
        for option in (latest, best, final):
            if option.exists():
                return option
        raise FileNotFoundError(
            f"No checkpoint file found in directory {candidate}. "
            "Expected one of latest_checkpoint.pt, best_model.pt, final_model.pt."
        )
    if not candidate.exists():
        raise FileNotFoundError(f"--resume-from-checkpoint does not exist: {candidate}")
    return candidate


def _resolve_wandb_entity(explicit_entity: str | None) -> str:
    if explicit_entity:
        return explicit_entity
    env_entity = os.environ.get("WANDB_ENTITY")
    if env_entity:
        return env_entity
    api = wandb.Api()
    default_entity = getattr(api, "default_entity", None)
    if default_entity:
        return default_entity
    raise ValueError(
        "Could not determine W&B entity for --resume-from-run-id. "
        "Pass --wandb-entity explicitly or set WANDB_ENTITY."
    )


def _resolve_resume_checkpoint_from_run_id(
    run_id: str, *, project: str, entity: str
) -> Path:
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve W&B run '{entity}/{project}/{run_id}'."
        ) from exc

    summary = run.summary
    checkpoint_path = (
        summary.get("checkpoint/latest_path")
        or summary.get("checkpoint/final_path")
        or summary.get("latest_checkpoint_path")
        or summary.get("final_model_path")
    )
    if not checkpoint_path:
        raise RuntimeError(
            "W&B run does not contain a saved checkpoint path in summary. "
            "Expected one of: checkpoint/latest_path, checkpoint/final_path, "
            "latest_checkpoint_path, final_model_path."
        )

    checkpoint = Path(str(checkpoint_path)).expanduser()
    if not checkpoint.exists():
        raise RuntimeError(
            "Checkpoint path resolved from W&B summary does not exist on this machine: "
            f"{checkpoint}"
        )
    return checkpoint.resolve()


def _config_differences(
    args: argparse.Namespace, checkpoint_config: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    diffs: dict[str, tuple[Any, Any]] = {}
    for name, old_value in checkpoint_config.items():
        if not hasattr(args, name):
            continue
        new_value = getattr(args, name)
        norm_old = _normalize_config_value(name, old_value)
        norm_new = _normalize_config_value(name, new_value)
        if norm_old != norm_new:
            diffs[name] = (norm_old, norm_new)
    return diffs


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _serialize_pairs(pairs: list[tuple[Path, Path]]) -> list[tuple[str, str]]:
    return [(str(m), str(i)) for m, i in pairs]


def _deserialize_pairs(
    payload: list[tuple[str, str]] | None,
) -> list[tuple[Path, Path]] | None:
    if payload is None:
        return None
    return [(Path(m), Path(i)) for m, i in payload]


def _summarize_resume_changes(
    explicit_overrides: set[str],
    checkpoint_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, tuple[Any, Any]]:
    changes: dict[str, tuple[Any, Any]] = {}
    for name in explicit_overrides:
        if name == "resume_from_checkpoint" or not hasattr(args, name):
            continue
        old_value = _normalize_config_value(name, checkpoint_config.get(name))
        new_value = _normalize_config_value(name, getattr(args, name))
        if old_value != new_value:
            changes[name] = (old_value, new_value)
    return changes


def _save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    network: MinimalNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    score: float,
    config: dict[str, Any],
    history: dict[str, list[float]],
    best_score: float,
    best_epoch: int,
    train_pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
    metrics: dict[str, Any] | None = None,
    wandb_run_id: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": score,
            "config": config,
            "metrics": metrics or {},
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "train_pairs": _serialize_pairs(train_pairs),
            "val_pairs": _serialize_pairs(val_pairs),
            "rng_state_python": random.getstate(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_torch_cpu": torch.get_rng_state(),
            "rng_state_torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "wandb_run_id": wandb_run_id,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "wandb_run_name": wandb_run_name,
        },
        path,
    )


def _update_wandb_checkpoint_summary(
    *,
    latest_checkpoint_path: Path | None = None,
    final_model_path: Path | None = None,
) -> None:
    if wandb.run is None:
        return
    if latest_checkpoint_path is not None:
        wandb.run.summary["checkpoint/latest_path"] = str(
            latest_checkpoint_path.resolve()
        )
    if final_model_path is not None:
        wandb.run.summary["checkpoint/final_path"] = str(final_model_path.resolve())
    wandb.run.summary["wandb/run_id"] = wandb.run.id
    wandb.run.summary["wandb/run_name"] = wandb.run.name


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_runtime_seed(fixed_seed: int, randomize_seed: bool) -> int:
    if not randomize_seed:
        return int(fixed_seed)
    return random.SystemRandom().randint(0, 2**31 - 1)


def build_timestamped_run_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}"


PREPROCESSED_SAMPLE_CACHE_VERSION = "v1"


def _orbital_selection_cache_key(orbital_selection: Any) -> str:
    if orbital_selection is None:
        return "none"
    if isinstance(orbital_selection, dict):
        return json.dumps(orbital_selection, sort_keys=True)
    return str(orbital_selection)


def get_preprocessed_sample_cache_file(
    *,
    snapshot_cache_dir: str | None,
    matrix_path: Path,
    info_path: Path,
    mapper: BlockIrrepMapper,
    sh_irreps: Irreps,
    n_radial: int,
    radial_embedding_scale: str,
    training_unit: str,
    hamiltonian_scale_from_hartree: float,
    cutoff_radius: float,
    apply_cutoff_to_targets: bool,
    orbital_selection: Any,
    matrix_targets: list[str],
    enable_forces: bool,
    require_exact_edge_match: bool,
    torch_dtype: torch.dtype,
) -> Path | None:
    if snapshot_cache_dir is None:
        return None
    cache_root = Path(snapshot_cache_dir).expanduser() / "preprocessed_samples"
    mat_stat = matrix_path.stat()
    info_stat = info_path.stat()
    key_payload = {
        "version": PREPROCESSED_SAMPLE_CACHE_VERSION,
        "matrix_path": str(matrix_path.resolve()),
        "info_path": str(info_path.resolve()),
        "matrix_mtime_ns": mat_stat.st_mtime_ns,
        "matrix_size": mat_stat.st_size,
        "info_mtime_ns": info_stat.st_mtime_ns,
        "info_size": info_stat.st_size,
        "orbital_cfg": mapper.orbital_cfg.to_dict(),
        "sh_irreps": str(sh_irreps),
        "n_radial": int(n_radial),
        "radial_embedding_scale": radial_embedding_scale,
        "training_unit": training_unit,
        "hamiltonian_scale_from_hartree": float(hamiltonian_scale_from_hartree),
        "cutoff_radius": float(cutoff_radius),
        "graph_cutoff_eps": GRAPH_NEIGHBORLIST_CUTOFF_EPS,
        "apply_cutoff_to_targets": bool(apply_cutoff_to_targets),
        "orbital_selection": _orbital_selection_cache_key(orbital_selection),
        "matrix_targets": list(matrix_targets),
        "enable_forces": bool(enable_forces),
        "require_exact_edge_match": bool(require_exact_edge_match),
        "torch_dtype": str(torch_dtype),
    }
    key_hash = hashlib.md5(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return cache_root / f"{matrix_path.stem}_{key_hash}.pt"


def load_preprocessed_sample_cache(path: Path) -> dict | None:
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            path.unlink()
        except Exception:
            pass
        return None


def save_preprocessed_sample_cache(path: Path, sample: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f"{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(sample, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def parse_orbital_selection(selection_raw: str | None) -> Any:
    if selection_raw is None:
        return None
    stripped = selection_raw.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    return selection_raw


def discover_snapshot_pairs(root: Path) -> list[tuple[Path, Path]]:
    """
    Discover (matrix, info) pairs under root.
    """
    matrix_candidates = set()
    for pattern in ("Si_DM", "*_DM", "*.matrix", "*.scfout"):
        for p in root.rglob(pattern):
            if p.is_file():
                matrix_candidates.add(p.resolve())

    pairs: list[tuple[Path, Path]] = []
    for mat in sorted(matrix_candidates):
        parent = mat.parent
        info_candidates = [
            parent / "info.dat",
            parent / "info.out",
            parent / "info.txt",
            parent / "Si.info.out",
        ]
        info_path = None
        for c in info_candidates:
            if c.exists():
                info_path = c
                break
        if info_path is None:
            # Fallback: first sibling file starting with "info".
            fallback = sorted(parent.glob("info*"))
            if fallback:
                info_path = fallback[0]
        if info_path is not None:
            pairs.append((mat, info_path.resolve()))
    return pairs


def parse_temperature_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    temps: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if p.endswith("K") or p.endswith("k"):
            p = p[:-1]
        temps.append(int(p))
    if len(temps) == 0:
        raise ValueError("Parsed --train-temps is empty")
    return temps


def discover_temperatures(root: Path) -> list[int]:
    temps: list[int] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.endswith("K"):
            continue
        t = name[:-1]
        if t.isdigit():
            temps.append(int(t))
    return sorted(set(temps))


def discover_snapshot_pairs_for_temp(root: Path, temp: int) -> list[tuple[Path, Path]]:
    temp_dir = root / f"{temp}K"
    if not temp_dir.exists():
        return []
    pairs: list[tuple[Path, Path]] = []
    for mat in sorted(temp_dir.glob("*/Si_DM")):
        if not mat.is_file():
            continue
        info = mat.parent / "info.dat"
        if not info.exists():
            fallback = sorted(mat.parent.glob("info*"))
            if not fallback:
                continue
            info = fallback[0]
        pairs.append((mat.resolve(), info.resolve()))
    return pairs


def get_diagonal_mask(edges_5d: torch.Tensor) -> torch.Tensor:
    sx, sy, sz, i, j = edges_5d[0], edges_5d[1], edges_5d[2], edges_5d[3], edges_5d[4]
    return (sx == 0) & (sy == 0) & (sz == 0) & (i == j)


def compute_edge_distances_by_key(
    block_matrix: BlockMatrix,
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """
    Compute PBC-aware edge distances for every key in a BlockMatrix.
    """
    out: dict[str, torch.Tensor] = {}
    for key, edges in block_matrix.pair_edges.items():
        sx, sy, sz = edges[0], edges[1], edges[2]
        src, dst = edges[3], edges[4]
        shift = torch.stack([sx, sy, sz], dim=1).to(positions.dtype)
        if box is not None:
            disp = positions[dst] - positions[src] + shift @ box
        else:
            disp = positions[dst] - positions[src]
        out[key] = torch.linalg.norm(disp, dim=1)
    return out


def canonicalize_block_matrix_edges(
    block_matrix: BlockMatrix,
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> BlockMatrix:
    order_dict: dict[str, torch.Tensor] = {}
    for key, edges_5d in block_matrix.pair_edges.items():
        edge_shift_local = edges_5d[:3]
        edge_index_local = edges_5d[3:]
        _, _, perm = canonicalize_edge_order(
            edge_index=edge_index_local,
            edge_shift=edge_shift_local,
            positions=positions,
            box=box,
        )
        order_dict[key] = perm
    return block_matrix.reorder_edges(order_dict)


def filter_block_matrix_by_cutoff(
    block_matrix: BlockMatrix,
    positions: torch.Tensor,
    box: torch.Tensor | None,
    cutoff_radius: float,
) -> BlockMatrix:
    mask_dict: dict[str, torch.Tensor] = {}
    for key, edges in block_matrix.pair_edges.items():
        sx, sy, sz = edges[0], edges[1], edges[2]
        src, dst = edges[3], edges[4]
        shift = torch.stack([sx, sy, sz], dim=1).to(positions.dtype)
        if box is not None:
            disp = positions[dst] - positions[src] + shift @ box
        else:
            disp = positions[dst] - positions[src]
        dist = torch.linalg.norm(disp, dim=1)
        mask_dict[key] = dist <= float(cutoff_radius)
    return block_matrix._apply_edge_mask(mask_dict, drop_empty=True)


def should_log_epoch(epoch_zero_based: int, log_interval: int, adaptive: bool) -> bool:
    log_interval = max(int(log_interval), 1)
    if not adaptive:
        return epoch_zero_based % log_interval == 0
    if epoch_zero_based <= 10:
        return True
    if epoch_zero_based < 100:
        return epoch_zero_based % 10 == 0
    return epoch_zero_based % log_interval == 0


GRAPH_NEIGHBORLIST_CUTOFF_EPS = 1e-6


def edge5_distance(
    edge: tuple[int, int, int, int, int],
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> float:
    sx, sy, sz, src, dst = edge
    src_t = torch.tensor(src, device=positions.device, dtype=torch.long)
    dst_t = torch.tensor(dst, device=positions.device, dtype=torch.long)
    disp = positions[dst_t] - positions[src_t]
    if box is not None:
        shift = torch.tensor(
            [sx, sy, sz], device=positions.device, dtype=positions.dtype
        )
        disp = disp + shift @ box
    return float(torch.linalg.norm(disp).item())


def block_matrix_edges_by_key(
    block_matrix: BlockMatrix,
) -> dict[str, list[tuple[int, int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int, int]]] = {}
    for key, edges_t in block_matrix.pair_edges.items():
        grouped[key] = [tuple(map(int, row)) for row in edges_t.t().tolist()]
    return grouped


def graph_edges_by_key(
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    atoms_list: list[str],
) -> dict[str, list[tuple[int, int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int, int]]] = {}
    n_edges = edge_index.shape[1]
    for e in range(n_edges):
        src = int(edge_index[0, e].item())
        dst = int(edge_index[1, e].item())
        key = f"{atoms_list[src]}-{atoms_list[dst]}"
        grouped.setdefault(key, []).append(
            (
                int(edge_shift[0, e].item()),
                int(edge_shift[1, e].item()),
                int(edge_shift[2, e].item()),
                src,
                dst,
            )
        )
    return grouped


def build_global_edge_tensors_from_pair_edges(
    pair_edges_by_key: dict[str, torch.Tensor],
    ordered_keys: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ordered_edges = [
        pair_edges_by_key[key]
        for key in ordered_keys
        if key in pair_edges_by_key and pair_edges_by_key[key].shape[1] > 0
    ]
    if not ordered_edges:
        raise RuntimeError("Cannot build graph edge tensors from empty pair_edges.")
    all_edges = torch.cat(ordered_edges, dim=1).to(device=device, dtype=torch.long)
    edge_index = all_edges[3:5]
    edge_shift = all_edges[:3]
    return edge_index, edge_shift


def reconcile_graph_edges_to_target(
    *,
    edge_index: torch.Tensor,
    edge_shift: torch.Tensor,
    reference_matrix: BlockMatrix,
    atoms_list: list[str],
    mapper: BlockIrrepMapper,
    positions: torch.Tensor,
    box: torch.Tensor | None,
    snapshot_label: str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    target_by_key = block_matrix_edges_by_key(reference_matrix)
    graph_by_key = graph_edges_by_key(edge_index, edge_shift, atoms_list)

    mismatch_lines: list[str] = []
    needs_fix = False
    all_keys = sorted(set(target_by_key) | set(graph_by_key))
    for key in all_keys:
        target_edges = target_by_key.get(key, [])
        graph_edges = graph_by_key.get(key, [])
        target_counter = Counter(target_edges)
        graph_counter = Counter(graph_edges)
        if target_counter == graph_counter:
            continue
        needs_fix = True
        missing_in_graph = list((target_counter - graph_counter).elements())
        extra_in_graph = list((graph_counter - target_counter).elements())
        if missing_in_graph:
            edge = missing_in_graph[0]
            mismatch_lines.append(
                "[GRAPH DIAGNOSTIC] "
                f"{snapshot_label} key={key} missing_in_graph "
                f"(sx,sy,sz,i,j,dist)=({edge[0]}, {edge[1]}, {edge[2]}, {edge[3]}, {edge[4]}, "
                f"{edge5_distance(edge, positions, box):.6f}) count={len(missing_in_graph)}"
            )
        if extra_in_graph:
            edge = extra_in_graph[0]
            mismatch_lines.append(
                "[GRAPH DIAGNOSTIC] "
                f"{snapshot_label} key={key} extra_in_graph "
                f"(sx,sy,sz,i,j,dist)=({edge[0]}, {edge[1]}, {edge[2]}, {edge[3]}, {edge[4]}, "
                f"{edge5_distance(edge, positions, box):.6f}) count={len(extra_in_graph)}"
            )

    if not needs_fix:
        return edge_index, edge_shift, False

    for line in mismatch_lines[:10]:
        print(line)
    print(
        "[GRAPH] Replacing graph edge list with target-matrix edge list "
        f"for {snapshot_label} so exact edge matching holds."
    )
    fixed_edge_index, fixed_edge_shift = build_global_edge_tensors_from_pair_edges(
        reference_matrix.pair_edges, mapper.edge_types, edge_index.device
    )
    return fixed_edge_index, fixed_edge_shift, True


def prepare_mapper_from_sample(
    x: dict,
    y: dict,
    orbital_selection: Any,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> BlockIrrepMapper:
    snap = Snapshot(
        hamiltonian=y["hamiltonian"],
        overlap=y["overlap"],
        density=y["density"],
        positions=x["positions"],
        box=x["box"],
    )
    if orbital_selection is not None:
        snap = snap.reduce_orbitals(orbital_selection)
    return BlockIrrepMapper(
        snap.hamiltonian.orbital_cfg, device=device, dtype=torch_dtype
    )


@contextmanager
def suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    old_stdout = sys.stdout
    sink = open(os.devnull, "w")
    try:
        sys.stdout = sink
        yield
    finally:
        sys.stdout = old_stdout
        sink.close()


def preprocess_sample(
    x: dict,
    y: dict,
    mapper: BlockIrrepMapper,
    sh_irreps: Irreps,
    n_radial: int,
    radial_embedding_scale: str,
    training_unit: str,
    hamiltonian_scale_from_hartree: float,
    cutoff_radius: float,
    apply_cutoff_to_targets: bool,
    orbital_selection: Any,
    matrix_targets: list[str],
    enable_forces: bool,
    require_exact_edge_match: bool,
    log_data: bool,
    log_model: bool,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> dict:
    snap = Snapshot(
        hamiltonian=y["hamiltonian"],
        overlap=y["overlap"],
        density=y["density"],
        positions=x["positions"],
        box=x["box"],
        forces=y.get("forces"),
    )

    if orbital_selection is not None:
        snap = snap.reduce_orbitals(orbital_selection)

    if log_data:
        log_snapshot_info(snap)

    positions = snap.positions.detach().to(device=device, dtype=torch_dtype)
    box = (
        snap.box.detach().to(device=device, dtype=torch_dtype)
        if snap.box is not None
        else None
    )
    atoms_list = list(snap.hamiltonian.atoms)
    atoms_tuple = tuple(atoms_list)
    atom_counts = Counter(atoms_list)

    H = snap.hamiltonian.to(device).detach() * float(hamiltonian_scale_from_hartree)
    S = snap.overlap.to(device).detach()
    D = snap.density.to(device).detach()
    if log_data:
        print(
            "  Converted Hamiltonian units: "
            f"Hartree -> {UNIT_DISPLAY_NAME[training_unit]} "
            f"(x{hamiltonian_scale_from_hartree:.7f})"
        )

    if log_model:
        log_orbital_config(mapper.orbital_cfg)
        log_mapper_info(mapper)

    if apply_cutoff_to_targets:
        before_edges = sum(edges.shape[1] for edges in H.pair_edges.values())
        H = filter_block_matrix_by_cutoff(H, positions, box, cutoff_radius)
        S = filter_block_matrix_by_cutoff(S, positions, box, cutoff_radius)
        D = filter_block_matrix_by_cutoff(D, positions, box, cutoff_radius)
        if log_data:
            after_edges = sum(edges.shape[1] for edges in H.pair_edges.values())
            log_cutoff_application(before_edges, after_edges, cutoff_radius)

    H = canonicalize_block_matrix_edges(H, positions, box)
    S = canonicalize_block_matrix_edges(S, positions, box)
    D = canonicalize_block_matrix_edges(D, positions, box)

    H = (H + H.transpose()) * 0.5
    target_matrices = {
        "hamiltonian": H,
        "overlap": S,
        "density": D,
    }
    target_matrices = {
        matrix_name: target_matrices[matrix_name] for matrix_name in matrix_targets
    }
    target_irreps_by_name = {
        matrix_name: target_matrices[matrix_name].to_vectors(mapper)
        for matrix_name in matrix_targets
    }
    target_hd_alignment = build_trace_alignment(H, D)
    target_ds_alignment = build_trace_alignment(D, S)
    energy_target = trace_matmul_sparse_block_matrix_aligned(H, D, target_hd_alignment)
    num_electrons_target = trace_matmul_sparse_block_matrix_aligned(
        D, S, target_ds_alignment
    )

    def compute_block_matrix_max_distance(block_matrix: BlockMatrix) -> float:
        max_dist = 0.0
        for edges_5d in block_matrix.pair_edges.values():
            if edges_5d.shape[1] == 0:
                continue
            src = edges_5d[3].long()
            dst = edges_5d[4].long()
            if box is not None:
                shift_float = edges_5d[:3].T.to(dtype=positions.dtype)
                edge_vec_local = positions[dst] - positions[src] + shift_float @ box
            else:
                edge_vec_local = positions[dst] - positions[src]
            edge_dist_local = torch.linalg.norm(edge_vec_local, dim=1)
            if edge_dist_local.numel() > 0:
                max_dist = max(max_dist, float(edge_dist_local.max().item()))
        return max_dist

    if log_data:
        target_max_by_matrix = {
            "hamiltonian": compute_block_matrix_max_distance(H),
            "overlap": compute_block_matrix_max_distance(S),
            "density": compute_block_matrix_max_distance(D),
        }
        print("\n  Target max edge distance by matrix:")
        for name, max_dist in target_max_by_matrix.items():
            print(f"    {name}: {max_dist:.6f} A")

    BOHR_TO_ANGSTROM = 0.529177210903
    force_scale = float(hamiltonian_scale_from_hartree) / BOHR_TO_ANGSTROM
    forces_target = None
    if enable_forces:
        if snap.forces is None:
            raise RuntimeError(
                "--enable-forces was set, but snapshot does not contain force targets."
            )
        forces_target = (
            snap.forces.detach().to(device=device, dtype=torch_dtype) * force_scale
        )
        if log_data:
            print(
                "  Converted force units: Hartree/Bohr -> selected training unit/Angstrom "
                f"(x{force_scale:.7f})"
            )

    ase_atoms = Atoms(
        symbols=atoms_list,
        positions=positions.detach().cpu().numpy(),
        cell=box.detach().cpu().numpy() if box is not None else None,
        pbc=box is not None,
    )
    graph_cutoff_radius = float(cutoff_radius) + GRAPH_NEIGHBORLIST_CUTOFF_EPS
    src_np, dst_np, offsets_np = neighbor_list(
        "ijS",
        ase_atoms,
        graph_cutoff_radius,
        self_interaction=False,
    )
    offdiag_counts = Counter(
        (
            int(offset[0]),
            int(offset[1]),
            int(offset[2]),
            int(src),
            int(dst),
        )
        for src, dst, offset in zip(
            src_np.tolist(), dst_np.tolist(), offsets_np.tolist()
        )
    )
    missing_reverse_edges: list[tuple[int, int, int, int, int]] = []
    for edge, count in offdiag_counts.items():
        sx, sy, sz, src_i, dst_i = edge
        reverse = (-sx, -sy, -sz, dst_i, src_i)
        reverse_count = offdiag_counts.get(reverse, 0)
        if reverse_count < count:
            missing_reverse_edges.extend([reverse] * (count - reverse_count))
    if missing_reverse_edges:
        missing_arr = np.asarray(missing_reverse_edges, dtype=np.int64)
        src_np = np.concatenate([src_np, missing_arr[:, 3]])
        dst_np = np.concatenate([dst_np, missing_arr[:, 4]])
        offsets_np = np.concatenate([offsets_np, missing_arr[:, :3]], axis=0)
        snapshot_label = (
            Path(str(snap.matrix_path)).name
            if snap.matrix_path is not None
            else "snapshot"
        )
        print(
            "[GRAPH] Added "
            f"{len(missing_reverse_edges)} missing reverse off-diagonal edge(s) "
            f"for {snapshot_label}."
        )

    num_atoms = len(atoms_list)
    self_src = torch.arange(num_atoms, dtype=torch.long, device=device)
    self_dst = torch.arange(num_atoms, dtype=torch.long, device=device)
    self_offsets = torch.zeros((num_atoms, 3), dtype=torch.long, device=device)
    src_offdiag = torch.from_numpy(src_np).to(device=device, dtype=torch.long)
    dst_offdiag = torch.from_numpy(dst_np).to(device=device, dtype=torch.long)
    offsets_offdiag = torch.from_numpy(offsets_np).to(device=device, dtype=torch.long)

    all_src = torch.cat([self_src, src_offdiag], dim=0)
    all_dst = torch.cat([self_dst, dst_offdiag], dim=0)
    all_offsets = torch.cat([self_offsets, offsets_offdiag], dim=0)

    edge_index = torch.stack([all_src, all_dst], dim=0)
    edge_shift = all_offsets.T
    edge_index, edge_shift, _ = canonicalize_edge_order(
        edge_index=edge_index,
        edge_shift=edge_shift,
        positions=positions,
        box=box,
    )

    if require_exact_edge_match:
        snapshot_label = (
            Path(str(snap.matrix_path)).name
            if snap.matrix_path is not None
            else "snapshot"
        )
        edge_index, edge_shift, graph_edges_fixed = reconcile_graph_edges_to_target(
            edge_index=edge_index,
            edge_shift=edge_shift,
            reference_matrix=H,
            atoms_list=atoms_list,
            mapper=mapper,
            positions=positions,
            box=box,
            snapshot_label=snapshot_label,
        )
        if graph_edges_fixed:
            print(
                "[GRAPH] Exact-match reconciliation applied "
                f"for {snapshot_label} (neighbor cutoff used {graph_cutoff_radius:.6f} A)."
            )

    num_self_edges = int(
        (
            (edge_index[0] == edge_index[1])
            & (edge_shift[0] == 0)
            & (edge_shift[1] == 0)
            & (edge_shift[2] == 0)
        )
        .sum()
        .item()
    )

    sh_non_scalar_slices = [
        sh_irreps.slices()[idx] for idx, (_, ir) in enumerate(sh_irreps) if ir.l > 0
    ]

    def compute_edge_features(curr_positions: torch.Tensor):
        if box is not None:
            shift_float_local = edge_shift.T.to(dtype=curr_positions.dtype)
            edge_vec_local = (
                curr_positions[edge_index[1]]
                - curr_positions[edge_index[0]]
                + shift_float_local @ box
            )
        else:
            edge_vec_local = (
                curr_positions[edge_index[1]] - curr_positions[edge_index[0]]
            )
        edge_dist_local = torch.linalg.norm(edge_vec_local, dim=1)
        edge_sh_local = spherical_harmonics(
            sh_irreps,
            edge_vec_local,
            normalize=True,
            normalization="component",
        )
        is_self_edge_local = (edge_index[0] == edge_index[1]) & (edge_shift == 0).all(
            dim=0
        )
        if is_self_edge_local.any() and sh_non_scalar_slices:
            edge_sh_local = edge_sh_local.clone()
            for slc in sh_non_scalar_slices:
                edge_sh_local[is_self_edge_local, slc] = 0.0
        edge_length_emb_local = soft_one_hot_linspace(
            edge_dist_local,
            start=0.0,
            end=cutoff_radius,
            number=n_radial,
            basis="gaussian",
            cutoff=False,
        )
        if radial_embedding_scale == "sqrt_n_radial":
            edge_length_emb_local = edge_length_emb_local * n_radial**0.5
        return edge_vec_local, edge_dist_local, edge_sh_local, edge_length_emb_local

    edge_vec, edge_dist, edge_sh, edge_length_emb = compute_edge_features(positions)

    if log_data:
        log_graph(
            atoms_list=atoms_list,
            positions=positions,
            box=box,
            cutoff_radius=cutoff_radius,
            src=src_np,
            dst=dst_np,
            offsets=offsets_np,
            edge_index=edge_index,
            num_self_edges=num_self_edges,
            edge_dist=edge_dist,
        )
        print(f"  Edge SH shape: {edge_sh.shape}")
        print(f"\n  Computing radial embeddings ({n_radial} basis functions)...")
        print(f"  Radial embedding scale: {radial_embedding_scale}")
        print(f"  Edge length embedding shape: {edge_length_emb.shape}")

    strict_reverse_edge_check(
        edge_index=edge_index, edge_shift=edge_shift, edge_set_name="graph"
    )

    element_to_idx = {
        elem: idx for idx, elem in enumerate(mapper.orbital_cfg.elements())
    }
    node_type_idx = torch.tensor(
        [element_to_idx[a] for a in atoms_list], dtype=torch.long, device=device
    )
    batch_node = torch.zeros(len(atoms_list), dtype=torch.long, device=device)
    batch_edge = torch.zeros(edge_index.shape[1], dtype=torch.long, device=device)
    edge_type_strs = [
        f"{atoms_list[edge_index[0, i].item()]}-{atoms_list[edge_index[1, i].item()]}"
        for i in range(edge_index.shape[1])
    ]
    edge_type_idx = torch.tensor(
        [mapper.edge_type2idx[edge_type] for edge_type in edge_type_strs],
        dtype=torch.long,
        device=device,
    )

    strict_edge_alignment_check(
        target_matrix=H,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="hamiltonian",
        require_exact=require_exact_edge_match,
    )
    strict_edge_alignment_check(
        target_matrix=S,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="overlap",
        require_exact=require_exact_edge_match,
    )
    strict_edge_alignment_check(
        target_matrix=D,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="density",
        require_exact=require_exact_edge_match,
    )
    if log_data:
        log_strict_checks_passed()

    pred_pair_edges_static = {}
    pred_lookup_static = {}
    pred_edge_keys = []
    target_basis_by_name = {
        matrix_name: target_matrices[matrix_name].basis
        for matrix_name in matrix_targets
    }
    for key in mapper.edge_types:
        type_idx = mapper.edge_type2idx[key]
        mask = edge_type_idx == type_idx
        if not bool(mask.any().item()):
            continue
        edges_5d = torch.cat([edge_shift[:, mask], edge_index[:, mask]], dim=0)
        pred_pair_edges_static[key] = edges_5d
        pred_edge_keys.append(key)
        for idx, edge_5d in enumerate(edges_5d.t().tolist()):
            sx, sy, sz, i, j = edge_5d
            pred_lookup_static[(int(sx), int(sy), int(sz), int(i), int(j))] = (
                key,
                idx,
            )

    pred_trace_alignment = build_trace_alignment_from_pair_edges(pred_pair_edges_static)

    return {
        "node_type_idx": node_type_idx,
        "edge_type_idx": edge_type_idx.to(device),
        "edge_index": edge_index.to(device),
        "edge_shift": edge_shift.to(device),
        "edge_length_emb": edge_length_emb.to(device),
        "edge_sh": edge_sh.to(device),
        "batch_node": batch_node,
        "batch_edge": batch_edge,
        "positions": positions,
        "box": box,
        "atoms_list": atoms_list,
        "atoms_tuple": atoms_tuple,
        "atom_counts": atom_counts,
        "target_matrices": target_matrices,
        "target_irreps_by_name": target_irreps_by_name,
        "target_basis_by_name": target_basis_by_name,
        "target_overlap": S,
        "target_density": D,
        "energy_target": energy_target,
        "num_electrons_target": num_electrons_target,
        "forces_target": forces_target,
        "pred_pair_edges_static": pred_pair_edges_static,
        "pred_lookup_static": pred_lookup_static,
        "pred_edge_keys": tuple(pred_edge_keys),
        "pred_edge_key_set": set(pred_edge_keys),
        "pred_trace_alignment": pred_trace_alignment,
        "sh_non_scalar_slices": sh_non_scalar_slices,
    }


def preprocess_dataset_samples(
    dataset,
    *,
    mapper: BlockIrrepMapper,
    sh_irreps: Irreps,
    n_radial: int,
    radial_embedding_scale: str,
    training_unit: str,
    hamiltonian_scale_from_hartree: float,
    cutoff_radius: float,
    apply_cutoff_to_targets: bool,
    orbital_selection: Any,
    matrix_targets: list[str],
    enable_forces: bool,
    require_exact_edge_match: bool,
    device: torch.device,
    torch_dtype: torch.dtype,
    log_first_sample: bool,
    log_model_first_sample: bool,
    progress_desc: str,
) -> list[dict]:
    jobs = [(idx, x, y) for idx, (x, y) in enumerate(dataset)]
    if not jobs:
        return []
    snapshot_paths = getattr(dataset, "snapshot_paths", None)
    dataset_cfg = getattr(dataset, "cfg", None)
    snapshot_cache_dir = (
        getattr(dataset_cfg, "snapshot_cache_dir", None)
        if dataset_cfg is not None
        else None
    )
    preprocessed_cache_hits = 0
    preprocessed_cache_misses = 0

    def _build_sample(job: tuple[int, dict, dict]) -> dict:
        nonlocal preprocessed_cache_hits, preprocessed_cache_misses
        idx, x, y = job
        cache_file = None
        if snapshot_paths is not None and idx < len(snapshot_paths):
            matrix_path, info_path = snapshot_paths[idx]
            cache_file = get_preprocessed_sample_cache_file(
                snapshot_cache_dir=snapshot_cache_dir,
                matrix_path=matrix_path,
                info_path=info_path,
                mapper=mapper,
                sh_irreps=sh_irreps,
                n_radial=n_radial,
                radial_embedding_scale=radial_embedding_scale,
                training_unit=training_unit,
                hamiltonian_scale_from_hartree=hamiltonian_scale_from_hartree,
                cutoff_radius=cutoff_radius,
                apply_cutoff_to_targets=apply_cutoff_to_targets,
                orbital_selection=orbital_selection,
                matrix_targets=matrix_targets,
                enable_forces=enable_forces,
                require_exact_edge_match=require_exact_edge_match,
                torch_dtype=torch_dtype,
            )
            if cache_file is not None and cache_file.exists():
                cached_sample = load_preprocessed_sample_cache(cache_file)
                if cached_sample is not None:
                    preprocessed_cache_hits += 1
                    if idx == 0 and (log_first_sample or log_model_first_sample):
                        print(
                            f"[CACHE] Loaded first preprocessed sample from {cache_file}"
                        )
                    return cached_sample
        preprocessed_cache_misses += 1
        return preprocess_sample(
            x=x,
            y=y,
            mapper=mapper,
            sh_irreps=sh_irreps,
            n_radial=n_radial,
            radial_embedding_scale=radial_embedding_scale,
            training_unit=training_unit,
            hamiltonian_scale_from_hartree=hamiltonian_scale_from_hartree,
            cutoff_radius=cutoff_radius,
            apply_cutoff_to_targets=apply_cutoff_to_targets,
            orbital_selection=orbital_selection,
            matrix_targets=matrix_targets,
            enable_forces=enable_forces,
            require_exact_edge_match=require_exact_edge_match,
            log_data=log_first_sample and idx == 0,
            log_model=log_model_first_sample and idx == 0,
            device=device,
            torch_dtype=torch_dtype,
        )

    samples = []
    for job in tqdm(jobs, desc=progress_desc):
        sample = _build_sample(job)
        idx = job[0]
        if snapshot_paths is not None and idx < len(snapshot_paths):
            matrix_path, info_path = snapshot_paths[idx]
            cache_file = get_preprocessed_sample_cache_file(
                snapshot_cache_dir=snapshot_cache_dir,
                matrix_path=matrix_path,
                info_path=info_path,
                mapper=mapper,
                sh_irreps=sh_irreps,
                n_radial=n_radial,
                radial_embedding_scale=radial_embedding_scale,
                training_unit=training_unit,
                hamiltonian_scale_from_hartree=hamiltonian_scale_from_hartree,
                cutoff_radius=cutoff_radius,
                apply_cutoff_to_targets=apply_cutoff_to_targets,
                orbital_selection=orbital_selection,
                matrix_targets=matrix_targets,
                enable_forces=enable_forces,
                require_exact_edge_match=require_exact_edge_match,
                torch_dtype=torch_dtype,
            )
            if cache_file is not None and not cache_file.exists():
                save_preprocessed_sample_cache(cache_file, sample)
        samples.append(sample)
    if snapshot_cache_dir is None:
        print(
            f"[CACHE] {progress_desc}: cache disabled, processed {len(samples)} sample(s)"
        )
    else:
        total = preprocessed_cache_hits + preprocessed_cache_misses
        print(
            f"[CACHE] {progress_desc}: "
            f"{preprocessed_cache_hits} hit(s), {preprocessed_cache_misses} miss(es), "
            f"{total} total"
        )
    return samples


def recompute_sample_edge_features(
    sample: dict,
    positions: torch.Tensor,
    *,
    sh_irreps: Irreps,
    cutoff_radius: float,
    n_radial: int,
    radial_embedding_scale: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index = sample["edge_index"]
    edge_shift = sample["edge_shift"]
    box = sample["box"]
    if box is not None:
        shift_float = edge_shift.T.to(dtype=positions.dtype)
        edge_vec = (
            positions[edge_index[1]] - positions[edge_index[0]] + shift_float @ box
        )
    else:
        edge_vec = positions[edge_index[1]] - positions[edge_index[0]]
    edge_dist = torch.linalg.norm(edge_vec, dim=1)
    edge_sh = spherical_harmonics(
        sh_irreps,
        edge_vec,
        normalize=True,
        normalization="component",
    )
    is_self_edge = (edge_index[0] == edge_index[1]) & (edge_shift == 0).all(dim=0)
    if is_self_edge.any() and sample["sh_non_scalar_slices"]:
        edge_sh = edge_sh.clone()
        for slc in sample["sh_non_scalar_slices"]:
            edge_sh[is_self_edge, slc] = 0.0
    edge_length_emb = soft_one_hot_linspace(
        edge_dist,
        start=0.0,
        end=cutoff_radius,
        number=n_radial,
        basis="gaussian",
        cutoff=False,
    )
    if radial_embedding_scale == "sqrt_n_radial":
        edge_length_emb = edge_length_emb * n_radial**0.5
    return edge_sh, edge_length_emb, edge_dist


def move_sample_to_device(sample: dict, device: torch.device) -> dict:
    def move_value(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, BlockMatrix):
            return value.to(device, non_blocking=True)
        if isinstance(value, IrrepsBlockData):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {k: move_value(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(move_value(v) for v in value)
        if isinstance(value, list):
            return [move_value(v) for v in value]
        return value

    moved = {}
    for key, value in sample.items():
        if key == "pred_lookup_static":
            moved[key] = value
        elif key == "sh_non_scalar_slices":
            moved[key] = list(value)
        elif key == "pred_edge_keys":
            moved[key] = tuple(value)
        elif key == "pred_edge_key_set":
            moved[key] = set(value)
        elif key == "pred_trace_alignment":
            moved[key] = {
                pair_key: (rev_key, idx.to(device, non_blocking=True))
                for pair_key, (rev_key, idx) in value.items()
            }
        else:
            moved[key] = move_value(value)
    return moved


def pin_sample_memory(sample: dict) -> dict:
    def pin_value(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.pin_memory()
        if isinstance(value, BlockMatrix):
            return value.pin_memory()
        if isinstance(value, IrrepsBlockData):
            return value.pin_memory()
        if isinstance(value, dict):
            return {k: pin_value(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(pin_value(v) for v in value)
        if isinstance(value, list):
            return [pin_value(v) for v in value]
        return value

    pinned = {}
    for key, value in sample.items():
        if key == "pred_lookup_static":
            pinned[key] = value
        elif key == "sh_non_scalar_slices":
            pinned[key] = list(value)
        elif key == "pred_edge_keys":
            pinned[key] = tuple(value)
        elif key == "pred_edge_key_set":
            pinned[key] = set(value)
        else:
            pinned[key] = pin_value(value)
    return pinned


def move_small_sample_tensors_to_device(sample: dict, device: torch.device) -> dict:
    moved = dict(sample)
    for key in ("positions", "box", "forces_target"):
        value = moved.get(key)
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
    pred_trace_alignment = moved.get("pred_trace_alignment")
    if pred_trace_alignment is not None:
        moved["pred_trace_alignment"] = {
            pair_key: (rev_key, idx.to(device, non_blocking=True))
            for pair_key, (rev_key, idx) in pred_trace_alignment.items()
        }
    return moved


def wrap_raw_predictions_to_irreps(
    pred_raw_by_matrix: dict[str, dict],
    sample: dict,
    mapper: BlockIrrepMapper,
    matrix_targets: list[str],
) -> dict[str, IrrepsBlockData]:
    pred_irreps_by_name = {}
    pred_edge_key_set = sample["pred_edge_key_set"]
    pred_edge_keys = sample["pred_edge_keys"]
    pred_pair_edges_static = sample["pred_pair_edges_static"]
    pred_lookup_static = sample["pred_lookup_static"]
    for matrix_name in matrix_targets:
        raw_matrix = pred_raw_by_matrix[matrix_name]
        raw_key_set = set(raw_matrix.keys())
        if raw_key_set != pred_edge_key_set:
            missing_keys = sorted(pred_edge_key_set - raw_key_set)
            extra_keys = sorted(raw_key_set - pred_edge_key_set)
            raise RuntimeError(
                f"Raw prediction keys for '{matrix_name}' do not match graph keys. "
                f"Missing: {missing_keys}, extra: {extra_keys}"
            )
        pair_vec = {key: raw_matrix[key]["vectors"] for key in pred_edge_keys}
        pred_irreps_by_name[matrix_name] = IrrepsBlockData(
            atoms=sample["atoms_tuple"],
            atom_counts=sample["atom_counts"],
            pair_vectors=pair_vec,
            pair_edges=pred_pair_edges_static,
            lookup=pred_lookup_static,
            orbital_cfg=mapper.orbital_cfg,
            basis=sample["target_basis_by_name"][matrix_name],
        )
    return pred_irreps_by_name


def compute_forces_from_pred_matrices(
    pred_matrix_by_name: dict[str, BlockMatrix],
    positions_tensor: torch.Tensor,
    pred_trace_alignment,
    *,
    create_graph: bool,
    retain_graph: bool,
) -> torch.Tensor:
    energy = trace_matmul_sparse_block_matrix_aligned(
        pred_matrix_by_name["hamiltonian"],
        pred_matrix_by_name["density"],
        pred_trace_alignment,
    )
    grad_pos = torch.autograd.grad(
        energy,
        positions_tensor,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )[0]
    return -grad_pos


def compute_loss_for_sample(
    pred_irreps_by_name: dict[str, IrrepsBlockData],
    pred_matrix_norm_by_name: dict[str, BlockMatrix],
    sample: dict,
    *,
    matrix_targets: list[str],
    loss_coef_density_matrix: float,
    enable_energy: bool,
    train_on_energy: bool,
    enable_num_electrons: bool,
    train_on_num_electrons: bool,
    loss_coef_observables: float,
    enable_forces: bool,
    train_on_forces: bool,
    loss_coef_forces: float,
    positions_for_forces: torch.Tensor | None,
    force_loss_requires_grad: bool,
) -> dict[str, Any]:
    matrix_block_losses: dict[str, torch.Tensor] = {}

    def mse_loss_blocks(
        pred_blocks: torch.Tensor,
        targ_blocks: torch.Tensor,
        edges_5d: torch.Tensor,
    ) -> torch.Tensor:
        min_n = min(pred_blocks.shape[0], targ_blocks.shape[0], edges_5d.shape[1])
        p = pred_blocks[:min_n]
        t = targ_blocks[:min_n]
        return F.mse_loss(p, t)

    for matrix_name in matrix_targets:
        target_matrix = sample["target_matrices"][matrix_name]
        pred_matrix = pred_matrix_norm_by_name[matrix_name]
        loss_total = torch.tensor(0.0, device=sample["node_type_idx"].device)
        for key in target_matrix.pair_blocks.keys():
            if key not in pred_matrix.pair_blocks:
                continue
            loss_total = loss_total + mse_loss_blocks(
                pred_matrix.pair_blocks[key],
                target_matrix.pair_blocks[key],
                target_matrix.pair_edges[key],
            )
        if matrix_name == "density":
            loss_total = loss_total * float(loss_coef_density_matrix)
        matrix_block_losses[matrix_name] = loss_total

    loss_block = (
        sum(matrix_block_losses.values())
        if matrix_block_losses
        else torch.tensor(0.0, device=sample["node_type_idx"].device)
    )
    pred_matrix_observables_by_name = pred_matrix_norm_by_name

    loss_energy_weighted = torch.tensor(0.0, device=sample["node_type_idx"].device)
    loss_num_electrons_weighted = torch.tensor(
        0.0, device=sample["node_type_idx"].device
    )
    loss_forces_weighted = torch.tensor(0.0, device=sample["node_type_idx"].device)
    energy_mae = None
    num_electrons_mae = None
    forces_mae = None
    forces_mse = None

    if (
        enable_energy
        and "hamiltonian" in pred_matrix_observables_by_name
        and "density" in pred_matrix_observables_by_name
    ):
        energy_pred = trace_matmul_sparse_block_matrix_aligned(
            pred_matrix_observables_by_name["hamiltonian"],
            pred_matrix_observables_by_name["density"],
            sample["pred_trace_alignment"],
        )
        energy_mae = torch.abs(energy_pred - sample["energy_target"])
        if train_on_energy:
            loss_energy_weighted = loss_coef_observables * F.mse_loss(
                energy_pred, sample["energy_target"]
            )

    if (
        enable_num_electrons
        and "overlap" in pred_matrix_observables_by_name
        and "density" in pred_matrix_observables_by_name
    ):
        num_electrons_pred = trace_matmul_sparse_block_matrix_aligned(
            pred_matrix_observables_by_name["density"],
            pred_matrix_observables_by_name["overlap"],
            sample["pred_trace_alignment"],
        )
        num_electrons_mae = torch.abs(
            num_electrons_pred - sample["num_electrons_target"]
        )
        if train_on_num_electrons:
            loss_num_electrons_weighted = loss_coef_observables * F.mse_loss(
                num_electrons_pred, sample["num_electrons_target"]
            )

    if (
        enable_forces
        and positions_for_forces is not None
        and sample["forces_target"] is not None
        and "hamiltonian" in pred_matrix_observables_by_name
        and "density" in pred_matrix_observables_by_name
    ):
        forces_pred = compute_forces_from_pred_matrices(
            pred_matrix_observables_by_name,
            positions_for_forces,
            sample["pred_trace_alignment"],
            create_graph=force_loss_requires_grad,
            retain_graph=force_loss_requires_grad,
        )
        forces_err = forces_pred - sample["forces_target"]
        forces_mae = torch.mean(torch.abs(forces_err))
        forces_mse = torch.mean(forces_err**2)
        if train_on_forces:
            loss_forces_weighted = loss_coef_forces * forces_mse

    loss_total = (
        loss_block
        + loss_energy_weighted
        + loss_num_electrons_weighted
        + loss_forces_weighted
    )
    return {
        "loss_total": loss_total,
        "loss_block_total": loss_block,
        "matrix_block_losses": matrix_block_losses,
        "loss_energy_weighted": loss_energy_weighted,
        "loss_num_electrons_weighted": loss_num_electrons_weighted,
        "loss_forces_weighted": loss_forces_weighted,
        "energy_mae": energy_mae,
        "num_electrons_mae": num_electrons_mae,
        "num_electrons_mae_pre_correction": num_electrons_mae,
        "forces_mae": forces_mae,
        "forces_mse": forces_mse,
    }


def predict_sample(
    network: MinimalNetwork,
    sample: dict,
    mapper: BlockIrrepMapper,
    *,
    matrix_targets: list[str],
    sh_irreps: Irreps,
    cutoff_radius: float,
    n_radial: int,
    radial_embedding_scale: str,
    symmetrize_preds: bool,
    rescale_density_to_num_electrons: bool,
    verbose_forward: bool,
    log_forward: bool,
    log_to_wandb: bool = False,
    positions_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    if positions_override is None:
        edge_sh = sample["edge_sh"]
        edge_length_emb = sample["edge_length_emb"]
    else:
        edge_sh, edge_length_emb, _ = recompute_sample_edge_features(
            sample,
            positions_override,
            sh_irreps=sh_irreps,
            cutoff_radius=cutoff_radius,
            n_radial=n_radial,
            radial_embedding_scale=radial_embedding_scale,
        )

    with suppress_stdout(not verbose_forward and not log_forward):
        raw = network(
            sample["node_type_idx"],
            sample["edge_type_idx"],
            sample["edge_index"],
            sample["edge_shift"],
            edge_length_emb,
            edge_sh,
            sample["batch_node"],
            sample["batch_edge"],
            log_to_wandb=log_to_wandb,
            verbose=verbose_forward or log_forward,
        )

    missing_targets = [name for name in matrix_targets if name not in raw]
    if missing_targets:
        raise RuntimeError(
            f"Network output is missing targets: {missing_targets}. "
            f"Got: {sorted(list(raw.keys()))}"
        )

    pred_irreps_by_name = wrap_raw_predictions_to_irreps(
        raw, sample, mapper, matrix_targets
    )
    pred_matrix_norm_by_name = {
        matrix_name: pred_irreps_by_name[matrix_name].to_blocks(mapper)
        for matrix_name in matrix_targets
    }
    if symmetrize_preds:
        pred_matrix_metrics_by_name = {
            matrix_name: pred_matrix_norm_by_name[matrix_name].symmetrize_aligned(
                sample["pred_trace_alignment"]
            )
            for matrix_name in matrix_targets
        }
        pred_irreps_metrics_by_name = {
            matrix_name: pred_matrix_metrics_by_name[matrix_name].to_vectors(mapper)
            for matrix_name in matrix_targets
        }
    else:
        pred_matrix_metrics_by_name = dict(pred_matrix_norm_by_name)
        pred_irreps_metrics_by_name = dict(pred_irreps_by_name)

    density_rescale_factor = None
    num_electrons_pred_pre_correction = None
    num_electrons_mae_pre_correction = None
    if (
        rescale_density_to_num_electrons
        and "density" in pred_matrix_metrics_by_name
        and "overlap" in pred_matrix_metrics_by_name
    ):
        num_electrons_pred_pre_correction = trace_matmul_sparse_block_matrix_aligned(
            pred_matrix_metrics_by_name["density"],
            pred_matrix_metrics_by_name["overlap"],
            sample["pred_trace_alignment"],
        )
        num_electrons_mae_pre_correction = torch.abs(
            num_electrons_pred_pre_correction - sample["num_electrons_target"]
        )
        pred_safe = torch.where(
            torch.abs(num_electrons_pred_pre_correction) > 1e-12,
            num_electrons_pred_pre_correction,
            torch.full_like(num_electrons_pred_pre_correction, 1e-12),
        )
        density_rescale_factor = sample["num_electrons_target"] / pred_safe
        pred_matrix_metrics_by_name["density"] = pred_matrix_metrics_by_name[
            "density"
        ] * float(density_rescale_factor.item())
        pred_irreps_metrics_by_name["density"] = pred_matrix_metrics_by_name[
            "density"
        ].to_vectors(mapper)

    return {
        "raw": raw,
        "pred_irreps_by_name": pred_irreps_by_name,
        "pred_irreps_metrics_by_name": pred_irreps_metrics_by_name,
        "pred_matrix_norm_by_name": pred_matrix_norm_by_name,
        "pred_matrix_metrics_by_name": pred_matrix_metrics_by_name,
        "density_rescale_factor": density_rescale_factor,
        "num_electrons_pred_pre_correction": num_electrons_pred_pre_correction,
        "num_electrons_mae_pre_correction": num_electrons_mae_pre_correction,
    }


def evaluate_split(
    network: MinimalNetwork,
    samples: list[dict],
    mapper: BlockIrrepMapper,
    mapper_cpu: BlockIrrepMapper,
    all_irreps: list,
    device: torch.device,
    *,
    matrix_targets: list[str],
    sh_irreps: Irreps,
    cutoff_radius: float,
    n_radial: int,
    radial_embedding_scale: str,
    symmetrize_preds: bool,
    rescale_density_to_num_electrons: bool,
    enable_energy: bool,
    train_on_energy: bool,
    enable_num_electrons: bool,
    train_on_num_electrons: bool,
    loss_coef_density_matrix: float,
    loss_coef_observables: float,
    enable_forces: bool,
    train_on_forces: bool,
    loss_coef_forces: float,
) -> dict:
    empty_detailed = {
        "mae": 0.0,
        "mse": 0.0,
        "mae_mod": 0.0,
        "mse_mod": 0.0,
        "mu_H": 0.0,
        "correction_mae": 0.0,
        "correction_mse": 0.0,
    }
    if not samples:
        return {
            "loss": 0.0,
            "detailed": empty_detailed,
            "basic_by_name": {},
            "per_irrep_by_name": {},
            "forces_mae": None,
            "forces_mse": None,
            "energy_mae": None,
            "num_electrons_mae": None,
            "num_electrons_mae_pre_correction": None,
            "first_pred_metrics_by_name": None,
            "first_sample": None,
        }

    network.eval()
    total_loss = 0.0
    detailed_sum: dict[str, float] = {}
    basic_sum_by_name: dict[str, dict[str, float]] = {}
    per_irrep_sum_by_name: dict[str, dict[str, float]] = {}
    forces_mae_sum = 0.0
    forces_mse_sum = 0.0
    forces_count = 0
    energy_mae_sum = 0.0
    energy_count = 0
    num_electrons_mae_sum = 0.0
    num_electrons_count = 0
    num_electrons_mae_pre_correction_sum = 0.0
    num_electrons_pre_correction_count = 0
    first_pred_metrics_by_name = None
    first_sample = None

    for idx, sample in enumerate(samples):
        sample_dev = move_sample_to_device(sample, device)
        positions_eval = None
        if enable_forces:
            positions_eval = (
                sample_dev["positions"].detach().clone().requires_grad_(True)
            )
        pred = predict_sample(
            network,
            sample_dev,
            mapper,
            matrix_targets=matrix_targets,
            sh_irreps=sh_irreps,
            cutoff_radius=cutoff_radius,
            n_radial=n_radial,
            radial_embedding_scale=radial_embedding_scale,
            symmetrize_preds=symmetrize_preds,
            rescale_density_to_num_electrons=rescale_density_to_num_electrons,
            verbose_forward=False,
            log_forward=False,
            positions_override=positions_eval,
        )
        loss_info = compute_loss_for_sample(
            pred["pred_irreps_by_name"],
            pred["pred_matrix_norm_by_name"],
            sample_dev,
            matrix_targets=matrix_targets,
            loss_coef_density_matrix=loss_coef_density_matrix,
            enable_energy=enable_energy,
            train_on_energy=train_on_energy,
            enable_num_electrons=enable_num_electrons,
            train_on_num_electrons=train_on_num_electrons,
            loss_coef_observables=loss_coef_observables,
            enable_forces=enable_forces,
            train_on_forces=train_on_forces,
            loss_coef_forces=loss_coef_forces,
            positions_for_forces=positions_eval,
            force_loss_requires_grad=False,
        )
        total_loss += float(loss_info["loss_total"].item())

        pred_metrics_by_name = {
            matrix_name: matrix.detach().to("cpu")
            for matrix_name, matrix in pred["pred_matrix_metrics_by_name"].items()
        }
        pred_irreps_metrics_by_name = {
            matrix_name: data.detach().to("cpu")
            for matrix_name, data in pred["pred_irreps_metrics_by_name"].items()
        }
        target_matrices_detached = {
            matrix_name: matrix.detach().to("cpu")
            for matrix_name, matrix in sample_dev["target_matrices"].items()
        }
        target_irreps_detached = {
            matrix_name: data.detach().to("cpu")
            for matrix_name, data in sample_dev["target_irreps_by_name"].items()
        }
        target_overlap_detached = sample_dev["target_overlap"].detach().to("cpu")

        with torch.no_grad():
            detailed = compute_detailed_metrics_aligned(
                pred_metrics_by_name["hamiltonian"],
                target_matrices_detached["hamiltonian"],
                target_overlap_detached,
            )
            for k, v in detailed.items():
                detailed_sum[k] = detailed_sum.get(k, 0.0) + float(v)

            for matrix_name in matrix_targets:
                target_matrix = target_matrices_detached[matrix_name]
                pred_matrix_metrics = pred_metrics_by_name[matrix_name]
                basic = compute_basic_matrix_metrics_aligned(
                    pred_matrix_metrics, target_matrix
                )
                basic_acc = basic_sum_by_name.setdefault(matrix_name, {})
                for k, v in basic.items():
                    basic_acc[k] = basic_acc.get(k, 0.0) + float(v)

                per_irrep = compute_irrep_metrics(
                    pred_irreps_metrics_by_name[matrix_name],
                    target_irreps_detached[matrix_name],
                    all_irreps,
                    mapper_cpu,
                )
                per_irrep_acc = per_irrep_sum_by_name.setdefault(matrix_name, {})
                for k, v in per_irrep.items():
                    per_irrep_acc[k] = per_irrep_acc.get(k, 0.0) + float(v)

        if loss_info["forces_mae"] is not None and loss_info["forces_mse"] is not None:
            forces_mae_sum += float(loss_info["forces_mae"].item())
            forces_mse_sum += float(loss_info["forces_mse"].item())
            forces_count += 1
        if loss_info["energy_mae"] is not None:
            energy_mae_sum += float(loss_info["energy_mae"].item())
            energy_count += 1
        if loss_info["num_electrons_mae"] is not None:
            num_electrons_mae_sum += float(loss_info["num_electrons_mae"].item())
            num_electrons_count += 1
        if pred["num_electrons_mae_pre_correction"] is not None:
            num_electrons_mae_pre_correction_sum += float(
                pred["num_electrons_mae_pre_correction"].item()
            )
            num_electrons_pre_correction_count += 1

        if idx == 0:
            first_pred_metrics_by_name = pred_metrics_by_name
            first_sample = {
                **sample_dev,
                "positions": sample_dev["positions"].detach().to("cpu"),
                "box": (
                    sample_dev["box"].detach().to("cpu")
                    if sample_dev["box"] is not None
                    else None
                ),
                "target_overlap": sample_dev["target_overlap"].detach().to("cpu"),
                "target_matrices": target_matrices_detached,
                "target_irreps_by_name": target_irreps_detached,
            }

    n = float(len(samples))
    return {
        "loss": total_loss / n,
        "detailed": {k: v / n for k, v in detailed_sum.items()},
        "basic_by_name": {
            matrix_name: {k: v / n for k, v in metrics.items()}
            for matrix_name, metrics in basic_sum_by_name.items()
        },
        "per_irrep_by_name": {
            matrix_name: {k: v / n for k, v in metrics.items()}
            for matrix_name, metrics in per_irrep_sum_by_name.items()
        },
        "forces_mae": (forces_mae_sum / forces_count) if forces_count > 0 else None,
        "forces_mse": (forces_mse_sum / forces_count) if forces_count > 0 else None,
        "energy_mae": (energy_mae_sum / energy_count) if energy_count > 0 else None,
        "num_electrons_mae": (
            num_electrons_mae_sum / num_electrons_count
            if num_electrons_count > 0
            else None
        ),
        "num_electrons_mae_pre_correction": (
            num_electrons_mae_pre_correction_sum / num_electrons_pre_correction_count
            if num_electrons_pre_correction_count > 0
            else None
        ),
        "first_pred_metrics_by_name": first_pred_metrics_by_name,
        "first_sample": first_sample,
    }


def main() -> None:
    parser, args = parse_args()
    explicit_overrides = _get_explicit_cli_overrides(parser, sys.argv[1:])
    resume_checkpoint = None
    resume_payload: dict[str, Any] | None = None
    resume_config: dict[str, Any] = {}
    resume_changes: dict[str, tuple[Any, Any]] = {}
    reset_scheduler_state = False
    start_epoch = 0
    train_end_epoch = 0
    resume_source_kind = "none"
    continue_wandb_run = args.resume_from_run_id is not None and not args.fresh_run
    resolved_wandb_entity = None

    if args.resume_from_run_id is not None:
        resolved_wandb_entity = _resolve_wandb_entity(args.wandb_entity)

    if args.resume_from_checkpoint is not None:
        resume_checkpoint = _resolve_resume_checkpoint_path(args.resume_from_checkpoint)
    elif args.resume_from_run_id is not None:
        resume_checkpoint = _resolve_resume_checkpoint_from_run_id(
            args.resume_from_run_id,
            project=args.checkpoint_source_project or args.wandb_project,
            entity=resolved_wandb_entity,
        )

    if resume_checkpoint is not None:
        resume_payload = _load_checkpoint(resume_checkpoint)
        resume_config = dict(resume_payload.get("config") or {})
        if not resume_config:
            raise ValueError(
                f"Checkpoint {resume_checkpoint} does not contain a usable config."
            )
        args = _apply_checkpoint_config_overrides(
            args, resume_config, explicit_overrides
        )
        resume_changes = _summarize_resume_changes(
            explicit_overrides, resume_config, args
        )
        architecture_changes = {
            k: v for k, v in resume_changes.items() if k in ARCHITECTURE_ARG_NAMES
        }
        forbidden_changes = {
            k: v
            for k, v in resume_changes.items()
            if k in FORBIDDEN_RESUME_OVERRIDE_ARG_NAMES
        }
        if architecture_changes or forbidden_changes:
            all_conflicts = {**architecture_changes, **forbidden_changes}
            details = ", ".join(
                f"{name}: {old!r} -> {new!r}"
                for name, (old, new) in sorted(all_conflicts.items())
            )
            raise ValueError(
                "Cannot resume with architecture-affecting or forbidden overrides. "
                f"Conflicts: {details}"
            )
        reset_scheduler_state = bool(
            set(resume_changes)
            & (DATA_OVERRIDE_ARG_NAMES | OBJECTIVE_OVERRIDE_ARG_NAMES)
        ) or bool(set(resume_changes) & SCHEDULER_OVERRIDE_ARG_NAMES)
        start_epoch = int(resume_payload.get("epoch", -1)) + 1
        train_end_epoch = start_epoch + max(int(args.num_epochs), 0)
        resume_source_kind = (
            "latest" if resume_checkpoint.name == "latest_checkpoint.pt" else "legacy"
        )
        if args.resume_from_run_id is not None and args.fresh_run:
            resume_source_kind = f"{resume_source_kind}+fresh_run_from_run_id"
    else:
        train_end_epoch = max(int(args.num_epochs), 0)

    valid_matrix_targets = {"hamiltonian", "overlap", "density"}
    matrix_targets = list(dict.fromkeys(args.matrix_targets))
    invalid_targets = sorted(set(matrix_targets) - valid_matrix_targets)
    if invalid_targets:
        raise ValueError(
            f"Invalid --matrix-targets entries: {invalid_targets}. "
            f"Valid targets: {sorted(valid_matrix_targets)}"
        )
    if "hamiltonian" not in matrix_targets:
        raise ValueError("Hamiltonian must be included in --matrix-targets.")
    if args.head_e3mlp_layers < 1:
        raise ValueError("--head-e3mlp-layers must be >= 1.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1.")
    if args.loss_coef_observables == 0.0 and (
        args.train_on_energy or args.train_on_num_electrons
    ):
        raise ValueError(
            "If training on energy or num electrons, loss-coef-observables must be nonzero."
        )
    if args.train_on_energy and not {"hamiltonian", "density"}.issubset(
        set(matrix_targets)
    ):
        raise ValueError(
            "--train-on-energy requires hamiltonian and density in --matrix-targets."
        )
    if args.train_on_num_electrons and not {"overlap", "density"}.issubset(
        set(matrix_targets)
    ):
        raise ValueError(
            "--train-on-num-electrons requires overlap and density in --matrix-targets."
        )
    if args.train_on_forces and not args.enable_forces:
        raise ValueError("--train-on-forces requires --enable-forces=True.")
    if args.train_on_forces and args.loss_coef_forces == 0.0:
        raise ValueError("If training on forces, loss-coef-forces must be nonzero.")
    if args.train_on_forces and not {"hamiltonian", "density"}.issubset(
        set(matrix_targets)
    ):
        raise ValueError(
            "--train-on-forces requires hamiltonian and density in --matrix-targets."
        )
    if args.rescale_density_to_num_electrons and not {"overlap", "density"}.issubset(
        set(matrix_targets)
    ):
        raise ValueError(
            "--rescale-density-to-num-electrons requires overlap and density in --matrix-targets."
        )
    torch_dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(torch_dtype)
    if args.device.startswith("cuda") and args.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    args.seed = choose_runtime_seed(args.seed, args.randomize_seed)
    set_seed(args.seed)

    print("=" * 80)
    print("MINIMAL SILICON STUDY - MULTI SNAPSHOT")
    print("=" * 80)
    if args.randomize_seed:
        print(f"[SEED] Randomized runtime seed: {args.seed}")
    else:
        print(f"[SEED] Fixed runtime seed: {args.seed}")

    if args.data_path is None:
        raise ValueError(
            "--data-path must be provided unless it is recovered from --resume-from-checkpoint."
        )
    data_root = Path(args.data_path).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data-path does not exist: {data_root}")

    temperature_mode = any(
        [
            args.train_temps is not None,
            args.val_temp is not None,
            args.n_snapshots_per_temp is not None,
            args.val_n_snapshots is not None,
        ]
    )

    all_pairs: list[tuple[Path, Path]] | None = None
    train_pairs: list[tuple[Path, Path]]
    val_pairs: list[tuple[Path, Path]]
    train_temps_effective: list[int] | None = None
    split_mode: str

    if temperature_mode:
        if args.val_temp is None:
            raise ValueError(
                "Temperature split mode requires --val-temp to be specified."
            )
        if args.n_snapshots_per_temp is None:
            raise ValueError(
                "Temperature split mode requires --n-snapshots-per-temp to be specified."
            )
        if args.n_snapshots_per_temp <= 0:
            raise ValueError("--n-snapshots-per-temp must be > 0")
        if args.val_n_snapshots is not None and args.val_n_snapshots < 0:
            raise ValueError("--val-n-snapshots must be >= 0 when provided")

        requested_train_temps = parse_temperature_list(args.train_temps)
        if requested_train_temps is None:
            discovered_temps = discover_temperatures(data_root)
            train_temps_effective = [t for t in discovered_temps if t != args.val_temp]
        else:
            train_temps_effective = sorted(set(requested_train_temps))

        if len(train_temps_effective) == 0:
            raise ValueError("No training temperatures selected.")

        train_pairs = []
        train_pairs_by_temp: dict[int, list[tuple[Path, Path]]] = {}
        for temp in train_temps_effective:
            temp_pairs = discover_snapshot_pairs_for_temp(data_root, temp)
            if len(temp_pairs) == 0:
                raise ValueError(
                    f"No snapshot pairs found for training temperature {temp}K"
                )
            n_pick = min(len(temp_pairs), args.n_snapshots_per_temp)
            picked = random.sample(temp_pairs, n_pick)
            train_pairs_by_temp[temp] = picked
            train_pairs.extend(picked)
        random.Random(args.seed).shuffle(train_pairs)

        val_n = (
            args.val_n_snapshots
            if args.val_n_snapshots is not None
            else args.n_snapshots_per_temp
        )
        val_n = max(int(val_n), 0)
        if val_n == 0:
            val_pairs = []
        else:
            val_temp_pairs = discover_snapshot_pairs_for_temp(data_root, args.val_temp)
            if len(val_temp_pairs) == 0:
                raise ValueError(
                    f"No snapshot pairs found for validation temperature {args.val_temp}K"
                )
            if args.val_temp in train_pairs_by_temp:
                train_set_this_temp = set(train_pairs_by_temp[args.val_temp])
                val_temp_pairs = [
                    p for p in val_temp_pairs if p not in train_set_this_temp
                ]
                if len(val_temp_pairs) == 0:
                    raise ValueError(
                        "No disjoint validation snapshots available at "
                        f"{args.val_temp}K after removing training snapshots."
                    )
            val_pairs = random.sample(val_temp_pairs, min(len(val_temp_pairs), val_n))
        split_mode = "temperature"
    else:
        if args.num_train is None or args.num_val is None:
            raise ValueError(
                "Provide either temperature split args "
                "(--train-temps/--val-temp/--n-snapshots-per-temp) "
                "or global split args (--num-train and --num-val)."
            )
        if args.num_train <= 0:
            raise ValueError("--num-train must be > 0")
        if args.num_val < 0:
            raise ValueError("--num-val must be >= 0")
        all_pairs = discover_snapshot_pairs(data_root)
        random.Random(args.seed).shuffle(all_pairs)
        if len(all_pairs) < args.num_train + args.num_val:
            raise ValueError(
                f"Requested train+val={args.num_train + args.num_val} pairs but found only {len(all_pairs)} under {data_root}"
            )
        train_pairs = all_pairs[: args.num_train]
        val_pairs = all_pairs[args.num_train : args.num_train + args.num_val]
        split_mode = "global"

    if len(train_pairs) == 0:
        raise ValueError("Resolved training snapshot list is empty.")

    device = torch.device(args.device)
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else 0
        torch.cuda.set_device(cuda_index)
        warmup_device = torch.device(f"cuda:{cuda_index}")
        warmup = torch.zeros(
            (1, 1), device=warmup_device, dtype=torch_dtype, requires_grad=True
        )
        warmup_loss = (warmup @ warmup).sum()
        warmup_loss.backward()
        torch.cuda.synchronize()
    orbital_selection_obj = parse_orbital_selection(args.orbital_selection)

    is_sweep_run = bool(os.environ.get("WANDB_SWEEP_ID"))
    wandb_kwargs = {"project": args.wandb_project}
    requested_run_name = args.run_name
    effective_run_name = requested_run_name
    if resolved_wandb_entity is not None:
        wandb_kwargs["entity"] = resolved_wandb_entity
    elif args.wandb_entity is not None:
        wandb_kwargs["entity"] = args.wandb_entity
    if continue_wandb_run:
        wandb_kwargs["id"] = args.resume_from_run_id
        wandb_kwargs["resume"] = "must"
    elif not is_sweep_run:
        # For non-sweep runs, log full argparse namespace directly.
        wandb_kwargs["config"] = vars(args)
    if requested_run_name is not None and not continue_wandb_run:
        effective_run_name = build_timestamped_run_name(requested_run_name)
        wandb_kwargs["name"] = effective_run_name
    wandb.init(**wandb_kwargs)
    if continue_wandb_run and not is_sweep_run:
        wandb.config.update(vars(args), allow_val_change=True)

    for name in BOOLEAN_ARG_NAMES:
        cfg_val = _get_wandb_config_value(name)
        if cfg_val is not None:
            setattr(args, name, _coerce_wandb_bool(name, cfg_val))

    config = {
        "data_path": str(data_root),
        "training_unit": args.training_unit,
        "hamiltonian_scale_from_hartree": UNIT_SCALE_FROM_HARTREE[args.training_unit],
        "matrix_targets": matrix_targets,
        "enable_energy": args.enable_energy,
        "enable_num_electrons": args.enable_num_electrons,
        "train_on_energy": args.train_on_energy,
        "train_on_num_electrons": args.train_on_num_electrons,
        "loss_coef_observables": args.loss_coef_observables,
        "loss_coef_density_matrix": args.loss_coef_density_matrix,
        "enable_forces": args.enable_forces,
        "train_on_forces": args.train_on_forces,
        "loss_coef_forces": args.loss_coef_forces,
        "rescale_density_to_num_electrons": args.rescale_density_to_num_electrons,
        "symmetrize_preds": args.symmetrize_preds,
        "split_mode": split_mode,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "train_temps": args.train_temps,
        "val_temp": args.val_temp,
        "n_snapshots_per_temp": args.n_snapshots_per_temp,
        "val_n_snapshots": args.val_n_snapshots,
        "effective_train_temps": train_temps_effective,
        "convention": args.convention,
        "orbital_selection": args.orbital_selection,
        "hidden_dim": args.hidden_dim,
        "l_max": args.l_max,
        "hidden_irreps": args.hidden_irreps,
        "num_layers": args.num_layers,
        "cutoff_radius": args.cutoff_radius,
        "n_radial": args.n_radial,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "dtype": args.dtype,
        "grad_clip": args.grad_clip,
        "grad_accum_steps": args.grad_accum_steps,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "log_interval": args.log_interval,
        "adaptive_log_interval": args.adaptive_log_interval,
        "benchmark": args.benchmark,
        "e3layernorm": args.e3layernorm,
        "separate_shifted_self": args.separate_shifted_self,
        "edge_encoder_use_sh_tensor_square": args.edge_encoder_use_sh_tensor_square,
        "radial_embedding_scale": args.radial_embedding_scale,
        "head_mlp_for_scalars": args.head_mlp_for_scalars,
        "head_use_tensor_square": args.head_use_tensor_square,
        "head_use_node_embeddings_for_self_edges": args.head_use_node_embeddings_for_self_edges,
        "head_e3mlp_layers": args.head_e3mlp_layers,
        "apply_cutoff_to_targets": args.apply_cutoff_to_targets,
        "require_exact_edge_match": args.require_exact_edge_match,
        "log_data": args.log_data,
        "log_model": args.log_model,
        "log_forward": args.log_forward,
        "verbose_forward": args.verbose_forward,
        "log_per_irrep_metrics": args.log_per_irrep_metrics,
        "print_per_irrep_metrics": args.print_per_irrep_metrics,
        "log_per_irrep_images": args.log_per_irrep_images,
        "generate_video": args.generate_video,
        "video_max_atoms": args.video_max_atoms,
        "device": args.device,
        "checkpoint_dir": args.checkpoint_dir,
        "wandb_project": args.wandb_project,
        "wandb_entity": wandb_kwargs.get("entity"),
        "snapshot_cache_dir": args.snapshot_cache_dir,
        "run_name": effective_run_name,
        "run_name_prefix": requested_run_name,
        "seed": args.seed,
        "randomize_seed": args.randomize_seed,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resume_from_run_id": args.resume_from_run_id,
        "fresh_run": args.fresh_run,
        # Logger-compat keys expected by detailed_logging.log_config from minimal_overfit_study.
        "partial_train": None,
        "box_convention": "rows",
        "log_activations_wandb": False,
    }

    run_name = wandb.run.name
    run_checkpoint_dir = Path(args.checkpoint_dir) / run_name
    print(f"[CHECKPOINTS] Run checkpoint directory: {run_checkpoint_dir}")
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frame_output_dir = run_checkpoint_dir / "frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "training unit: "
        f"{UNIT_DISPLAY_NAME[args.training_unit]} "
        f"(x{config['hamiltonian_scale_from_hartree']:.7g} from Hartree)"
    )
    log_config(config, run_checkpoint_dir, frame_output_dir)

    # Build datasets via DatasetFactory (requested).
    # Use very large cutoff when targets should stay unfiltered.
    cfg_ds = NetConfig()
    cfg_ds.dtype = torch_dtype
    cfg_ds.device = "cpu"
    cfg_ds.verbosity = 0
    cfg_ds.snapshot_cache_dir = args.snapshot_cache_dir
    cfg_ds.precompute_edge_features = False
    cfg_ds.cutoff_radius = args.cutoff_radius if args.apply_cutoff_to_targets else None
    cfg_ds.enable_forces = False
    cfg_ds.train_on_forces = False

    fac = DatasetFactory(cfg_ds, convention=args.convention)
    for m, i in train_pairs:
        fac.add_snapshot(m, i, purpose="train")
    for m, i in val_pairs:
        fac.add_snapshot(m, i, purpose="val")
    train_ds, val_ds, _mapper_full = fac.create()

    if len(train_ds) == 0:
        raise ValueError(
            "DatasetFactory created an empty train dataset. "
            "Check file integrity (Si_DM/info.dat) and convention/orbital settings."
        )

    if args.log_data:
        if all_pairs is not None:
            print(f"[DATA] discovered pairs: {len(all_pairs)}")
        if split_mode == "temperature":
            print(
                f"[DATA] temperature split: train_temps={train_temps_effective}, val_temp={args.val_temp}"
            )
            print(
                f"[DATA] n_snapshots_per_temp={args.n_snapshots_per_temp}, "
                f"val_n_snapshots={args.val_n_snapshots if args.val_n_snapshots is not None else args.n_snapshots_per_temp}"
            )
        print(f"[DATA] using train={len(train_pairs)}, val={len(val_pairs)}")
        print(f"[DATA] first train pair: {train_pairs[0] if train_pairs else 'N/A'}")

    # Build final mapper from first sample after optional orbital reduction.
    x0, y0 = train_ds[0]
    mapper_cpu = prepare_mapper_from_sample(
        x0, y0, orbital_selection_obj, torch.device("cpu"), torch_dtype
    )
    mapper = BlockIrrepMapper(
        mapper_cpu.orbital_cfg,
        device=device,
        dtype=torch_dtype,
    )

    # Graph feature config used per sample.
    sh_irreps = Irreps.spherical_harmonics(args.l_max)

    if args.log_data:
        print("[DATA] preprocessing train samples...")
    train_samples = preprocess_dataset_samples(
        train_ds,
        mapper=mapper_cpu,
        sh_irreps=sh_irreps,
        n_radial=args.n_radial,
        radial_embedding_scale=args.radial_embedding_scale,
        training_unit=args.training_unit,
        hamiltonian_scale_from_hartree=config["hamiltonian_scale_from_hartree"],
        cutoff_radius=args.cutoff_radius,
        apply_cutoff_to_targets=args.apply_cutoff_to_targets,
        orbital_selection=orbital_selection_obj,
        matrix_targets=matrix_targets,
        enable_forces=args.enable_forces,
        require_exact_edge_match=args.require_exact_edge_match,
        device=torch.device("cpu"),
        torch_dtype=torch_dtype,
        log_first_sample=args.log_data,
        log_model_first_sample=args.log_model,
        progress_desc="Preprocessing train samples",
    )
    for idx, sample in enumerate(train_samples):
        if device.type == "cuda":
            sample = pin_sample_memory(sample)
            sample = move_small_sample_tensors_to_device(sample, device)
        train_samples[idx] = sample

    if args.log_data:
        print("[DATA] preprocessing val samples...")
    val_iterable = val_ds if val_ds is not None else []
    val_samples = preprocess_dataset_samples(
        val_iterable,
        mapper=mapper_cpu,
        sh_irreps=sh_irreps,
        n_radial=args.n_radial,
        radial_embedding_scale=args.radial_embedding_scale,
        training_unit=args.training_unit,
        hamiltonian_scale_from_hartree=config["hamiltonian_scale_from_hartree"],
        cutoff_radius=args.cutoff_radius,
        apply_cutoff_to_targets=args.apply_cutoff_to_targets,
        orbital_selection=orbital_selection_obj,
        matrix_targets=matrix_targets,
        enable_forces=args.enable_forces,
        require_exact_edge_match=args.require_exact_edge_match,
        device=torch.device("cpu"),
        torch_dtype=torch_dtype,
        log_first_sample=False,
        log_model_first_sample=False,
        progress_desc="Preprocessing val samples",
    )
    for idx, sample in enumerate(val_samples):
        if device.type == "cuda":
            sample = pin_sample_memory(sample)
            sample = move_small_sample_tensors_to_device(sample, device)
        val_samples[idx] = sample

    if args.hidden_irreps is not None:
        hidden_irreps = Irreps(args.hidden_irreps)
    else:
        hidden_irreps = build_hidden_irreps(
            l_max=args.l_max, base_dim=args.hidden_dim, use_odd_features=True
        )
        config["hidden_irreps"] = str(hidden_irreps)

    num_elements = len(mapper.orbital_cfg.elements())
    num_edge_types = num_elements**2

    with suppress_stdout(not args.log_model):
        network = MinimalNetwork(
            num_elements=num_elements,
            n_radial=args.n_radial,
            num_edge_types=num_edge_types,
            hidden_irreps=hidden_irreps,
            sh_irreps=sh_irreps,
            num_layers=args.num_layers,
            mapper=mapper,
            matrix_targets=matrix_targets,
            head_e3mlp_layers=args.head_e3mlp_layers,
            edge_encoder_use_sh_tensor_square=args.edge_encoder_use_sh_tensor_square,
            magnitude_factorization=False,
            head_mlp_for_scalars=args.head_mlp_for_scalars,
            head_use_tensor_square=args.head_use_tensor_square,
            head_use_node_embeddings_for_self_edges=args.head_use_node_embeddings_for_self_edges,
            separate_shifted_self=args.separate_shifted_self,
            use_e3layernorm=args.e3layernorm,
        ).to(device=device, dtype=torch_dtype)
    if args.log_model:
        print("[OK] Network architecture complete.")

    all_irreps = get_all_irreps_in_hamiltonian(mapper)
    optimizer = Adam(network.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=10,
        min_lr=1e-6,
        verbose=True,
    )

    if resume_payload is not None:
        network.load_state_dict(resume_payload["model_state_dict"])
        if "optimizer_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
        if "lr" in explicit_overrides:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        if (
            not reset_scheduler_state
            and "scheduler_state_dict" in resume_payload
            and resume_payload["scheduler_state_dict"] is not None
        ):
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])

    print("")
    if resume_payload is not None:
        print("=" * 80)
        print("RESUME")
        print("=" * 80)
        print(f"checkpoint: {resume_checkpoint}")
        print(f"checkpoint epoch: {int(resume_payload.get('epoch', -1)) + 1}")
        print(f"resume source: {resume_source_kind}")
        if resume_changes:
            print("explicit config overrides:")
            for name in sorted(resume_changes):
                old_value, new_value = resume_changes[name]
                print(f"  {name}: {old_value!r} -> {new_value!r}")
        else:
            print("explicit config overrides: none")
        print(
            "scheduler state: "
            + ("reset from current args" if reset_scheduler_state else "restored")
        )
        print("best/history state: reset for new run")

    print("")
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"train samples: {len(train_samples)} | val samples: {len(val_samples)}")
    if resume_payload is not None and train_end_epoch <= start_epoch:
        print(
            "resume requested, but additional --num-epochs is <= 0; "
            "skipping training loop."
        )
    has_validation = len(val_samples) > 0

    matrix_alias = {"hamiltonian": "H", "overlap": "S", "density": "D"}
    irrep_prefix_by_matrix = {"hamiltonian": "H_", "overlap": "S_", "density": "D_"}
    matrix_frame_output_dirs = {}
    for matrix_name in matrix_targets:
        matrix_dir = frame_output_dir / matrix_name
        matrix_dir.mkdir(parents=True, exist_ok=True)
        matrix_frame_output_dirs[matrix_name] = matrix_dir

    history = {"train_loss": [], "val_loss": [], "val_mae_H": []}
    best_score = float("inf")
    best_epoch = -1
    best_model_path = run_checkpoint_dir / "best_model.pt"
    latest_checkpoint_path = run_checkpoint_dir / "latest_checkpoint.pt"
    last_log_time = time.time()
    last_logged_epoch = -1
    total_run_epochs = max(train_end_epoch - start_epoch, 0)
    last_completed_epoch = start_epoch - 1
    use_global_display_epoch = continue_wandb_run
    display_total_epochs = (
        train_end_epoch if use_global_display_epoch else total_run_epochs
    )

    benchmark_order = [
        "epoch_total",
        "train_forward",
        "train_loss",
        "backward_total",
        "optimizer_step",
        "eval_total",
        "wandb_log",
    ]
    benchmark_accum = {k: 0.0 for k in benchmark_order}
    benchmark_epochs = 0

    def benchmark_add(name: str, dt: float) -> None:
        if args.benchmark and name in benchmark_accum:
            benchmark_accum[name] += max(float(dt), 0.0)

    def benchmark_report(epoch_zero_based: int) -> None:
        if not args.benchmark or benchmark_epochs <= 0:
            return
        epoch_ms = benchmark_accum["epoch_total"] * 1000.0 / benchmark_epochs
        print(
            f"\n[BENCHMARK EPOCH {epoch_zero_based + 1}/{max(display_total_epochs, 1)}]"
        )
        print(f"  averaged over {benchmark_epochs} epoch(s): {epoch_ms:.3f} ms/epoch")
        for k in benchmark_order:
            ms = benchmark_accum[k] * 1000.0 / benchmark_epochs
            pct = 100.0 * ms / max(epoch_ms, 1e-12)
            print(f"  {k:24s} {ms:10.3f} ms  ({pct:6.2f}%)")

    if resume_payload is not None:
        print("")
        print("=" * 80)
        print("INITIAL EVALUATION")
        print("=" * 80)
        initial_eval = evaluate_split(
            network=network,
            samples=val_samples if len(val_samples) > 0 else train_samples,
            mapper=mapper,
            mapper_cpu=mapper_cpu,
            all_irreps=all_irreps,
            device=device,
            matrix_targets=matrix_targets,
            sh_irreps=sh_irreps,
            cutoff_radius=args.cutoff_radius,
            n_radial=args.n_radial,
            radial_embedding_scale=args.radial_embedding_scale,
            symmetrize_preds=args.symmetrize_preds,
            rescale_density_to_num_electrons=args.rescale_density_to_num_electrons,
            enable_energy=args.enable_energy,
            train_on_energy=args.train_on_energy,
            enable_num_electrons=args.enable_num_electrons,
            train_on_num_electrons=args.train_on_num_electrons,
            loss_coef_density_matrix=args.loss_coef_density_matrix,
            loss_coef_observables=args.loss_coef_observables,
            enable_forces=args.enable_forces,
            train_on_forces=args.train_on_forces,
            loss_coef_forces=args.loss_coef_forces,
        )
        initial_detailed_metrics = initial_eval["detailed"]
        log_detailed_training_metrics(
            avg_epoch_time=0.0,
            epochs_since_last_log=0,
            time_elapsed=0.0,
            loss_value=float(initial_eval["loss"]),
            detailed_metrics=initial_detailed_metrics,
            irrep_losses=None,
        )
        initial_wandb_log: dict[str, float] = {"epoch": 0}
        initial_wandb_log.update(
            build_wandb_detailed_metrics_log(
                epoch_zero_based=0,
                loss_value=float(initial_eval["loss"]),
                detailed_metrics=initial_detailed_metrics,
            )
        )
        basic_by_name = initial_eval["basic_by_name"]
        if "hamiltonian" in basic_by_name:
            initial_wandb_log["initial/mae_H"] = basic_by_name["hamiltonian"]["mae"]
            initial_wandb_log["initial/mse_H"] = basic_by_name["hamiltonian"]["mse"]
        if "overlap" in basic_by_name:
            initial_wandb_log["initial/mae_S"] = basic_by_name["overlap"]["mae"]
            initial_wandb_log["initial/mse_S"] = basic_by_name["overlap"]["mse"]
        if "density" in basic_by_name:
            initial_wandb_log["initial/mae_D"] = basic_by_name["density"]["mae"]
            initial_wandb_log["initial/mse_D"] = basic_by_name["density"]["mse"]
        initial_wandb_log["initial/loss_total"] = float(initial_eval["loss"])
        if (
            initial_eval["forces_mae"] is not None
            and initial_eval["forces_mse"] is not None
        ):
            initial_wandb_log["initial/mae_F"] = initial_eval["forces_mae"]
            initial_wandb_log["initial/mse_F"] = initial_eval["forces_mse"]
        if initial_eval["energy_mae"] is not None:
            initial_wandb_log["initial/energy_mae"] = initial_eval["energy_mae"]
        if initial_eval["num_electrons_mae"] is not None:
            initial_wandb_log["initial/num_electrons_mae"] = initial_eval[
                "num_electrons_mae"
            ]
        if initial_eval["num_electrons_mae_pre_correction"] is not None:
            initial_wandb_log["initial/num_electrons_mae_pre_correction"] = (
                initial_eval["num_electrons_mae_pre_correction"]
            )
        for matrix_name in matrix_targets:
            per_irrep = initial_eval["per_irrep_by_name"].get(matrix_name, {})
            if per_irrep:
                per_irrep_log = build_wandb_per_irrep_metrics_log(
                    epoch_zero_based=0,
                    all_irreps=all_irreps,
                    per_irrep_metrics=per_irrep,
                    metric_prefix=irrep_prefix_by_matrix.get(matrix_name, ""),
                )
                initial_wandb_log.update(
                    {
                        (
                            key
                            if key == "epoch"
                            else f"initial/{matrix_alias.get(matrix_name, matrix_name)}_{key}"
                        ): value
                        for key, value in per_irrep_log.items()
                    }
                )
        wandb.log(initial_wandb_log)

    try:
        for epoch in range(start_epoch, train_end_epoch):
            local_epoch = epoch - start_epoch
            wandb_epoch = epoch if continue_wandb_run else local_epoch
            display_epoch = epoch if use_global_display_epoch else local_epoch
            epoch_t0 = time.perf_counter() if args.benchmark else 0.0
            network.train()
            should_log_now = should_log_epoch(
                display_epoch, args.log_interval, args.adaptive_log_interval
            )
            train_loss_total = 0.0
            epoch_wandb_log: dict[str, float] = {"epoch": wandb_epoch}
            last_grad_norm = None
            num_train_samples = len(train_samples)
            group_start_idx = 0
            optimizer.zero_grad(set_to_none=True)

            for sample_idx, sample_cpu in enumerate(train_samples):
                sample = move_sample_to_device(sample_cpu, device)
                t_fwd = time.perf_counter() if args.benchmark else 0.0
                positions_train = None
                if args.enable_forces:
                    positions_train = (
                        sample["positions"].detach().clone().requires_grad_(True)
                    )
                pred = predict_sample(
                    network,
                    sample,
                    mapper,
                    matrix_targets=matrix_targets,
                    sh_irreps=sh_irreps,
                    cutoff_radius=args.cutoff_radius,
                    n_radial=args.n_radial,
                    radial_embedding_scale=args.radial_embedding_scale,
                    symmetrize_preds=args.symmetrize_preds,
                    rescale_density_to_num_electrons=args.rescale_density_to_num_electrons,
                    verbose_forward=args.verbose_forward,
                    log_forward=args.log_forward and should_log_now,
                    positions_override=positions_train,
                )
                benchmark_add(
                    "train_forward",
                    time.perf_counter() - t_fwd if args.benchmark else 0.0,
                )

                t_loss = time.perf_counter() if args.benchmark else 0.0
                loss_info = compute_loss_for_sample(
                    pred["pred_irreps_by_name"],
                    pred["pred_matrix_norm_by_name"],
                    sample,
                    matrix_targets=matrix_targets,
                    loss_coef_density_matrix=args.loss_coef_density_matrix,
                    enable_energy=args.enable_energy,
                    train_on_energy=args.train_on_energy,
                    enable_num_electrons=args.enable_num_electrons,
                    train_on_num_electrons=args.train_on_num_electrons,
                    loss_coef_observables=args.loss_coef_observables,
                    enable_forces=args.enable_forces,
                    train_on_forces=args.train_on_forces,
                    loss_coef_forces=args.loss_coef_forces,
                    positions_for_forces=positions_train,
                    force_loss_requires_grad=args.train_on_forces,
                )
                loss = loss_info["loss_total"]
                benchmark_add(
                    "train_loss",
                    time.perf_counter() - t_loss if args.benchmark else 0.0,
                )

                current_group_size = min(
                    args.grad_accum_steps, num_train_samples - group_start_idx
                )
                loss_to_backward = loss / float(current_group_size)
                t_bwd = time.perf_counter() if args.benchmark else 0.0
                loss_to_backward.backward()
                benchmark_add(
                    "backward_total",
                    time.perf_counter() - t_bwd if args.benchmark else 0.0,
                )

                should_step = (sample_idx - group_start_idx + 1) >= current_group_size
                if should_step:
                    if args.grad_clip > 0:
                        grad_norm = clip_grad_norm_(
                            network.parameters(), args.grad_clip
                        )
                        last_grad_norm = float(grad_norm.item())

                    t_opt = time.perf_counter() if args.benchmark else 0.0
                    optimizer.step()
                    benchmark_add(
                        "optimizer_step",
                        time.perf_counter() - t_opt if args.benchmark else 0.0,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    group_start_idx = sample_idx + 1
                    last_completed_epoch = epoch

                train_loss_total += float(loss.item())
                for matrix_name, matrix_loss in loss_info[
                    "matrix_block_losses"
                ].items():
                    key = f"train/loss_block_{matrix_name}"
                    epoch_wandb_log[key] = epoch_wandb_log.get(key, 0.0) + float(
                        matrix_loss.item()
                    )
                epoch_wandb_log["train/loss_block_total"] = epoch_wandb_log.get(
                    "train/loss_block_total", 0.0
                ) + float(loss_info["loss_block_total"].item())
                epoch_wandb_log["train/loss_total"] = epoch_wandb_log.get(
                    "train/loss_total", 0.0
                ) + float(loss.item())
                epoch_wandb_log["train/loss_step"] = float(loss.item())
                epoch_wandb_log["lr"] = optimizer.param_groups[0]["lr"]

                if args.train_on_energy:
                    epoch_wandb_log["train/loss_energy_weighted"] = epoch_wandb_log.get(
                        "train/loss_energy_weighted", 0.0
                    ) + float(loss_info["loss_energy_weighted"].item())
                if args.train_on_num_electrons:
                    epoch_wandb_log["train/loss_num_electrons_weighted"] = (
                        epoch_wandb_log.get("train/loss_num_electrons_weighted", 0.0)
                        + float(loss_info["loss_num_electrons_weighted"].item())
                    )
                if args.train_on_forces:
                    epoch_wandb_log["train/loss_forces_weighted"] = epoch_wandb_log.get(
                        "train/loss_forces_weighted", 0.0
                    ) + float(loss_info["loss_forces_weighted"].item())
                if loss_info["energy_mae"] is not None:
                    epoch_wandb_log["train/energy_mae"] = epoch_wandb_log.get(
                        "train/energy_mae", 0.0
                    ) + float(loss_info["energy_mae"].item())
                if loss_info["num_electrons_mae"] is not None:
                    epoch_wandb_log["train/num_electrons_mae"] = epoch_wandb_log.get(
                        "train/num_electrons_mae", 0.0
                    ) + float(loss_info["num_electrons_mae"].item())
                if pred["num_electrons_mae_pre_correction"] is not None:
                    epoch_wandb_log["train/num_electrons_mae_pre_correction"] = (
                        epoch_wandb_log.get(
                            "train/num_electrons_mae_pre_correction", 0.0
                        )
                        + float(pred["num_electrons_mae_pre_correction"].item())
                    )
                if (
                    loss_info["forces_mae"] is not None
                    and loss_info["forces_mse"] is not None
                ):
                    epoch_wandb_log["train/forces_mae"] = epoch_wandb_log.get(
                        "train/forces_mae", 0.0
                    ) + float(loss_info["forces_mae"].item())
                    epoch_wandb_log["train/forces_mse"] = epoch_wandb_log.get(
                        "train/forces_mse", 0.0
                    ) + float(loss_info["forces_mse"].item())

            denom = max(len(train_samples), 1)
            train_loss = train_loss_total / denom
            for key, value in list(epoch_wandb_log.items()):
                if key.startswith("train/") or key.startswith("partial/"):
                    epoch_wandb_log[key] = value / denom
            if last_grad_norm is not None:
                epoch_wandb_log["grad_norm"] = last_grad_norm

            do_log = should_log_now
            val_eval = {
                "loss": train_loss,
                "detailed": {},
                "basic_by_name": {},
                "per_irrep_by_name": {},
                "forces_mae": None,
                "forces_mse": None,
                "energy_mae": None,
                "num_electrons_mae": None,
                "num_electrons_mae_pre_correction": None,
                "first_pred_metrics_by_name": None,
                "first_sample": None,
            }
            if do_log:
                t_eval = time.perf_counter() if args.benchmark else 0.0
                val_eval = evaluate_split(
                    network=network,
                    samples=val_samples if len(val_samples) > 0 else train_samples,
                    mapper=mapper,
                    mapper_cpu=mapper_cpu,
                    all_irreps=all_irreps,
                    device=device,
                    matrix_targets=matrix_targets,
                    sh_irreps=sh_irreps,
                    cutoff_radius=args.cutoff_radius,
                    n_radial=args.n_radial,
                    radial_embedding_scale=args.radial_embedding_scale,
                    symmetrize_preds=args.symmetrize_preds,
                    rescale_density_to_num_electrons=args.rescale_density_to_num_electrons,
                    enable_energy=args.enable_energy,
                    train_on_energy=args.train_on_energy,
                    enable_num_electrons=args.enable_num_electrons,
                    train_on_num_electrons=args.train_on_num_electrons,
                    loss_coef_density_matrix=args.loss_coef_density_matrix,
                    loss_coef_observables=args.loss_coef_observables,
                    enable_forces=args.enable_forces,
                    train_on_forces=args.train_on_forces,
                    loss_coef_forces=args.loss_coef_forces,
                )
                benchmark_add(
                    "eval_total",
                    time.perf_counter() - t_eval if args.benchmark else 0.0,
                )

            if has_validation:
                if do_log:
                    scheduler.step(float(val_eval["loss"]))
            else:
                scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(
                float(val_eval["loss"]) if do_log else float("nan")
            )
            history["val_mae_H"].append(
                float(val_eval["detailed"].get("mae", float("nan")))
                if do_log
                else float("nan")
            )

            score = float(val_eval["loss"]) if has_validation else train_loss
            should_update_best = do_log if has_validation else True
            if should_update_best and score < best_score:
                best_score = score
                best_epoch = display_epoch
                _save_training_checkpoint(
                    best_model_path,
                    epoch=epoch,
                    network=network,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    score=score,
                    config=config,
                    history=history,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    train_pairs=train_pairs,
                    val_pairs=val_pairs,
                    metrics=val_eval["detailed"],
                )
                print(
                    f"[OK] Best model saved to {best_model_path.name} (score: {score:.6e})"
                )

            latest_score = float(val_eval["loss"]) if do_log else train_loss

            if do_log:
                current_time = time.time()
                time_elapsed = current_time - last_log_time
                epochs_since_last_log = (
                    (display_epoch - last_logged_epoch)
                    if last_logged_epoch >= 0
                    else (display_epoch + 1)
                )
                avg_epoch_time = time_elapsed / max(epochs_since_last_log, 1)
                last_log_time = current_time
                last_logged_epoch = display_epoch

                print(f"\n{'=' * 60}")
                print(
                    f"EPOCH {display_epoch + 1}/{max(display_total_epochs, 1)}  |  lr={current_lr:.6e}"
                )
                print(f"{'=' * 60}")
                log_detailed_training_metrics(
                    avg_epoch_time=avg_epoch_time,
                    epochs_since_last_log=epochs_since_last_log,
                    time_elapsed=time_elapsed,
                    loss_value=train_loss,
                    detailed_metrics=val_eval["detailed"],
                    irrep_losses=None,
                )
                if args.print_per_irrep_metrics:
                    for matrix_name in matrix_targets:
                        per_irrep = val_eval["per_irrep_by_name"].get(matrix_name, {})
                        if per_irrep:
                            log_per_irrep_metrics(
                                f"Validation Per-Irrep Metrics ({matrix_name}):",
                                all_irreps,
                                per_irrep,
                                metric_prefix=irrep_prefix_by_matrix.get(
                                    matrix_name, ""
                                ),
                            )

                if (
                    val_eval["first_pred_metrics_by_name"] is not None
                    and val_eval["first_sample"] is not None
                ):
                    fs = val_eval["first_sample"]
                    for matrix_name in matrix_targets:
                        try:
                            save_hamiltonian_frame_to_disk(
                                val_eval["first_pred_metrics_by_name"][matrix_name],
                                fs["target_matrices"][matrix_name],
                                (
                                    fs["target_overlap"]
                                    if matrix_name == "hamiltonian"
                                    else None
                                ),
                                fs["atoms_list"],
                                mapper.orbital_cfg,
                                matrix_frame_output_dirs[matrix_name],
                                display_epoch,
                                sx=0,
                                sy=0,
                                sz=0,
                                dynamic_range=False,
                                diff_dynamic_range=True,
                                percentile=99.0,
                                matrix_label=matrix_alias.get(matrix_name, matrix_name),
                                max_atoms=(
                                    args.video_max_atoms
                                    if args.video_max_atoms is not None
                                    and args.video_max_atoms > 0
                                    else None
                                ),
                            )
                        except Exception as exc:
                            print(
                                f"[WARN] Could not save {matrix_name} frame for epoch {epoch}: {exc}"
                            )

                epoch_wandb_log.update(
                    build_wandb_detailed_metrics_log(
                        epoch_zero_based=wandb_epoch,
                        loss_value=train_loss,
                        detailed_metrics=val_eval["detailed"],
                    )
                )
                basic_by_name = val_eval["basic_by_name"]
                if "hamiltonian" in basic_by_name:
                    epoch_wandb_log["mae_H"] = basic_by_name["hamiltonian"]["mae"]
                    epoch_wandb_log["mse_H"] = basic_by_name["hamiltonian"]["mse"]
                if "overlap" in basic_by_name:
                    epoch_wandb_log["mae_S"] = basic_by_name["overlap"]["mae"]
                    epoch_wandb_log["mse_S"] = basic_by_name["overlap"]["mse"]
                if "density" in basic_by_name:
                    epoch_wandb_log["mae_D"] = basic_by_name["density"]["mae"]
                    epoch_wandb_log["mse_D"] = basic_by_name["density"]["mse"]
                if (
                    val_eval["forces_mae"] is not None
                    and val_eval["forces_mse"] is not None
                ):
                    epoch_wandb_log["mae_F"] = val_eval["forces_mae"]
                    epoch_wandb_log["mse_F"] = val_eval["forces_mse"]
                if val_eval["energy_mae"] is not None:
                    epoch_wandb_log["val/energy_mae"] = val_eval["energy_mae"]
                if val_eval["num_electrons_mae"] is not None:
                    epoch_wandb_log["val/num_electrons_mae"] = val_eval[
                        "num_electrons_mae"
                    ]
                if val_eval["num_electrons_mae_pre_correction"] is not None:
                    epoch_wandb_log["val/num_electrons_mae_pre_correction"] = val_eval[
                        "num_electrons_mae_pre_correction"
                    ]
                for matrix_name in matrix_targets:
                    per_irrep = val_eval["per_irrep_by_name"].get(matrix_name, {})
                    if per_irrep:
                        epoch_wandb_log.update(
                            build_wandb_per_irrep_metrics_log(
                                epoch_zero_based=wandb_epoch,
                                all_irreps=all_irreps,
                                per_irrep_metrics=per_irrep,
                                metric_prefix=irrep_prefix_by_matrix.get(
                                    matrix_name, ""
                                ),
                            )
                        )
                epoch_wandb_log["val/loss"] = float(val_eval["loss"])
                epoch_wandb_log["lr"] = current_lr

                t_wandb = time.perf_counter() if args.benchmark else 0.0
                wandb.log(epoch_wandb_log)
                benchmark_add(
                    "wandb_log",
                    time.perf_counter() - t_wandb if args.benchmark else 0.0,
                )
                _save_training_checkpoint(
                    latest_checkpoint_path,
                    epoch=epoch,
                    network=network,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    score=latest_score,
                    config=config,
                    history=history,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    train_pairs=train_pairs,
                    val_pairs=val_pairs,
                    metrics=val_eval["detailed"],
                    wandb_run_id=wandb.run.id if wandb.run is not None else None,
                    wandb_project=args.wandb_project,
                    wandb_entity=wandb_kwargs.get("entity"),
                    wandb_run_name=wandb.run.name if wandb.run is not None else None,
                )
                _update_wandb_checkpoint_summary(
                    latest_checkpoint_path=latest_checkpoint_path
                )

                if args.benchmark:
                    benchmark_add("epoch_total", time.perf_counter() - epoch_t0)
                    benchmark_epochs += 1
                    if do_log:
                        benchmark_report(display_epoch)
                        benchmark_accum = {k: 0.0 for k in benchmark_order}
                        benchmark_epochs = 0

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user; running final evaluation.")

    print("")
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    final_eval = evaluate_split(
        network=network,
        samples=val_samples if len(val_samples) > 0 else train_samples,
        mapper=mapper,
        mapper_cpu=mapper_cpu,
        all_irreps=all_irreps,
        device=device,
        matrix_targets=matrix_targets,
        sh_irreps=sh_irreps,
        cutoff_radius=args.cutoff_radius,
        n_radial=args.n_radial,
        radial_embedding_scale=args.radial_embedding_scale,
        symmetrize_preds=args.symmetrize_preds,
        rescale_density_to_num_electrons=args.rescale_density_to_num_electrons,
        enable_energy=args.enable_energy,
        train_on_energy=args.train_on_energy,
        enable_num_electrons=args.enable_num_electrons,
        train_on_num_electrons=args.train_on_num_electrons,
        loss_coef_density_matrix=args.loss_coef_density_matrix,
        loss_coef_observables=args.loss_coef_observables,
        enable_forces=args.enable_forces,
        train_on_forces=args.train_on_forces,
        loss_coef_forces=args.loss_coef_forces,
    )

    final_detailed_metrics = final_eval["detailed"] or {
        "mae": 0.0,
        "mse": 0.0,
        "mae_mod": 0.0,
        "mse_mod": 0.0,
        "mu_H": 0.0,
        "correction_mae": 0.0,
        "correction_mse": 0.0,
    }
    log_final_metrics(final_detailed_metrics)

    final_metrics = {
        "final/mae_H": final_detailed_metrics["mae"],
        "final/mse_H": final_detailed_metrics["mse"],
        "final/mae_H_mod": final_detailed_metrics["mae_mod"],
        "final/mse_H_mod": final_detailed_metrics["mse_mod"],
        "final/mu_H": final_detailed_metrics["mu_H"],
        "final/correction_mae": final_detailed_metrics["correction_mae"],
        "final/correction_mse": final_detailed_metrics["correction_mse"],
    }
    basic_by_name = final_eval["basic_by_name"]
    if "overlap" in basic_by_name:
        final_metrics["final/mae_S"] = basic_by_name["overlap"]["mae"]
        final_metrics["final/mse_S"] = basic_by_name["overlap"]["mse"]
    if "density" in basic_by_name:
        final_metrics["final/mae_D"] = basic_by_name["density"]["mae"]
        final_metrics["final/mse_D"] = basic_by_name["density"]["mse"]
    if final_eval["energy_mae"] is not None:
        final_metrics["final/energy_mae"] = final_eval["energy_mae"]
    if final_eval["num_electrons_mae"] is not None:
        final_metrics["final/num_electrons_mae"] = final_eval["num_electrons_mae"]
    if final_eval["num_electrons_mae_pre_correction"] is not None:
        final_metrics["final/num_electrons_mae_pre_correction"] = final_eval[
            "num_electrons_mae_pre_correction"
        ]
    if final_eval["forces_mae"] is not None and final_eval["forces_mse"] is not None:
        final_metrics["final/mae_F"] = final_eval["forces_mae"]
        final_metrics["final/mse_F"] = final_eval["forces_mse"]

    fs = final_eval["first_sample"]
    pred_metrics_by_name = final_eval["first_pred_metrics_by_name"]
    if fs is not None and pred_metrics_by_name is not None:
        if "hamiltonian" in pred_metrics_by_name:
            dos_plot_path = run_checkpoint_dir / "dos_comparison_final.png"
            try:
                dos_metrics = save_dos_comparison_plot(
                    H_pred=pred_metrics_by_name["hamiltonian"],
                    H_gt=fs["target_matrices"]["hamiltonian"],
                    S=fs["target_overlap"],
                    output_path=dos_plot_path,
                    sigma=0.2,
                    bin_width=0.1,
                    title="DOS Comparison (Representative)",
                )
                for k, v in dos_metrics.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        final_metrics[f"final/{k}"] = float(v)
                wandb.log(
                    {"final/dos_comparison_plot": wandb.Image(str(dos_plot_path))}
                )
            except Exception as exc:
                print(f"[WARN] Could not generate DOS comparison plot: {exc}")

        for matrix_name in matrix_targets:
            dist_curve = compute_distance_error_curve(
                H_pred=pred_metrics_by_name[matrix_name],
                H_gt=fs["target_matrices"][matrix_name],
                positions=fs["positions"],
                box=fs["box"],
                n_bins=16,
            )
            if dist_curve is not None:
                curve_json_path = (
                    run_checkpoint_dir / f"distance_error_curve_{matrix_name}.json"
                )
                curve_plot_path = (
                    run_checkpoint_dir / f"distance_error_curve_{matrix_name}.png"
                )
                with open(curve_json_path, "w", encoding="utf-8") as f:
                    json.dump(dist_curve, f, indent=2)
                save_distance_error_curve_plot(
                    dist_curve,
                    curve_plot_path,
                    title=f"Distance Error Curves ({matrix_name}, Final)",
                )
                wandb.log(
                    {
                        f"distance_curve/{matrix_alias.get(matrix_name, matrix_name)}_plot": wandb.Image(
                            str(curve_plot_path)
                        )
                    }
                )

        if args.log_per_irrep_images:
            irrep_output_dir = run_checkpoint_dir / "per_irrep_images"
            irrep_output_dir.mkdir(parents=True, exist_ok=True)
            for matrix_name in matrix_targets:
                for irrep in all_irreps:
                    ir_str = str(irrep)
                    try:
                        pred_ir = split_hamiltonian_by_irrep(
                            pred_metrics_by_name[matrix_name], mapper_cpu, ir_str
                        )
                        targ_ir = split_hamiltonian_by_irrep(
                            fs["target_matrices"][matrix_name], mapper_cpu, ir_str
                        )
                        filename_prefix = f"{matrix_name}_{ir_str}"
                        visualize_hamiltonians(
                            pred_ir,
                            targ_ir,
                            (
                                fs["target_overlap"]
                                if matrix_name == "hamiltonian"
                                else None
                            ),
                            fs["atoms_list"],
                            mapper.orbital_cfg,
                            k_range=0,
                            output_dir=irrep_output_dir,
                            dynamic_range=True,
                            diff_dynamic_range=True,
                            per_panel_dynamic_range=True,
                            filename_prefix=filename_prefix,
                            percentile=99.0,
                        )
                        image_path = (
                            irrep_output_dir / f"{filename_prefix}_sx+0_sy+0_sz+0.png"
                        )
                        if image_path.exists():
                            wandb.log(
                                {
                                    f"irrep_images/{matrix_alias.get(matrix_name, matrix_name)}/{ir_str}": wandb.Image(
                                        str(image_path)
                                    )
                                }
                            )
                    except Exception as exc:
                        print(
                            f"[WARN] Irrep image failed for {matrix_name}/{ir_str}: {exc}"
                        )

    wandb.log(final_metrics)

    final_model_path = run_checkpoint_dir / "final_model.pt"
    _save_training_checkpoint(
        latest_checkpoint_path,
        epoch=last_completed_epoch,
        network=network,
        optimizer=optimizer,
        scheduler=scheduler,
        score=history["train_loss"][-1] if history["train_loss"] else float("nan"),
        config=config,
        history=history,
        best_score=best_score,
        best_epoch=best_epoch,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        metrics=final_metrics,
        wandb_run_id=wandb.run.id if wandb.run is not None else None,
        wandb_project=args.wandb_project,
        wandb_entity=wandb_kwargs.get("entity"),
        wandb_run_name=wandb.run.name if wandb.run is not None else None,
    )
    torch.save(
        {
            "epoch": last_completed_epoch,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": (
                history["train_loss"][-1] if history["train_loss"] else float("nan")
            ),
            "config": config,
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "train_pairs": _serialize_pairs(train_pairs),
            "val_pairs": _serialize_pairs(val_pairs),
            "final_metrics": final_metrics,
            "final_metrics_hamiltonian_detailed": final_detailed_metrics,
            "wandb_run_id": wandb.run.id if wandb.run is not None else None,
            "wandb_project": args.wandb_project,
            "wandb_entity": wandb_kwargs.get("entity"),
            "wandb_run_name": wandb.run.name if wandb.run is not None else None,
        },
        final_model_path,
    )
    _update_wandb_checkpoint_summary(
        latest_checkpoint_path=latest_checkpoint_path,
        final_model_path=final_model_path,
    )

    log_study_complete(
        run_name=run_name,
        total_training_epochs=(
            max(last_completed_epoch + 1, 0)
            if use_global_display_epoch
            else max(len(history["train_loss"]), 0)
        ),
        final_loss=history["train_loss"][-1] if history["train_loss"] else float("nan"),
        best_loss=best_score if best_score < float("inf") else float("nan"),
        best_epoch=(best_epoch + 1) if best_epoch >= 0 else -1,
        run_checkpoint_dir=run_checkpoint_dir,
        final_model_path=final_model_path,
    )

    if args.generate_video:
        for matrix_name in matrix_targets:
            try:
                video_path = run_checkpoint_dir / f"training_progress_{matrix_name}.mp4"
                compile_frames_to_video(
                    matrix_frame_output_dirs[matrix_name],
                    video_path,
                    fps=5,
                    pattern="frame_epoch_*.png",
                    format="mp4",
                )
                wandb.log(
                    {
                        f"training_video_{matrix_alias.get(matrix_name, matrix_name)}": wandb.Video(
                            str(video_path), fps=5, format="mp4"
                        )
                    }
                )
            except Exception as exc:
                print(f"[WARN] Could not generate final video for {matrix_name}: {exc}")

    wandb.finish()


if __name__ == "__main__":
    main()
