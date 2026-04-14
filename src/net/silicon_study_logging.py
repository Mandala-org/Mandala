from __future__ import annotations

from pathlib import Path
from typing import Iterable


MATRIX_ALIAS = {"hamiltonian": "H", "overlap": "S", "density": "D"}
IRREP_PREFIX_BY_MATRIX = {"hamiltonian": "H_", "overlap": "S_", "density": "D_"}


def should_log_epoch(epoch_zero_based: int, log_interval: int, adaptive: bool) -> bool:
    log_interval = max(int(log_interval), 1)
    if not adaptive:
        return epoch_zero_based % log_interval == 0
    if epoch_zero_based <= 10:
        return True
    if epoch_zero_based < 100:
        return epoch_zero_based % 10 == 0
    return epoch_zero_based % log_interval == 0


def log_config(config: dict, run_checkpoint_dir: Path, frame_output_dir: Path) -> None:
    print("\n[CONFIG] Setting up hyperparameters...")
    print(f"  Run-specific checkpoint directory: {run_checkpoint_dir}")
    print(f"  Frame output directory: {frame_output_dir}")
    print(f"  Device: {config['device']}")
    print(f"  Hidden irreps: {config.get('hidden_irreps')}")
    print(f"  L_max: {config['l_max']}")
    print(f"  Num layers: {config['num_layers_gnn']}")
    print(
        "  e3LayerNorm: enabled"
        if config.get("e3layernorm", True)
        else "  e3LayerNorm: disabled"
    )
    print(
        "  Edge encoder SH TensorSquare: enabled"
        if config.get("edge_encoder_use_sh_tensor_square", False)
        else "  Edge encoder SH TensorSquare: disabled"
    )
    print(
        "  Head TensorSquare: enabled"
        if config.get("head_use_tensor_square", False)
        else "  Head TensorSquare: disabled"
    )
    print(
        "  Node embeddings for self edges: enabled"
        if config.get("head_use_node_embeddings_for_self_edges", False)
        else "  Node embeddings for self edges: disabled"
    )
    print(f"  Cutoff radius: {config['cutoff_radius']} A")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Epochs: {config['max_epochs']}")
    print(f"  Checkpoint directory: {config['save_dir']}")
    grad_clip = config.get("grad_clip_val")
    print(
        f"  Gradient clipping: {grad_clip}"
        if grad_clip is not None and grad_clip > 0
        else "  Gradient clipping: disabled"
    )
    print(
        f"  Separate shifted-self head: {'enabled' if config.get('separate_shifted_self', False) else 'disabled'}"
    )
    print(
        f"  Target cutoff filtering: enabled (using cutoff_radius={config['cutoff_radius']} A)"
        if config.get("apply_cutoff_to_targets", True)
        else "  Target cutoff filtering: disabled"
    )
    print(
        f"  Require exact edge match: {'enabled' if config.get('require_exact_edge_match', False) else 'disabled'}"
    )
    if config.get("adaptive_log_interval", False):
        print(
            f"  Adaptive logging: enabled (1-10: every epoch, 11-100: every 10, >100: every {config['log_interval']})"
        )
    else:
        print(f"  Adaptive logging: disabled (every {config['log_interval']} epochs)")
    print(f"  Log data: {'enabled' if config.get('log_data', False) else 'disabled'}")
    print(f"  Log model: {'enabled' if config.get('log_model', False) else 'disabled'}")
    print(
        f"  Log forward console details: {'enabled' if config.get('log_forward', False) else 'disabled'}"
    )
    print(
        f"  Log per-irrep metrics console details: {'enabled' if config.get('print_per_irrep_metrics', config.get('log_per_irrep_metrics', False)) else 'disabled'}"
    )
    print(
        f"  Benchmark timing: {'enabled' if config.get('benchmark', False) else 'disabled'}"
    )
    print(
        f"  Video generation: {'enabled' if config.get('generate_video', False) else 'disabled'}"
    )


def log_snapshot_info(x: dict, y: dict) -> None:
    print("\n[DATA] Snapshot loaded:")
    print(f"  Elements: {x.get('atoms')}")
    print(f"  Num atoms: {len(x.get('atoms', []))}")
    if "positions" in x:
        print(f"  Positions shape: {tuple(x['positions'].shape)}")
    print(f"  Box shape: {None if x.get('box') is None else tuple(x['box'].shape)}")
    for name in ("hamiltonian", "overlap", "density"):
        if name in y:
            print(f"  {name.capitalize()} keys: {list(y[name].pair_blocks.keys())}")


def log_mapper_info(mapper) -> None:
    print("\n[MAPPER] Creating BlockIrrepMapper...")
    print(f"  Mapper edge types: {mapper.edge_types}")
    print(f"  Mapper edge_type2idx: {mapper.edge_type2idx}")


def log_cutoff_application(
    before_edges: int, after_edges: int, cutoff_radius: float
) -> None:
    print(
        f"\n  Applied cutoff to GT matrices: {before_edges} -> {after_edges} Hamiltonian edges "
        f"(cutoff={cutoff_radius} A)"
    )


def log_orbital_config(orbital_cfg) -> None:
    print("\n  Orbital configuration:")
    for elem in orbital_cfg.elements():
        irreps = orbital_cfg.element_to_irreps[elem]
        print(f"    {elem}: {irreps} (dim={irreps.dim})")


def log_strict_checks_passed() -> None:
    print("  [OK] Edge alignment checks passed.")


def log_graph(x: dict) -> None:
    if "edge_index" not in x or "edge_length_emb" not in x:
        return
    atoms_list = list(x.get("atoms", ()))
    edge_index = x["edge_index"]
    num_self_edges = int(x.get("num_self_edges", 0))
    edge_length_emb = x["edge_length_emb"]
    print("\n[GRAPH] Constructing molecular graph...")
    print(f"  Atoms: {atoms_list}")
    if "positions" in x:
        print(f"  Positions:\n{x['positions']}")
    if x.get("box") is not None:
        print(f"  Box:\n{x['box']}")
    print(
        f"  Total edges: {edge_index.shape[1]} "
        f"({num_self_edges} self + {edge_index.shape[1] - num_self_edges} off-diagonal)"
    )
    print(f"  Edge length embedding shape: {tuple(edge_length_emb.shape)}")
    if "edge_sh" in x:
        print(f"  Edge SH shape: {tuple(x['edge_sh'].shape)}")


def log_per_irrep_metrics(
    title: str,
    all_irreps: Iterable,
    per_irrep_metrics: dict,
    metric_prefix: str = "",
) -> None:
    print(f"\n  {title}")
    for irrep in sorted(all_irreps, key=str):
        irrep_str = str(irrep)
        print(f"    {metric_prefix}{irrep_str:4s}:")
        for suffix, label in (
            ("l1_elem", "L1 elem"),
            ("l2_elem", "L2 elem"),
            ("l1_block", "L1 block"),
            ("l1_block_rel", "L1 block rel"),
            ("l2_block", "L2 block"),
            ("l2_block_rel", "L2 block rel"),
            ("l1_full_rel", "L1 full rel"),
            ("l2_full_rel", "L2 full rel"),
        ):
            value = per_irrep_metrics.get(f"{irrep_str}_{suffix}")
            if value is not None:
                print(f"      {label:12s}: {value:.6e}")


def log_final_metrics(final_detailed_metrics: dict) -> None:
    print("\n[FINAL PREDICTIONS - Hamiltonian]")
    print("\n  Overall Metrics:")
    print(f"    MAE H:                {final_detailed_metrics['mae']:.6e}")
    print(f"    MSE H:                {final_detailed_metrics['mse']:.6e}")
    print(f"    MAE H (modified):     {final_detailed_metrics['mae_mod']:.6e}")
    print(f"    MSE H (modified):     {final_detailed_metrics['mse_mod']:.6e}")
    print(f"    mu_H:                 {final_detailed_metrics['mu_H']:.6e}")
    print(f"    Correction MAE:       {final_detailed_metrics['correction_mae']:.6e}")
    print(f"    Correction MSE:       {final_detailed_metrics['correction_mse']:.6e}")


def log_detailed_training_metrics(
    avg_epoch_time: float,
    epochs_since_last_log: int,
    time_elapsed: float,
    loss_value: float,
    detailed_metrics: dict,
) -> None:
    print("\n[METRICS]")
    print(
        f"  Avg epoch time:       {avg_epoch_time:.3f}s "
        f"({epochs_since_last_log} epochs in {time_elapsed:.1f}s)"
    )
    print(f"  Loss (MSE):           {loss_value:.6e}")
    print(f"  MAE H:                {detailed_metrics['mae']:.6e}")
    print(f"  MSE H:                {detailed_metrics['mse']:.6e}")
    print(f"  MAE H (modified):     {detailed_metrics['mae_mod']:.6e}")
    print(f"  MSE H (modified):     {detailed_metrics['mse_mod']:.6e}")
    print(f"  mu_H:                 {detailed_metrics['mu_H']:.6e}")
    print(f"  Correction MAE:       {detailed_metrics['correction_mae']:.6e}")
    print(f"  Correction MSE:       {detailed_metrics['correction_mse']:.6e}")


def build_wandb_detailed_metrics_log(
    epoch_zero_based: int, loss_value: float, detailed_metrics: dict
) -> dict:
    return {
        "epoch": epoch_zero_based,
        "loss": loss_value,
        "mse_H": detailed_metrics["mse"],
        "mae_H": detailed_metrics["mae"],
        "mae_H_mod": detailed_metrics["mae_mod"],
        "mse_H_mod": detailed_metrics["mse_mod"],
        "mu_H": detailed_metrics["mu_H"],
        "correction_mae": detailed_metrics["correction_mae"],
        "correction_mse": detailed_metrics["correction_mse"],
    }


def build_wandb_per_irrep_metrics_log(
    epoch_zero_based: int,
    all_irreps,
    per_irrep_metrics: dict,
    metric_prefix: str = "",
) -> dict:
    payload = {"epoch": epoch_zero_based}
    for irrep in sorted(all_irreps, key=str):
        irrep_str = str(irrep)
        prefix_ir = f"{metric_prefix}{irrep_str}"
        for suffix in (
            "l1_elem",
            "l2_elem",
            "l1_block",
            "l1_block_rel",
            "l2_block",
            "l2_block_rel",
            "l1_full_rel",
            "l2_full_rel",
        ):
            payload[f"irrep_metrics/{prefix_ir}_{suffix}"] = per_irrep_metrics.get(
                f"{irrep_str}_{suffix}", 0.0
            )
    return payload


def log_study_complete(
    run_name: str,
    total_training_epochs: int,
    final_loss: float,
    best_loss: float,
    best_epoch: int,
    run_checkpoint_dir: Path,
    final_model_path: Path,
) -> None:
    print("")
    print("=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)
    print(f"\nRun name: {run_name}")
    print(f"Total training epochs: {total_training_epochs}")
    print(f"Final loss: {final_loss:.6e}")
    print(f"Best loss: {best_loss:.6e} (epoch {best_epoch})")
    print(f"\nCheckpoints saved to: {run_checkpoint_dir}")
    print(f"  Best model: {run_checkpoint_dir / 'best_model.pt'}")
    print(f"  Final model: {final_model_path}")
    print("\n[OK] Silicon study-style logging finished successfully!")
