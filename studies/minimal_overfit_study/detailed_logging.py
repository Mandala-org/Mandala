"""Compact, semantic logging helpers for the minimal overfit study."""

from __future__ import annotations


def log_config(config: dict, run_checkpoint_dir, frame_output_dir) -> None:
    print("\n[CONFIG] Setting up hyperparameters...")
    print(f"  Run-specific checkpoint directory: {run_checkpoint_dir}")
    print(f"  Frame output directory: {frame_output_dir}")
    print(f"  Device: {config['device']}")
    print(f"  Box convention: {config.get('box_convention', 'rows')}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  L_max: {config['l_max']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Cutoff radius: {config['cutoff_radius']} A")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Checkpoint directory: {config['checkpoint_dir']}")
    print(
        f"  Gradient clipping: {config['grad_clip']}"
        if config["grad_clip"] > 0
        else "  Gradient clipping: disabled"
    )
    print(
        f"  Partial training: {config['partial_train']} blocks only"
        if config["partial_train"] is not None
        else "  Partial training: disabled (training on all blocks)"
    )
    print(
        "  Train on irrep parts: enabled (decomposed per-irrep loss)"
        if config["train_on_irrep_parts"]
        else "  Train on irrep parts: disabled (standard loss)"
    )
    print(
        "  Video generation: enabled"
        if config["generate_video"]
        else "  Video generation: disabled"
    )
    print(
        "  Verbose forward pass: enabled (showing shapes and irreps)"
        if config["verbose_forward"]
        else "  Verbose forward pass: disabled"
    )
    print(
        "  Block normalization: enabled (per-key, per-diagonal-status)"
        if config["normalize_blocks"]
        else "  Block normalization: disabled"
    )
    print(
        f"  Target cutoff filtering: enabled (using cutoff_radius={config['cutoff_radius']} A)"
        if config["apply_cutoff_to_targets"]
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
        f"  Log per-irrep metrics console details: {'enabled' if config.get('log_per_irrep_metrics', False) else 'disabled'}"
    )
    print(
        f"  Benchmark timing: {'enabled' if config.get('benchmark', False) else 'disabled'}"
    )


def log_snapshot_info(snapshot) -> None:
    print("\n[DATA] Snapshot loaded:")
    print(f"  Elements: {snapshot.hamiltonian.atoms}")
    print(f"  Num atoms: {len(snapshot.hamiltonian.atoms)}")
    print(f"  Positions shape: {snapshot.positions.shape}")
    print(f"  Box shape: {snapshot.box.shape if snapshot.box is not None else None}")
    print(f"  Basis: {snapshot.hamiltonian.basis}")
    print(f"  Hamiltonian keys: {list(snapshot.hamiltonian.pair_blocks.keys())}")
    print(f"  Overlap keys: {list(snapshot.overlap.pair_blocks.keys())}")
    print(f"  Density keys: {list(snapshot.density.pair_blocks.keys())}")


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
    print("  ✓ Edge alignment checks passed.")


def log_graph(
    atoms_list,
    positions,
    box,
    cutoff_radius: float,
    src,
    dst,
    offsets,
    edge_index,
    num_self_edges: int,
    edge_dist,
) -> None:
    print("\n[GRAPH] Constructing molecular graph...")
    print(f"  Atoms: {atoms_list}")
    print(f"  Positions (e3nn frame):\n{positions}")
    if box is not None:
        print(f"  Box (e3nn frame):\n{box}")
    print(f"\n  Finding neighbors within {cutoff_radius} A...")
    print(f"  Found {len(src)} off-diagonal edges")
    print("  Edge list (first 10):")
    for i in range(min(10, len(src))):
        print(f"    {src[i]} -> {dst[i]} (offset: {offsets[i]})")
    print(
        f"  Total edges: {edge_index.shape[1]} "
        f"({num_self_edges} self + {edge_index.shape[1] - num_self_edges} off-diagonal)"
    )
    print("\n  Computing edge displacements and distances...")
    print(
        f"  Edge distances (A): min={edge_dist.min().item():.3f}, "
        f"max={edge_dist.max().item():.3f}, mean={edge_dist.mean().item():.3f}"
    )
    print(f"  Self-edge distances: {edge_dist[:num_self_edges]}")


def log_per_irrep_metrics(title: str, all_irreps, per_irrep_metrics: dict) -> None:
    print(f"\n  {title}")
    for irrep in sorted(all_irreps, key=str):
        irrep_str = str(irrep)
        l1_elem = per_irrep_metrics.get(f"{irrep_str}_l1_elem", 0.0)
        l2_elem = per_irrep_metrics.get(f"{irrep_str}_l2_elem", 0.0)
        l1_block = per_irrep_metrics.get(f"{irrep_str}_l1_block", 0.0)
        l1_block_rel = per_irrep_metrics.get(f"{irrep_str}_l1_block_rel", 0.0)
        l2_block = per_irrep_metrics.get(f"{irrep_str}_l2_block", 0.0)
        l2_block_rel = per_irrep_metrics.get(f"{irrep_str}_l2_block_rel", 0.0)
        l1_full_rel = per_irrep_metrics.get(f"{irrep_str}_l1_full_rel", 0.0)
        l2_full_rel = per_irrep_metrics.get(f"{irrep_str}_l2_full_rel", 0.0)

        print(f"    {irrep_str:4s}:")
        print(f"      Element-level: L1(MAE)={l1_elem:.3e} L2(RMSE)={l2_elem:.3e}")
        print(
            f"      Block-level:   L1={l1_block:.3e} (rel={l1_block_rel:.3%}), "
            f"L2={l2_block:.3e} (rel={l2_block_rel:.3%})"
        )
        print(
            f"      Full-matrix:   L1_rel={l1_full_rel:.3%}, L2_rel={l2_full_rel:.3%}"
        )


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


def log_training_failed_banner(epoch_one_based: int) -> None:
    print("")
    print(f"\n{'='*80}")
    print(f"TRAINING FAILED at epoch {epoch_one_based}")
    print(f"{'='*80}")


def log_invalid_loss_failure(
    epoch_zero_based: int,
    loss_value: float,
    is_nan: bool,
    last_valid_loss,
    best_epoch_zero_based: int,
    best_loss: float,
) -> dict:
    log_training_failed_banner(epoch_zero_based + 1)
    print("❌ Detected invalid loss value.")
    if is_nan:
        print(f"Loss is NaN: {loss_value}")
        print("This indicates numerical instability in the forward pass.")
        failure_type = "NaN"
    else:
        print(f"Loss is Inf: {loss_value}")
        print("Loss has exploded. Try reducing learning rate or adding regularization.")
        failure_type = "Inf"
    print(f"\nLast valid loss (epoch {epoch_zero_based}): {last_valid_loss}")
    print(
        f"\nTraining stopped. Checkpoint saved at epoch {best_epoch_zero_based + 1} "
        f"with loss {best_loss:.6e}"
    )
    print(f"{'='*80}\n")
    return {
        "training_failed": True,
        "failure_epoch": epoch_zero_based,
        "failure_type": failure_type,
    }


def log_invalid_gradient_failure(
    epoch_zero_based: int,
    param_name: str,
    lr: float,
    grad_clip: float,
    last_valid_loss,
    best_epoch_zero_based: int,
    best_loss: float,
) -> dict:
    log_training_failed_banner(epoch_zero_based + 1)
    print("❌ Detected invalid gradient values.")
    print(f"Gradient contains NaN or Inf in parameter: {param_name}")
    print("This indicates numerical instability in the backward pass.")
    print("\nSuggestions:")
    print(f"  - Reduce learning rate (current: {lr})")
    print(f"  - Increase gradient clipping (current: {grad_clip})")
    print("  - Check if irreps contain invalid combinations")
    print(f"\nLast valid loss (epoch {epoch_zero_based}): {last_valid_loss}")
    print(
        f"Training stopped. Checkpoint saved at epoch {best_epoch_zero_based + 1} "
        f"with loss {best_loss:.6e}"
    )
    print(f"{'='*80}\n")
    return {
        "training_failed": True,
        "failure_epoch": epoch_zero_based,
        "failure_type": "gradient_nan_inf",
        "failed_param": param_name,
    }


def log_detailed_training_metrics(
    avg_epoch_time: float,
    epochs_since_last_log: int,
    time_elapsed: float,
    loss_value: float,
    detailed_metrics: dict,
    irrep_losses: dict | None = None,
) -> None:
    print("\n[METRICS]")
    print(
        f"  Avg epoch time:       {avg_epoch_time:.3f}s "
        f"({epochs_since_last_log} epochs in {time_elapsed:.1f}s)"
    )
    print(f"  Loss (MSE):           {loss_value:.6e}")

    if irrep_losses:
        print("\n  Per-Irrep Loss Contributions:")
        sorted_irreps = sorted(
            irrep_losses.items(), key=lambda x: x[1].item(), reverse=True
        )
        total_loss_check = sum(irrep_loss.item() for _, irrep_loss in sorted_irreps)
        for irrep_str, irrep_loss in sorted_irreps:
            percentage = (
                (irrep_loss.item() / total_loss_check * 100)
                if total_loss_check > 0
                else 0
            )
            print(f"    {irrep_str:4s}: {irrep_loss.item():.6e} ({percentage:5.1f}%)")

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
    epoch_zero_based: int, all_irreps, per_irrep_metrics: dict
) -> dict:
    payload = {"epoch": epoch_zero_based}
    for irrep in sorted(all_irreps, key=str):
        irrep_str = str(irrep)
        payload[f"irrep_metrics/{irrep_str}_l1_elem"] = per_irrep_metrics.get(
            f"{irrep_str}_l1_elem", 0.0
        )
        payload[f"irrep_metrics/{irrep_str}_l2_elem"] = per_irrep_metrics.get(
            f"{irrep_str}_l2_elem", 0.0
        )
        payload[f"irrep_metrics/{irrep_str}_l1_block"] = per_irrep_metrics.get(
            f"{irrep_str}_l1_block", 0.0
        )
        payload[f"irrep_metrics/{irrep_str}_l1_block_rel"] = per_irrep_metrics.get(
            f"{irrep_str}_l1_block_rel", 0.0
        )
        payload[f"irrep_metrics/{irrep_str}_l2_block"] = per_irrep_metrics.get(
            f"{irrep_str}_l2_block", 0.0
        )
        payload[f"irrep_metrics/{irrep_str}_l2_block_rel"] = per_irrep_metrics.get(
            f"{irrep_str}_l2_block_rel", 0.0
        )
        payload[f"irrep_metrics/{irrep_str}_l1_full_rel"] = per_irrep_metrics.get(
            f"{irrep_str}_l1_full_rel", 0.0
        )
        payload[f"irrep_metrics/{irrep_str}_l2_full_rel"] = per_irrep_metrics.get(
            f"{irrep_str}_l2_full_rel", 0.0
        )
    return payload


def log_study_complete(
    run_name: str,
    total_training_epochs: int,
    final_loss: float,
    best_loss: float,
    best_epoch: int,
    run_checkpoint_dir,
    final_model_path,
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
    print("\n✓ Minimal overfit study finished successfully!")
