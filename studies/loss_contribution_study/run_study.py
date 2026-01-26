import sys
from pathlib import Path
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import gc
import os
import psutil

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from data.factory import DatasetFactory  # noqa: E402


def log_memory(stage=""):
    """Logs the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{stage}] Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Loss Contribution Study")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of training runs to perform.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs for each training run.",
    )
    parser.add_argument(
        "--loss_coef_observable",
        type=float,
        required=True,
        help="Coefficient for energy and electron number loss.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--train_target",
        type=str,
        choices=["irreps", "matrix"],
        required=True,
        help="Target for the main matrix loss.",
    )
    args = parser.parse_args()

    log_memory("Study Start")

    # --- Output Directory Setup ---
    output_dir = (
        Path(__file__).parent
        / f"target_{args.train_target}_lr_{args.lr}_coef_{args.loss_coef_observable}"
    )
    output_dir.mkdir(exist_ok=True)
    print(f"Saving results to: {output_dir}")

    print("Loading silicon snapshot data...")
    data_cfg = Config(cutoff_radius=8.0)
    fac = DatasetFactory(data_cfg)
    fac.add_snapshot(
        "../../data/big/silicon/900K/Si_DM", "../../data/big/silicon/900K/info.txt"
    )
    train_ds, _, mapper = fac.create()

    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=lambda b: b[0])
    print("Data loaded.")
    log_memory("Data Loaded")

    # --- Main Experiment Loop ---
    all_runs_df = []
    for i in tqdm(range(args.num_samples), desc="Running Training Samples"):
        print(f"\n--- Starting Run {i+1}/{args.num_samples} ---")
        log_memory(f"Loop Start (Run {i+1})")

        run_log_dir = output_dir / f"temp_run_{i}"

        # 1. Configure and Train
        cfg = Config(
            lr=args.lr,
            train_target=args.train_target,
            loss_coef_observables=args.loss_coef_observable,
            max_epochs=args.num_epochs,
            use_lr_scheduler=False,
        )
        print(
            f"Config for Run {i+1}: lr={cfg.lr}, target={cfg.train_target}, coef={cfg.loss_coef_observables}"
        )
        log_memory(f"After Config (Run {i+1})")

        model = E3GNN(mapper, cfg)
        log_memory(f"After Model Init (Run {i+1})")

        logger = CSVLogger(save_dir=str(run_log_dir))
        log_memory(f"After Logger Init (Run {i+1})")

        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cuda",
            log_every_n_steps=1,
        )
        log_memory(f"After Trainer Init (Run {i+1})")

        print("Starting training...")
        trainer.fit(model=model, train_dataloaders=train_loader)
        print("Finished training.")
        log_memory(f"After Fit (Run {i+1})")

        # 2. Load and process results
        results_path = Path(logger.log_dir) / "metrics.csv"
        if results_path.exists():
            run_df = pd.read_csv(results_path)
            run_df["run_id"] = i
            all_runs_df.append(run_df)
            log_memory(f"After Results Load (Run {i+1})")

        # Clean up
        shutil.rmtree(run_log_dir)
        del model
        del trainer
        del logger
        gc.collect()
        print("Cleaned up objects for the run.")
        log_memory(f"After Cleanup (Run {i+1})")

    # --- Data Aggregation and Plotting ---
    if not all_runs_df:
        print("No training data was generated. Exiting.")
        return

    full_df = pd.concat(all_runs_df)
    full_df.to_csv(output_dir / "all_runs_metrics.csv", index=False)
    print(f"\nSaved combined metrics to {output_dir / 'all_runs_metrics.csv'}")

    # --- Figure and Subplots ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(
        f"Loss Analysis (target={args.train_target}, lr={args.lr}, coef={args.loss_coef_observable})",
        fontsize=16,
    )

    # --- Plot 1: Loss Contribution Percentage ---
    percent_cols = ["train_percent_matrix", "train_percent_E", "train_percent_N"]
    df_cleaned = full_df.dropna(subset=["epoch"])
    df_cleaned[percent_cols] = df_cleaned[percent_cols].ffill()

    df_melted_percent = df_cleaned.melt(
        id_vars=["epoch"],
        value_vars=percent_cols,
        var_name="loss_component",
        value_name="percentage",
    )

    sns.lineplot(
        data=df_melted_percent,
        x="epoch",
        y="percentage",
        hue="loss_component",
        errorbar=("pi", 50),
        ax=axes[0],
    )
    axes[0].set_title("Loss Contribution Analysis")
    axes[0].set_ylabel("Percentage of Total Loss")
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)
    axes[0].set_ylim(0, 100)

    # --- Plot 2: Mean Absolute Errors (Log Scale) ---
    mae_cols = ["train_mae_matrix", "train_abs_error_E", "train_abs_error_N"]
    df_cleaned[mae_cols] = df_cleaned[mae_cols].ffill()

    df_melted_mae = df_cleaned.melt(
        id_vars=["epoch"],
        value_vars=mae_cols,
        var_name="error_component",
        value_name="mae",
    )

    sns.lineplot(
        data=df_melted_mae,
        x="epoch",
        y="mae",
        hue="error_component",
        errorbar=("pi", 50),
        ax=axes[1],
    )
    axes[1].set_title("Mean Absolute Error (MAE) Analysis")
    axes[1].set_ylabel("Mean Absolute Error (Log Scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- Save Figure ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = output_dir / "loss_analysis_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved combined plot to {plot_path}")
    log_memory("Study End")


if __name__ == "__main__":
    main()
