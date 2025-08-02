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
    data_cfg = Config(cutoff_gnn=5.0, cutoff_matrix=8.0)
    fac = DatasetFactory(data_cfg)
    fac.add_snapshot("../../data/big/silicon/900K/Si_DM", "../../data/big/silicon/900K/info.txt")
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
            loss_coef_energy=args.loss_coef_observable,
            loss_coef_num_electrons=args.loss_coef_observable,
            max_epochs=args.num_epochs,
            use_lr_scheduler=False,
        )
        print(
            f"Config for Run {i+1}: lr={cfg.lr}, target={cfg.train_target}, coef={cfg.loss_coef_energy}"
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
            accelerator="cpu",
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

    percent_cols = ["train_percent_matrix", "train_percent_E", "train_percent_N"]
    full_df = full_df.dropna(subset=["epoch"])
    full_df[percent_cols] = full_df[percent_cols].ffill()

    df_melted = full_df.melt(
        id_vars=["epoch"],
        value_vars=percent_cols,
        var_name="loss_component",
        value_name="percentage",
    )

    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_melted,
        x="epoch",
        y="percentage",
        hue="loss_component",
        errorbar=("pi", 50),
    )
    plt.title(
        f"Loss Contribution Analysis\n(target={args.train_target}, lr={args.lr}, coef={args.loss_coef_observable})"
    )
    plt.ylabel("Percentage of Total Loss")
    plt.xlabel("Epoch")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ylim(0, 100)

    plot_path = output_dir / "loss_contribution_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")
    log_memory("Study End")


if __name__ == "__main__":
    main()
