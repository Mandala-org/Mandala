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

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from net.benchmark import BenchmarkCallback  # noqa: E402
from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from data.factory import DatasetFactory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Overfitting Study on a Single Snapshot"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of independent training runs to perform.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs for each training run.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--train_target",
        type=str,
        choices=["irreps", "matrix"],
        default="irreps",
        help="Target for the main matrix loss.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="overfitting_results",
        help="Folder to save the results.",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        action="store_true",
        help="Enable ReduceLROnPlateau learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler_patience",
        type=int,
        default=10,
        help="Patience for ReduceLROnPlateau scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler_factor",
        type=float,
        default=0.5,
        help="Factor for ReduceLROnPlateau scheduler.",
    )
    args = parser.parse_args()

    # --- Output Directory Setup ---
    output_dir = Path(__file__).parent / args.output_folder
    output_dir.mkdir(exist_ok=True)
    print(f"Saving results to: {output_dir}")

    # --- Data Loading ---
    print("Loading single silicon snapshot...")
    cfg = Config(
        cutoff_gnn=5.0,
        cutoff_matrix=8.0,
        lr=args.lr,
        train_target=args.train_target,
        loss_coef_observables=0.0,  # Disable observable losses
        train_on_energy=False,
        train_on_num_electrons=False,
        scheduler_target="train_loss_matrix",
        max_epochs=args.num_epochs,
        use_lr_scheduler=args.use_lr_scheduler,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        hidden_base_dim=64,
        num_layers_gnn=4,
        num_layers_matrix=2,
        neck_depth=3,
        head_depth=2,
        bench_verbosity=2,
        log_activation_mag=True,
    )
    fac = DatasetFactory(cfg)
    # Using a small water snapshot for faster testing, but can be changed
    fac.add_snapshot(
        "../../data/big/silicon/900K/Si_DM",
        "../../data/big/silicon/900K/info.txt",
    )
    train_ds, _, mapper = fac.create()
    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=lambda b: b[0])
    print("Data loaded.")

    # --- Main Experiment Loop ---
    all_runs_df = []
    for i in tqdm(range(args.num_samples), desc="Running Training Samples"):
        run_log_dir = output_dir / f"temp_run_{i}"

        # 1. Configure and Train

        callbacks = [
            BenchmarkCallback(
                verbosity=cfg.bench_verbosity, log_activation_mag=cfg.log_activation_mag
            ),
        ]

        model = E3GNN(mapper, cfg)
        logger = CSVLogger(save_dir=str(run_log_dir))
        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            logger=logger,
            callbacks=callbacks,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            accelerator="cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu",
            log_every_n_steps=1,
        )

        trainer.fit(model=model, train_dataloaders=train_loader)

        # 2. Load and process results
        results_path = Path(logger.log_dir) / "metrics.csv"
        if results_path.exists():
            run_df = pd.read_csv(results_path)
            run_df["run_id"] = i
            all_runs_df.append(run_df)

        # Clean up
        shutil.rmtree(run_log_dir)
        del model, trainer, logger
        gc.collect()

    # --- Data Aggregation and Plotting ---
    if not all_runs_df:
        print("No training data was generated. Exiting.")
        return

    full_df = pd.concat(all_runs_df)
    full_df.to_csv(output_dir / "overfitting_metrics.csv", index=False)
    print(f"\nSaved combined metrics to {output_dir / 'overfitting_metrics.csv'}")

    # --- Plotting ---
    error_cols = ["train_loss_matrix", "train_mae_matrix"]
    df_cleaned = full_df.dropna(subset=["epoch"] + error_cols)
    df_cleaned = df_cleaned.ffill()

    df_melted = df_cleaned.melt(
        id_vars=["epoch"],
        value_vars=error_cols,
        var_name="error_metric",
        value_name="error_value",
    )

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_melted,
        x="epoch",
        y="error_value",
        hue="error_metric",
        errorbar=("pi", 50),  # 50% percentile interval as error band
    )
    plt.title("Overfitting a Single Snapshot (10 Runs)")
    plt.ylabel("Error (Log Scale)")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plot_path = output_dir / "overfitting_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
