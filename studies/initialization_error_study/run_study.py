import sys
from pathlib import Path
import random
import argparse

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402
from data.factory import DatasetFactory  # noqa: E402
from core.sparse_math import trace_matmul_sparse_snap_vectorized  # noqa: E402

# --- Hyperparameter Search Space ---
SEARCH_SPACE = {
    "l_max_gnn": {"distribution": "int_uniform", "min": 2, "max": 3},
    "num_layers_gnn": {"distribution": "int_uniform", "min": 2, "max": 4},
    "num_layers_matrix": {"distribution": "int_uniform", "min": 1, "max": 3},
    "edge_update_linear": {"values": ["pre", "post"]},
    "edge_update": {"values": ["tensor_product", "concat", "replace"]},
    "edge_update_residual": {"values": [True, False]},
    "head_use_mlp_log_scale": {"values": [True, False]},
    "nonlin_kind": {"values": ["normact", "gate_scalars_mlp", "gate_magnitudes"]},
    "activation_scalar": {"values": ["silu", "leakyrelu", "tanh"]},
    "activation_gate": {"values": ["sigmoid", "leakyrelu", "tanh"]},
    "hidden_base_dim": {"values": [64]},
    "neck_depth": {"values": [2, 3, 4, 5]},
    "head_depth": {"values": [2, 3, 4]},
}


def sample_hyperparameters():
    """Randomly samples a configuration from the SEARCH_SPACE."""
    config = {}
    for param, spec in SEARCH_SPACE.items():
        if "distribution" in spec and spec["distribution"] == "int_uniform":
            config[param] = random.randint(spec["min"], spec["max"])
        elif "values" in spec:
            config[param] = random.choice(spec["values"])
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Initialization Error Study"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of random configurations to test.",
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Run a quick test with 5 real evaluations and 95 dummy ones.",
    )
    args = parser.parse_args()

    # 1. Load Silicon Snapshot Data (one sample per l_max_gnn)
    print("Loading silicon snapshot data for all required l_max_gnn values...")
    data_samples = {}
    l_max_gnn_space = SEARCH_SPACE["l_max_gnn"]
    for l_max in range(l_max_gnn_space["min"], l_max_gnn_space["max"] + 1):
        print(f"  Generating data for l_max_gnn = {l_max}...")
        data_cfg = Config(cutoff_gnn=5.0, cutoff_matrix=8.0, l_max_gnn=l_max)
        fac = DatasetFactory(data_cfg)
        fac.add_snapshot(
            "../../data/big/silicon/900K/Si_DM",
            "../../data/big/silicon/900K/info.txt",
        )
        train_ds, _, mapper = fac.create()
        data_samples[l_max] = (train_ds[0], mapper)
    print("Data loaded.")

    # 2. Main Evaluation Loop
    results = []
    num_real_evals = 5 if args.test_run else args.num_samples

    for i in tqdm(range(args.num_samples), desc="Evaluating Configurations"):
        # Sample a random configuration
        hparams = sample_hyperparameters()

        if i < num_real_evals:
            # Get the correct data sample for the sampled l_max_gnn
            l_max_gnn = hparams["l_max_gnn"]
            x, y = data_samples[l_max_gnn][0]
            mapper = data_samples[l_max_gnn][1]

            # Create config and model
            cfg = Config(**hparams)
            model = E3GNN(mapper, cfg)
            model.eval()

            # Run forward pass and compute error
            with torch.no_grad():
                preds = model(x)

                blk_ham = preds["hamiltonian"].to_blocks(mapper)
                blk_den = preds["density"].to_blocks(mapper)

                E_pred = trace_matmul_sparse_snap_vectorized(blk_ham, blk_den)
                E_true = y["energy"]
                error = E_pred.item() - E_true.item()
        else:
            # For test runs, fill the rest with dummy data
            error = random.gauss(0, 1)  # Dummy error

        # Store results
        result_row = hparams.copy()
        result_row["energy_error"] = error
        results.append(result_row)

    # 3. Data Analysis and Plotting
    df = pd.DataFrame(results)
    output_dir = Path(__file__).parent
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / "results.csv", index=False)
    print(
        f"\nSaved results for {args.num_samples} runs to {output_dir / 'results.csv'}"
    )

    print("Generating plots...")
    for param in SEARCH_SPACE.keys():
        plt.figure(figsize=(10, 6))

        if "distribution" in SEARCH_SPACE[param]:  # Numerical parameter
            # Use regplot for scatter + best-fit line
            sns.regplot(data=df, x=param, y="energy_error", scatter_kws={"alpha": 0.5})
            plt.xscale("log")
        else:  # Categorical parameter
            # Use stripplot to see individual points and boxplot for distribution summary
            sns.stripplot(data=df, x=param, y="energy_error", alpha=0.5, jitter=True)
            sns.boxplot(data=df, x=param, y="energy_error", color="white", fliersize=0)

        plt.title(f"Energy Error vs. {param}")
        plt.ylabel("Residual Energy Error (Predicted - True)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = plots_dir / f"{param}_vs_error.png"
        plt.savefig(plot_path)
        plt.close()

    print(f"Saved all plots to {plots_dir}")


if __name__ == "__main__":
    main()
