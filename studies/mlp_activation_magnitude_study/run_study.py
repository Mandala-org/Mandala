import sys
from pathlib import Path
import collections

import torch
import yaml
import matplotlib.pyplot as plt
from e3nn.o3 import Irreps, Linear
from torch import nn
import re

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from net.common import Config, build_hidden_irreps  # noqa: E402
from net.activations import make_nonlinearity  # noqa: E402

# --- Study Configuration ---
L_MAX_VALUES = [2, 3, 4]
HIDDEN_BASE_DIM = 32
NUM_LAYERS = 10
NUM_SEEDS = 10
# Updated list based on activations.py. s2act is skipped because it requires all multiplicities to be 1.
ACTIVATION_FUNCTIONS = ["normact", "gate_scalars_mlp", "gate_magnitudes"]
OUTPUT_DIR = Path(__file__).parent


def get_average_magnitudes(features: torch.Tensor, irreps: Irreps) -> dict[str, float]:
    """Computes the average magnitude for each irrep in a feature tensor."""
    mags = {}
    current_dim = 0
    for mul, ir in irreps:
        slice_dim = mul * ir.dim
        if slice_dim == 0:
            continue

        # Slice the tensor for the current irrep
        act_slice = features[:, current_dim : current_dim + slice_dim]

        # Reshape to (batch * multiplicity, irrep_dimension)
        reshaped_slice = act_slice.reshape(-1, ir.dim)

        # Compute the norm (magnitude) and then the average
        if reshaped_slice.shape[0] > 0:
            magnitude = torch.linalg.norm(reshaped_slice, dim=1).mean().item()
            mags[str(ir)] = magnitude

        current_dim += slice_dim
    return mags


def main():
    """Runs the MLP activation magnitude study."""
    for act_func_name in ACTIVATION_FUNCTIONS:
        print(f"--- Running study for activation: {act_func_name} ---")
        all_l_max_results = {}

        for l_max in L_MAX_VALUES:
            hidden_irreps = build_hidden_irreps(l_max, HIDDEN_BASE_DIM)
            print(f"\n  l_max = {l_max}, hidden_irreps = {hidden_irreps}")

            seed_results = collections.defaultdict(
                lambda: collections.defaultdict(list)
            )

            for seed in range(NUM_SEEDS):
                print(f"    Running seed {seed+1}/{NUM_SEEDS}")
                torch.manual_seed(seed)

                # 1. Build the MLP
                mlp_layers = []
                # Special config for gate_magnitudes
                if act_func_name == "gate_magnitudes":
                    cfg = Config(
                        nonlin_kind=act_func_name, activation_magnitude="sigmoid"
                    )
                else:
                    cfg = Config(nonlin_kind=act_func_name)

                for _ in range(NUM_LAYERS):
                    lin = Linear(hidden_irreps, hidden_irreps)
                    act = make_nonlinearity(hidden_irreps, cfg)
                    mlp_layers.append(nn.ModuleList([lin, act]))
                mlp = nn.ModuleList(mlp_layers)

                # 2. Create random input
                x = torch.randn(1, hidden_irreps.dim)

                # 3. Forward pass and record magnitudes
                for i, (lin, act) in enumerate(mlp):
                    x_before = lin(x)
                    mags_before = get_average_magnitudes(x_before, hidden_irreps)
                    for ir_str, mag in mags_before.items():
                        seed_results[ir_str][f"layer_{i+1}_before"].append(mag)

                    x_after = act(x_before)
                    mags_after = get_average_magnitudes(x_after, hidden_irreps)
                    for ir_str, mag in mags_after.items():
                        seed_results[ir_str][f"layer_{i+1}_after"].append(mag)
                    x = x_after

            # 4. Average the results over the seeds
            final_l_max_results = {}
            for ir_str, steps in seed_results.items():
                final_l_max_results[ir_str] = {
                    step: sum(mags) / len(mags) for step, mags in steps.items()
                }
            all_l_max_results[f"l_max_{l_max}"] = final_l_max_results
            print(
                f"    Finished l_max = {l_max}. Sample averaged magnitude for {list(final_l_max_results.keys())[0]} at layer_1_after: {list(final_l_max_results.values())[0]['layer_1_after']:.4f}"
            )

        # 5. Save YAML report
        report_path = OUTPUT_DIR / f"activation_magnitudes_{act_func_name}.yaml"
        with open(report_path, "w") as f:
            yaml.dump(all_l_max_results, f, sort_keys=False)
        print(f"\n  Saved report to {report_path}")

        # 6. Generate and save plot
        fig, axes = plt.subplots(
            1, len(L_MAX_VALUES), figsize=(6 * len(L_MAX_VALUES), 5), sharey=True
        )
        fig.suptitle(f"Activation Magnitude Flow: '{act_func_name}'")

        for i, l_max in enumerate(L_MAX_VALUES):
            ax = axes[i]
            results = all_l_max_results[f"l_max_{l_max}"]

            def sort_key(ir_str):
                match = re.match(r"(\d+x)?(\d+)([eo])", ir_str)
                l = int(match.group(2))
                p = match.group(3)
                return (l, p)

            sorted_irreps = sorted(results.keys(), key=sort_key)

            for ir_str in sorted_irreps:
                data = results[ir_str]
                before_mags = [data[f"layer_{j+1}_before"] for j in range(NUM_LAYERS)]
                after_mags = [data[f"layer_{j+1}_after"] for j in range(NUM_LAYERS)]

                x_axis = range(1, NUM_LAYERS + 1)
                (line,) = ax.plot(
                    x_axis,
                    before_mags,
                    marker="o",
                    linestyle="-",
                    label=f"{ir_str} (Before)",
                )
                ax.plot(
                    x_axis,
                    after_mags,
                    marker="x",
                    linestyle="--",
                    color=line.get_color(),
                    label=f"{ir_str} (After)",
                )

            ax.set_title(f"l_max = {l_max}")
            ax.set_xlabel("MLP Layer")
            ax.set_xticks(range(1, NUM_LAYERS + 1))
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        axes[0].set_ylabel("Average Activation Magnitude")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.15, 0.5))
        fig.tight_layout(rect=[0, 0, 0.9, 1])

        plot_path = OUTPUT_DIR / f"activation_magnitudes_{act_func_name}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"  Saved plot to {plot_path}\n")
        plt.close(fig)


if __name__ == "__main__":
    main()
