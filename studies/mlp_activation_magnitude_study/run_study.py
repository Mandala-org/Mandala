import sys
from pathlib import Path
import collections
import argparse
import re

import torch
import yaml
import matplotlib.pyplot as plt
from e3nn.o3 import Irreps, Linear
from torch import nn

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from net.common import Config, build_hidden_irreps  # noqa: E402
from net.activations import make_nonlinearity  # noqa: E402


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="MLP Activation Magnitude Study")
    parser.add_argument("--hidden_base_dim", type=int, default=32)
    parser.add_argument("--max_l_max", type=int, default=4)
    parser.add_argument("--min_l_max", type=int, default=2)
    parser.add_argument("--plot_log_scale", action="store_true")
    parser.add_argument("--inputs_sigma", type=float, default=1.0)
    parser.add_argument("--layer_weights_mult", type=float, default=1.0)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--separate_folder", action="store_true")
    return parser.parse_args()


def get_average_magnitudes(features: torch.Tensor, irreps: Irreps) -> dict[str, float]:
    """Computes the average magnitude for each irrep in a feature tensor."""
    mags = {}
    current_dim = 0
    for mul, ir in irreps:
        slice_dim = mul * ir.dim
        if slice_dim == 0:
            continue
        act_slice = features[:, current_dim : current_dim + slice_dim]
        reshaped_slice = act_slice.reshape(-1, ir.dim)
        if reshaped_slice.shape[0] > 0:
            magnitude = torch.linalg.norm(reshaped_slice, dim=1).mean().item()
            mags[str(ir)] = magnitude
        current_dim += slice_dim
    return mags


def main():
    """Runs the MLP activation magnitude study."""
    args = parse_args()

    # --- Study Configuration ---
    L_MAX_VALUES = range(args.min_l_max, args.max_l_max + 1)
    GATE_ACTIVATIONS = ["leakyrelu", "sigmoid", "tanh"]

    activation_configs = [
        {"nonlin_kind": "normact"},
        # s2act is skipped because it requires all multiplicities to be 1.
    ]
    for gate_act in GATE_ACTIVATIONS:
        activation_configs.append(
            {"nonlin_kind": "gate_scalars_mlp", "activation_gate": gate_act}
        )
        activation_configs.append(
            {"nonlin_kind": "gate_magnitudes", "activation_gate": gate_act}
        )

    # --- Output Directory Setup ---
    base_output_dir = Path(__file__).parent
    if args.separate_folder:
        folder_name = f"hbd{args.hidden_base_dim}_sigma{args.inputs_sigma}_wmult{args.layer_weights_mult}"
        output_dir = base_output_dir / folder_name
    else:
        output_dir = base_output_dir
    output_dir.mkdir(exist_ok=True)
    print(f"Saving results to: {output_dir}")

    for act_config in activation_configs:
        act_func_name = act_config["nonlin_kind"]
        if "activation_gate" in act_config:
            act_func_name += f"_{act_config['activation_gate']}"

        print(f"--- Running study for activation: {act_func_name} ---")
        all_l_max_results = {}

        for l_max in L_MAX_VALUES:
            hidden_irreps = build_hidden_irreps(l_max, args.hidden_base_dim)
            print(f"\n  l_max = {l_max}, hidden_irreps = {hidden_irreps}")

            seed_results = collections.defaultdict(
                lambda: collections.defaultdict(list)
            )

            for seed in range(args.num_seeds):
                print(f"    Running seed {seed+1}/{args.num_seeds}")
                torch.manual_seed(seed)

                # 1. Build the MLP
                mlp_layers = []
                cfg = Config(**act_config)

                for _ in range(args.num_layers):
                    lin = Linear(hidden_irreps, hidden_irreps)
                    with torch.no_grad():
                        lin.weight.data *= args.layer_weights_mult
                    act = make_nonlinearity(hidden_irreps, cfg)
                    mlp_layers.append(nn.ModuleList([lin, act]))
                mlp = nn.ModuleList(mlp_layers)

                # 2. Create random input
                x = torch.randn(1, hidden_irreps.dim) * args.inputs_sigma

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
            if final_l_max_results:
                sample_ir = list(final_l_max_results.keys())[0]
                sample_mag = list(final_l_max_results[sample_ir].values())[
                    1
                ]  # layer_1_after
                print(
                    f"    Finished l_max = {l_max}. Sample avg mag for {sample_ir} at layer_1_after: {sample_mag:.4f}"
                )

        # 5. Save YAML report
        report_path = output_dir / f"activation_magnitudes_{act_func_name}.yaml"
        with open(report_path, "w") as f:
            yaml.dump(all_l_max_results, f, sort_keys=False)
        print(f"\n  Saved report to {report_path}")

        # 6. Generate and save plot
        fig, axes = plt.subplots(
            1, len(L_MAX_VALUES), figsize=(7 * len(L_MAX_VALUES), 6), sharey=True
        )
        fig.suptitle(f"Activation Magnitude Flow: '{act_func_name}'")

        for i, l_max in enumerate(L_MAX_VALUES):
            ax = axes[i] if len(L_MAX_VALUES) > 1 else axes
            results = all_l_max_results.get(f"l_max_{l_max}", {})

            def sort_key(ir_str):
                match = re.match(r"(\d+x)?(\d+)([eo])", ir_str)
                l = int(match.group(2))
                p = match.group(3)
                return (l, p)

            sorted_irreps = sorted(results.keys(), key=sort_key)

            for ir_str in sorted_irreps:
                data = results[ir_str]
                before_mags = [
                    data.get(f"layer_{j+1}_before", 0) for j in range(args.num_layers)
                ]
                after_mags = [
                    data.get(f"layer_{j+1}_after", 0) for j in range(args.num_layers)
                ]

                x_axis = range(1, args.num_layers + 1)
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
            ax.set_xticks(range(1, args.num_layers + 1))
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            if args.plot_log_scale:
                ax.set_yscale("log")

        (axes[0] if len(L_MAX_VALUES) > 1 else axes).set_ylabel(
            "Average Activation Magnitude"
        )
        handles, labels = (
            axes[-1] if len(L_MAX_VALUES) > 1 else axes
        ).get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(rect=[0, 0, 0.9, 1])

        plot_path = output_dir / f"activation_magnitudes_{act_func_name}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"  Saved plot to {plot_path}\n")
        plt.close(fig)


if __name__ == "__main__":
    main()
