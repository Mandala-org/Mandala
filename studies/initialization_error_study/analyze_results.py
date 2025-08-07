"""
Analyzes the results of the initialization error study, filtering outliers.

This script loads the 'results.csv' file, removes records with empty
energy_error values, then removes a fixed number of the highest energy
error values, and finally generates the same set of plots as the original
run_study.py script, saving them to a new 'plots_filtered' directory.
"""

import sys
from pathlib import Path
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to the Python path to import SEARCH_SPACE
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the search space from the original study to replicate plots
from studies.initialization_error_study.run_study import SEARCH_SPACE  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot initialization error study results after filtering outliers."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(__file__).parent / "results.csv",
        help="Path to the results.csv file to analyze.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "plots_filtered",
        help="Directory to save the filtered plots.",
    )
    parser.add_argument(
        "--trim_top",
        type=int,
        default=10,
        help="Number of records to trim from the top of the energy_error distribution.",
    )
    args = parser.parse_args()

    # 1. Load the data
    if not args.input_csv.exists():
        print(f"Error: Input file not found at {args.input_csv}")
        sys.exit(1)

    print(f"Loading results from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    initial_rows = len(df)
    print(f"Loaded {initial_rows} data points.")

    # 2a. Remove records with empty energy error
    df.dropna(subset=["energy_error"], inplace=True)
    rows_after_na = len(df)
    if rows_after_na < initial_rows:
        print(
            f"Removed {initial_rows - rows_after_na} records with empty energy_error."
        )

    # 2b. Discard outliers by removing a fixed number of top records
    if rows_after_na < args.trim_top:
        print(
            f"Error: Cannot trim {args.trim_top} records, as there are only {rows_after_na} total records after removing NAs."
        )
        sys.exit(1)

    df_sorted = df.sort_values("energy_error", ascending=True)
    df_filtered = df_sorted.iloc[: -args.trim_top]

    filtered_rows = len(df_filtered)
    print(f"Filtering the {args.trim_top} highest energy error values...")
    print(
        f"Kept {filtered_rows} data points ({filtered_rows / initial_rows:.2%} of original)."
    )

    # 3. Create plots using the filtered data
    args.output_dir.mkdir(exist_ok=True)
    print(f"Generating {len(SEARCH_SPACE)} new plots in {args.output_dir}...")

    for param in SEARCH_SPACE.keys():
        plt.figure(figsize=(10, 6))

        # Check if the parameter is numerical or categorical from SEARCH_SPACE
        if "distribution" in SEARCH_SPACE[param]:  # Numerical
            sns.regplot(
                data=df_filtered, x=param, y="energy_error", scatter_kws={"alpha": 0.5}
            )
        else:  # Categorical
            sns.stripplot(
                data=df_filtered, x=param, y="energy_error", alpha=0.5, jitter=True
            )
            sns.boxplot(
                data=df_filtered, x=param, y="energy_error", color="white", fliersize=0
            )

        plt.title(f"Energy Error vs. {param} (Outliers Removed)")
        plt.ylabel("Residual Energy Error (Predicted - True)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = args.output_dir / f"{param}_vs_error_filtered.png"
        plt.savefig(plot_path)
        plt.close()

    print("All filtered plots have been generated successfully.")


if __name__ == "__main__":
    main()
