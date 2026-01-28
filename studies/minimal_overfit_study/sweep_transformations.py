#!/usr/bin/env python3
"""
Grid search over transformation variants.

This script launches multiple training runs with different transformation
configurations to systematically identify the correct data loading pipeline.

Focus: 1e irrep (axial vectors) loss as the primary validation metric.
"""

import subprocess
import itertools

# Define transformation variants to explore
VARIANTS = {
    "coord_permutation": ["2,0,1", "1,2,0", "none"],
    "box_permutation": ["same", "none"],
    "sh_transform_mode": ["standard", "inverse", "none"],
    "sh_l1_override": [None, "2,0,1", "1,2,0", "none"],
    "mirror_x": [False, True],
}

# Training hyperparameters (fixed)
TRAIN_ARGS = {
    "lr": 3e-3,
    "num_layers": 1,
    "num_epochs": 500,  # Shorter for exploration
    "log_interval": 100,
    "cutoff_radius": 8.0,
    "n_radial": 64,
    "l_max": 4,
    "hidden_irreps": "16x0e+8x1o+8x1e+4x2e+4x2o+4x3o+4x3e+2x4e",
    "train_on_irrep_parts": True,  # Required to track 1e loss
}


def build_command(variant_config):
    """Build command line for a specific variant."""
    cmd = [
        "python",
        "studies/minimal_overfit_study/overfit_water_transform_exploration.py",
    ]

    # Add fixed training args
    for key, value in TRAIN_ARGS.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key.replace('_', '-')}")
        else:
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))

    # Add variant-specific args
    for key, value in variant_config.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key.replace('_', '-')}")
        else:
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))

    return cmd


def run_sweep(max_runs=None, dry_run=False):
    """
    Run grid search over all transformation variants.

    Args:
        max_runs: Maximum number of runs (None = all)
        dry_run: If True, only print commands without running
    """
    # Generate all combinations
    keys = list(VARIANTS.keys())
    values = [VARIANTS[k] for k in keys]

    all_combinations = list(itertools.product(*values))

    print(f"Total combinations: {len(all_combinations)}")

    if max_runs:
        all_combinations = all_combinations[:max_runs]
        print(f"Running first {max_runs} combinations")

    # Run each combination
    for i, combo in enumerate(all_combinations, 1):
        config = dict(zip(keys, combo))

        # Skip invalid combinations
        # (e.g., if box_permutation="same", it will follow coord_permutation)
        if config["box_permutation"] == "same":
            config["box_permutation"] = (
                None  # None means "same as coords" in the script
            )

        cmd = build_command(config)

        print(f"\n{'='*80}")
        print(f"Run {i}/{len(all_combinations)}")
        print(f"Config: {config}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*80}")

        if dry_run:
            continue

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Run {i} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Run {i} failed with error: {e}")
            # Continue with next run
        except KeyboardInterrupt:
            print(f"\n\nSweep interrupted by user at run {i}")
            break


def run_minimal_sweep(dry_run=False):
    """
    Run a minimal sweep focusing on the most likely variants.

    This tests:
    - All 3 coord permutations
    - All 3 SH modes
    - All 4 l=1 overrides (including None)
    - No mirroring
    - Box follows coords

    Total: 3 * 3 * 4 = 36 runs
    """
    minimal_variants = {
        "coord_permutation": ["2,0,1", "1,2,0", "none"],
        "box_permutation": [None],  # Always follow coords
        "sh_transform_mode": ["standard", "inverse", "none"],
        "sh_l1_override": [None, "2,0,1", "1,2,0", "none"],
        "mirror_x": [False],  # No mirroring
    }

    keys = list(minimal_variants.keys())
    values = [minimal_variants[k] for k in keys]

    all_combinations = list(itertools.product(*values))

    print(f"Minimal sweep: {len(all_combinations)} combinations")

    for i, combo in enumerate(all_combinations, 1):
        config = dict(zip(keys, combo))
        cmd = build_command(config)

        print(f"\n{'='*80}")
        print(f"Run {i}/{len(all_combinations)}")
        print(f"Config: {config}")
        print(f"{'='*80}")

        if dry_run:
            print(f"Command: {' '.join(cmd)}")
            continue

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Run {i} completed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Run {i} failed: {e}")
        except KeyboardInterrupt:
            print(f"\n\nSweep interrupted at run {i}")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run transformation exploration sweep")
    parser.add_argument(
        "--mode",
        type=str,
        default="minimal",
        choices=["minimal", "full"],
        help="Sweep mode: 'minimal' (36 runs) or 'full' (all combinations)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum number of runs (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running",
    )

    args = parser.parse_args()

    if args.mode == "minimal":
        run_minimal_sweep(dry_run=args.dry_run)
    else:
        run_sweep(max_runs=args.max_runs, dry_run=args.dry_run)
