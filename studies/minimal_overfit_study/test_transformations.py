#!/usr/bin/env python3
"""
Quick test of transformation exploration.

Test a few specific configurations to verify the pipeline works.
"""

import subprocess
import sys

TEST_CONFIGS = [
    {
        "name": "standard_current",
        "args": [
            "--coord-permutation",
            "2,0,1",
            "--sh-transform-mode",
            "standard",
            "--num-epochs",
            "50",
        ],
        "description": "Current standard configuration",
    },
    {
        "name": "inverse_permutation",
        "args": [
            "--coord-permutation",
            "1,2,0",
            "--sh-transform-mode",
            "standard",
            "--num-epochs",
            "50",
        ],
        "description": "Inverse coordinate permutation",
    },
    {
        "name": "no_transform",
        "args": [
            "--coord-permutation",
            "none",
            "--sh-transform-mode",
            "none",
            "--num-epochs",
            "50",
        ],
        "description": "No transformations applied",
    },
    {
        "name": "l1_override",
        "args": [
            "--coord-permutation",
            "2,0,1",
            "--sh-transform-mode",
            "standard",
            "--sh-l1-override",
            "1,2,0",
            "--num-epochs",
            "50",
        ],
        "description": "Override l=1 with different permutation",
    },
]

BASE_ARGS = [
    "python",
    "studies/minimal_overfit_study/overfit_water_transform_exploration.py",
    "--lr",
    "3e-3",
    "--num-layers",
    "1",
    "--log-interval",
    "25",
    "--cutoff-radius",
    "8.0",
    "--n-radial",
    "64",
    "--l-max",
    "4",
    "--hidden-irreps",
    "16x0e+8x1o+8x1e+4x2e+4x2o+4x3o+4x3e+2x4e",
    "--train-on-irrep-parts",
]


def run_test(config):
    """Run a single test configuration."""
    cmd = BASE_ARGS + config["args"]

    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Test '{config['name']}' passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Test '{config['name']}' failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n\nTest interrupted by user")
        return False


if __name__ == "__main__":
    print("Running transformation exploration tests...")
    print(f"Total tests: {len(TEST_CONFIGS)}")

    passed = 0
    failed = 0

    for config in TEST_CONFIGS:
        if run_test(config):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*80}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*80}")

    sys.exit(0 if failed == 0 else 1)
