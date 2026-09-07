#!/usr/bin/env python3
"""Validate one Stage-6 scoped run, including the numerical amendment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pair_hamiltonian.stage6_validation import assess_stage6_scoped_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    config = json.loads((args.run_dir / "config.json").read_text())
    assessment = assess_stage6_scoped_run(
        summary, float(config["float32_symmetry_tolerance"])
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    if not assessment["accepted"]:
        raise SystemExit("Stage-6 scoped run failed numerical certification")


if __name__ == "__main__":
    main()
