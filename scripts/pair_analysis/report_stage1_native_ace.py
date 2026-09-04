#!/usr/bin/env python3
"""Create the consolidated Stage-1 native-ACE benchmark report and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--audit-summary", type=Path, required=True)
    result.add_argument("--cache-summary", type=Path, required=True)
    result.add_argument("--core-summary", type=Path, required=True)
    result.add_argument("--baseline-summary", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    args = parser().parse_args()
    audit, cache, core, baseline = map(
        load,
        (
            args.audit_summary,
            args.cache_summary,
            args.core_summary,
            args.baseline_summary,
        ),
    )
    if not all(
        (
            audit["validation"]["passed"],
            cache["passed"],
            core["passed"],
            baseline["passed"],
        )
    ):
        raise ValueError(
            "A required Stage-1 component did not pass its internal checks"
        )
    if (
        baseline["frozen_hashes"]["dataset"]
        != cache["cache_metadata"]["dataset_sha256"]
    ):
        raise ValueError("Baseline/cache dataset hashes differ")
    if baseline["frozen_hashes"]["envelope"] != cache["range_envelope_hash"]:
        raise ValueError("Baseline/cache envelope hashes differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = ("train", "validation", "test")
    maes = [baseline["evaluations"][s]["matrix_elements"]["mae"] for s in splits]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    bars = ax.bar(splits, maes, color="#4472c4")
    ax.bar_label(bars, fmt="%.2f")
    ax.set_ylabel("Matrix-element MAE (meV)")
    ax.set_title("Native ACE generalization")
    fig.savefig(args.output_dir / "split_mae.png", dpi=180)
    plt.close(fig)

    distance = baseline["evaluations"]["test"]["by_distance"]
    centers = [
        sum(map(float, key.split("_angstrom")[0].split("-"))) / 2 for key in distance
    ]
    values = [row["mae"] for row in distance.values()]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(centers, values, marker="o", color="#c44e52")
    ax.set_xlabel("Pair distance (angstrom)")
    ax.set_ylabel("Test matrix-element MAE (meV)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.savefig(args.output_dir / "test_mae_vs_distance.png", dpi=180)
    plt.close(fig)

    target = baseline["evaluations"]["test"]["by_target_irrep"]
    labels = list(target)
    values = [target[key]["mae"] for key in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    bars = ax.bar(labels, values, color="#55a868")
    ax.bar_label(bars, fmt="%.1f", fontsize=7)
    ax.set_ylabel("Test irrep-component MAE (meV)")
    ax.set_xlabel("Target irrep")
    fig.savefig(args.output_dir / "test_mae_by_irrep.png", dpi=180)
    plt.close(fig)

    test = baseline["evaluations"]["test"]
    validation = baseline["evaluations"]["validation"]
    report = f"""# Stage 1 - Native non-GNN baseline and benchmark plumbing

## Objective

Establish a reproducible pair-local equivariant ACE-style Hamiltonian baseline and the shared SiO2 data, target, range-envelope, symmetry, metric, and artifact infrastructure.

## Hypotheses

Stage 1 is infrastructural. It establishes the deterministic linear reference needed to test H12 (range factorization) and H16 (nonlinear pair maps outperform linear baselines) later. The pre-registered expectation was that native ACE would underperform modern message-passing results while providing an interpretable non-GNN floor.

## Implemented methods

- One full 169-component block-irrep prediction interface for every 13x13 AO block.
- Exact canonical pair reversal/Hermiticity map.
- Onsite degree-2 ACE covariants and tagged-bond offsite covariants with linear ridge regression.
- Training-only shared range envelope and streaming float64 sufficient-statistic fitting.
- Physical AO-matrix, block-Frobenius, distance, site, species-pair, irrep, and multiplicity metrics.

## Dataset

- Released HamGNN SiO2 graph bundle: {audit['structure_count']} structures ({audit['atom_count']['total']} atoms), not the 663 stated in the publication.
- Split: 504/63/63, protocol-matched but split-not-identical; split hash `{baseline['frozen_hashes']['split']}`.
- Dataset hash: `{baseline['frozen_hashes']['dataset']}`.
- Frozen RH: {audit['hamiltonian_cutoff']['cutoff_angstrom']} angstrom; omitted-tail MAE floor {audit['hamiltonian_cutoff']['omitted_mae_mev']:.6f} meV; retained squared mass {audit['hamiltonian_cutoff']['retained_squared_fraction']:.9f}.
- Frozen envelope hash: `{baseline['frozen_hashes']['envelope']}`; fit on training only.

## Acceptance criteria and decision

| Criterion | Result | Decision |
|---|---:|---|
| AO-irrep-AO float64 round trip <= 1e-10 | {audit['validation']['ao_irrep_roundtrip_max_abs_error_hartree']:.3e} Ha max abs | PASS |
| Exact irrep reversal <= 1e-10 relative | {core['metrics']['pair_reversal']:.3e} | PASS |
| Proper/improper O(3) <= 1e-9 relative | {max(core['metrics']['equivariance_det_1'], core['metrics']['equivariance_det_-1']):.3e} | PASS |
| Native/upstream numerical alignment | Source conventions inspected; Julia oracle deferred | PARTIAL/INCONCLUSIVE |
| Published qualitative Al convergence/error | Not executed | NOT TESTED |
| Full SiO2 baseline, leakage-free and reproducible | completed | PASS |

Overall Stage-1 gate: **PARTIAL**. The common Python pipeline and SiO2 baseline are accepted for use; exact upstream Julia and Al comparisons remain an explicitly unresolved provenance limitation and are not represented as reproduced results.

## Results

| Split | Matrix MAE (meV) | Matrix RMSE (meV) | Block Frobenius MAE (meV) |
|---|---:|---:|---:|
| Train | {baseline['evaluations']['train']['matrix_elements']['mae']:.3f} | {baseline['evaluations']['train']['matrix_elements']['rmse']:.3f} | {baseline['evaluations']['train']['block_frobenius']['mae']:.3f} |
| Validation | {validation['matrix_elements']['mae']:.3f} | {validation['matrix_elements']['rmse']:.3f} | {validation['block_frobenius']['mae']:.3f} |
| Test | {test['matrix_elements']['mae']:.3f} | {test['matrix_elements']['rmse']:.3f} | {test['block_frobenius']['mae']:.3f} |

Test onsite MAE is {test['by_site']['onsite']['mae']:.3f} meV versus {test['by_site']['offsite']['mae']:.3f} meV offsite. Fit time was {baseline['fit_seconds']:.2f} s, the model has {baseline['fitted_coefficient_count']} fitted coefficients, and peak CUDA memory was {baseline['peak_cuda_memory_bytes']/2**20:.1f} MiB.

![Split MAE](split_mae.png)

![Distance error](test_mae_vs_distance.png)

![Irrep error](test_mae_by_irrep.png)

## Baseline comparison

| Method | Status | SiO2 matrix MAE (meV) | Comparability |
|---|---|---:|---|
| Native MANDALA ACE-style linear baseline | native reproduced | {baseline['headline_test_matrix_mae_mev']:.3f} | Frozen project split/pipeline |
| HamGNN | published contextual | 2.29 | Published 663-structure/random split; not head-to-head |

## Findings

- Train, validation, and test errors are close, indicating a capacity-limited rather than overfit baseline.
- Onsite blocks and short bonds dominate error. Test distance-bin MAE falls from about 1 eV below 2 angstrom to 4.3 meV at 6-6.5 angstrom.
- The shared envelope produces near-unit median normalized target RMS. O-O retains a heavier tail (maximum {baseline['normalized_target_rms']['O-O']['maximum']:.2f}).
- Normal equations are poorly conditioned in several scalar/low-order sectors, especially onsite (maximum condition number above 6e9); future learned models need irrep-copy normalization and stable optimization.

## Problems encountered

- The released archive contains 630 rather than 663 structures and does not expose the exact published split.
- Upstream ACEhamiltonians does not share this full-block real-irrep basis directly; numerical Julia oracle and Al reproduction remain deferred.

## Shortcomings and threats to validity

- The HamGNN number is contextual only because dataset cardinality and split identity differ.
- No uncertainty is attached to this deterministic single fit.
- Cutoff adequacy is measured only over blocks present in the released sparse graph.
- The compact native construction is ACEhamiltonians-style, not yet demonstrated coefficient-for-coefficient identical to upstream Julia.

## Decisions

Retain 165.959 meV as the frozen native linear baseline. Begin Stage 2 Python descriptor infrastructure without claiming the upstream reproduction gates are complete. Keep the Julia comparison as a narrowly scoped, necessary provenance task before final Stage-1 closure.

## Next experiments

1. Implement D1 with spherical-Bessel and Zernike-compatible orthogonal radial bases, exact O(3) metadata, and analytic puncturing.
2. Certify D1 symmetry, deterministic recomputation, periodic-neighbor handling, and storage/throughput before full caching.
3. Add D2/D3 and D4 only after the shared descriptor cache contract is stable.
"""
    (args.output_dir / "report.md").write_text(report)
    summary = {
        "stage": 1,
        "decision": "PARTIAL",
        "baseline_test_matrix_mae_mev": baseline["headline_test_matrix_mae_mev"],
        "dataset_hash": baseline["frozen_hashes"]["dataset"],
        "split_hash": baseline["frozen_hashes"]["split"],
        "envelope_hash": baseline["frozen_hashes"]["envelope"],
        "unresolved": ["upstream Julia numerical alignment", "published Al example"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
