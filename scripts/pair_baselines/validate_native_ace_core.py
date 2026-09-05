#!/usr/bin/env python3
"""Validate native ACE symmetry, precision, coverage, and pair throughput."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from e3nn import o3
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_descriptors import AtomicNeighborDensity
from pair_hamiltonian.output_schema import (
    FullBlockIrrepTransform,
    o3_representation_matrix,
)
from pair_hamiltonian.hermiticity import project_directed_irreps
from pair_hamiltonian.sio2_cache import DIRECTED_PAIR_NAMES
from pair_mappers import NativeACEPairMapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--descriptor-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--density-radial-count", type=int, required=True)
    parser.add_argument("--density-l-max", type=int, required=True)
    parser.add_argument("--onsite-correlation-order", type=int, required=True)
    parser.add_argument("--onsite-max-degree", type=int, required=True)
    parser.add_argument("--bond-radial-count", type=int, required=True)
    parser.add_argument("--bond-l-max", type=int, required=True)
    parser.add_argument("--bond-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--offsite-max-degree", type=int, required=True)
    parser.add_argument("--ridge", type=float, required=True)
    parser.add_argument("--pair-batch-size", type=int, required=True)
    parser.add_argument("--warmup-iterations", type=int, required=True)
    parser.add_argument("--benchmark-iterations", type=int, required=True)
    return parser


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_state() -> dict[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def action(irreps: o3.Irreps, rotation: torch.Tensor) -> torch.Tensor:
    return o3_representation_matrix(irreps, rotation)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected).clamp_min(1e-30)
        ).item()
    )


def build_components(args, dtype: torch.dtype, device: torch.device):
    orbital_config = OrbitalIrrepConfig.from_dict({"O": "2s2p1d", "Si": "2s2p1d"})
    transform = FullBlockIrrepTransform(orbital_config, dtype=dtype, device=device)
    density = AtomicNeighborDensity(
        (8, 14),
        n_radial=args.density_radial_count,
        l_max=args.density_l_max,
        cutoff=args.descriptor_cutoff_angstrom,
    ).to(device=device, dtype=dtype)
    model = NativeACEPairMapper(
        transform,
        density.layout,
        onsite_correlation_order=args.onsite_correlation_order,
        onsite_max_degree=args.onsite_max_degree,
        bond_n_radial=args.bond_radial_count,
        bond_l_max=args.bond_l_max,
        bond_cutoff=args.bond_cutoff_angstrom,
        offsite_max_degree=args.offsite_max_degree,
        ridge=args.ridge,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    return transform, density, model


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    for name in (
        "density_radial_count",
        "onsite_correlation_order",
        "onsite_max_degree",
        "bond_radial_count",
        "offsite_max_degree",
        "pair_batch_size",
        "benchmark_iterations",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    output_dir = args.output_dir.resolve()
    existing = (
        []
        if not output_dir.exists()
        else [path for path in output_dir.iterdir() if path.name != "run.log"]
    )
    if existing:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = {
        **vars(args),
        "output_dir": str(output_dir),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "e3nn": __import__("e3nn").__version__,
            "numpy": np.__version__,
        },
        "hardware": {
            "device": str(device),
            "name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
        "git": git_state(),
    }
    atomic_json(output_dir / "config.json", config)
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)
    print("[1/5] Constructing float64 reference", flush=True)
    transform, density, model = build_components(args, torch.float64, device)
    for name, buffer in model.named_buffers():
        if "weight_" in name:
            buffer.copy_(torch.randn_like(buffer))
    target_labels = {copy.irrep_label for copy in transform.schema("O-Si").copies}
    onsite_labels = {str(irrep) for _mul, irrep in model.onsite_basis.irreps_out}
    offsite_labels = {str(irrep) for _mul, irrep in model.offsite_basis.irreps_out}
    missing_onsite = sorted(target_labels - onsite_labels)
    missing_offsite = sorted(target_labels - offsite_labels)
    if missing_onsite or missing_offsite:
        raise RuntimeError(
            f"ACE feature coverage incomplete: onsite={missing_onsite}, offsite={missing_offsite}"
        )

    batch = args.pair_batch_size
    descriptor_i = torch.randn(
        batch, density.irreps_out.dim, dtype=torch.float64, device=device
    )
    descriptor_j = torch.randn_like(descriptor_i)
    displacement = torch.randn(batch, 3, dtype=torch.float64, device=device)
    displacement[:, 0] += 2.0
    print("[2/5] Proper/improper O(3) and reversal checks", flush=True)
    metrics: list[dict[str, object]] = []
    base = model.predict_offsite(("O", "Si"), descriptor_i, displacement, descriptor_j)
    for determinant in (1, -1):
        rotation = o3.rand_matrix(dtype=torch.float64, device=device)
        if determinant == -1:
            rotation = -rotation
        descriptor_action = action(density.irreps_out, rotation)
        output_action = transform.output_action("O-Si", rotation)
        actual = model.predict_offsite(
            ("O", "Si"),
            descriptor_i @ descriptor_action.T,
            displacement @ rotation.T,
            descriptor_j @ descriptor_action.T,
        )
        error = relative_error(actual, base @ output_action.T)
        metrics.append(
            {
                "metric": f"equivariance_det_{determinant}",
                "value": error,
                "unit": "relative",
            }
        )
    reverse = model.predict_offsite(
        ("Si", "O"), descriptor_j, -displacement, descriptor_i
    )
    reversal_error = relative_error(reverse, transform.reverse("O-Si", base))
    metrics.append(
        {"metric": "raw_pair_reversal", "value": reversal_error, "unit": "relative"}
    )
    raw = torch.cat((base, reverse), dim=0)
    inverse = torch.cat(
        (
            torch.arange(batch, 2 * batch, device=device),
            torch.arange(batch, device=device),
        )
    )
    pair_types = torch.cat(
        (
            torch.full((batch,), 1, dtype=torch.long, device=device),
            torch.full((batch,), 2, dtype=torch.long, device=device),
        )
    )
    projected = project_directed_irreps(
        transform, DIRECTED_PAIR_NAMES, raw, pair_types, inverse
    )
    projected_reversal_error = relative_error(
        projected[batch:], transform.reverse("O-Si", projected[:batch])
    )
    metrics.append(
        {
            "metric": "projected_pair_reversal",
            "value": projected_reversal_error,
            "unit": "relative",
        }
    )

    print("[3/5] Float32 agreement and stability", flush=True)
    transform32, density32, model32 = build_components(args, torch.float32, device)
    model32.load_state_dict(model.state_dict())
    prediction32 = model32.predict_offsite(
        ("O", "Si"), descriptor_i.float(), displacement.float(), descriptor_j.float()
    ).double()
    float32_error = relative_error(prediction32, base)
    metrics.append(
        {"metric": "float32_vs_float64", "value": float32_error, "unit": "relative"}
    )
    if not torch.isfinite(prediction32).all():
        raise RuntimeError("Non-finite float32 prediction")

    print("[4/5] Measuring offsite feature throughput", flush=True)
    for _ in range(args.warmup_iterations):
        model32.offsite_features(
            descriptor_i.float(), displacement.float(), descriptor_j.float()
        )
    synchronize(device)
    started = time.perf_counter()
    for _ in range(args.benchmark_iterations):
        model32.offsite_features(
            descriptor_i.float(), displacement.float(), descriptor_j.float()
        )
    synchronize(device)
    elapsed = time.perf_counter() - started
    pairs_per_second = batch * args.benchmark_iterations / elapsed
    metrics.append(
        {
            "metric": "offsite_feature_throughput",
            "value": pairs_per_second,
            "unit": "pairs_per_second",
        }
    )
    metrics.append(
        {
            "metric": "offsite_feature_dimension",
            "value": model.offsite_basis.irreps_out.dim,
            "unit": "components",
        }
    )
    metrics.append(
        {
            "metric": "onsite_feature_dimension",
            "value": model.onsite_basis.irreps_out.dim,
            "unit": "components",
        }
    )
    metrics.append(
        {
            "metric": "target_dimension",
            "value": transform.irreps("O-Si").dim,
            "unit": "components",
        }
    )
    tolerances = {
        "equivariance": 1e-9,
        "reversal": 1e-10,
        "float32_vs_float64": 2e-5,
    }
    metric_values = {row["metric"]: float(row["value"]) for row in metrics}
    passed = (
        metric_values["equivariance_det_1"] <= tolerances["equivariance"]
        and metric_values["equivariance_det_-1"] <= tolerances["equivariance"]
        and metric_values["projected_pair_reversal"] <= tolerances["reversal"]
        and metric_values["float32_vs_float64"] <= tolerances["float32_vs_float64"]
    )
    summary = {
        "passed": passed,
        "tolerances": tolerances,
        "metrics": metric_values,
        "target_irrep_labels": sorted(target_labels),
        "onsite_feature_layout_hash": model.onsite_basis.layout.content_hash,
        "offsite_feature_layout_hash": model.offsite_basis.layout.content_hash,
        "upstream_reference": {
            "repository": "ACEsuit/ACEhamiltonians.jl",
            "commit": "9ff15724583f9d2115d8bf48562a47fb94f45b40",
            "comparison_status": "source conventions inspected; numerical Julia oracle deferred",
        },
    }
    print("[5/5] Writing artifacts", flush=True)
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value", "unit"))
        writer.writeheader()
        writer.writerows(metrics)
    atomic_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        "# Native ACE core validation\n\n"
        f"Acceptance: **{'PASS' if passed else 'FAIL'}**\n\n"
        f"Offsite feature throughput: {pairs_per_second:.3f} pairs/s on {config['hardware']['name']}.\n\n"
        "This validates the native fixed ACE algebra, independent directed raw calls, and evaluation-only Hermitian projection. Raw reversal error is diagnostic and is not constrained. It is not an upstream Julia basis-value comparison or a fitted SiO2 accuracy result.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError("Native ACE core acceptance gate failed")


if __name__ == "__main__":
    main()
