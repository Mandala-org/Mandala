#!/usr/bin/env python3
"""Run the pre-registered Stage-4 correctness and microbenchmark suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

from e3nn import o3
import numpy as np
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from pair_descriptors.ace_covariants import (
    AtomicNeighborDensity,
    EquivariantFeatureLayout,
)
from pair_hamiltonian.output_schema import (
    FullBlockIrrepTransform,
    o3_representation_matrix,
)
from pair_mappers import FullBlockNeuralPairMapper, NativeACEPairMapper
from pair_mappers.neural import BondFramePairKernel


ARCHITECTURES = ("m0", "m1", "m2", "m3", "m5", "m7")
DESCRIPTOR_IRREPS = o3.Irreps("0e + 1o + 2e + 0e + 1o + 2e")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architectures", nargs="+", choices=ARCHITECTURES, required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--fit-samples", type=int, default=384)
    parser.add_argument("--fit-validation-samples", type=int, default=128)
    parser.add_argument("--fit-steps", type=int, default=800)
    parser.add_argument("--fit-learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--benchmark-batch-size", type=int, default=256)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--equivariance-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--reversal-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--synthetic-fit-relative-rms", type=float, default=5.0e-2)
    return parser.parse_args()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(1.0e-15)
    return float((torch.linalg.vector_norm(actual - expected) / denominator).item())


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.mean(expected.square()).sqrt().clamp_min(1.0e-15)
    return float((torch.mean((actual - expected).square()).sqrt() / denominator).item())


def _make_transform(dtype: torch.dtype) -> FullBlockIrrepTransform:
    config = OrbitalIrrepConfig.from_dict({"A": "1s1p", "B": "1s1p"})
    return FullBlockIrrepTransform(config, dtype=dtype)


def _density_layout() -> EquivariantFeatureLayout:
    density = AtomicNeighborDensity((8, 14), n_radial=1, l_max=2, cutoff=5.0)
    assert density.irreps_out == DESCRIPTOR_IRREPS
    return density.layout


def _make_mapper(
    architecture: str, dtype: torch.dtype, device: torch.device
) -> torch.nn.Module:
    transform = _make_transform(dtype)
    if architecture == "m1":
        model: torch.nn.Module = NativeACEPairMapper(
            transform,
            _density_layout(),
            onsite_correlation_order=2,
            onsite_max_degree=4,
            bond_n_radial=2,
            bond_l_max=2,
            bond_cutoff=6.0,
            offsite_max_degree=5,
            ridge=1.0e-10,
            dtype=dtype,
        )
    else:
        model = FullBlockNeuralPairMapper(
            architecture,
            transform,
            DESCRIPTOR_IRREPS,
            bond_n_radial=2,
            bond_l_max=2,
            bond_cutoff=6.0,
            hidden_multiplicity=2,
            hidden_l_max=3,
            invariant_hidden=32,
            factorization_rank=4,
            dtype=dtype,
        )
    return model.to(device=device, dtype=dtype)


def _random_inputs(
    count: int, *, dtype: torch.dtype, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    descriptor_i = torch.randn(
        count, DESCRIPTOR_IRREPS.dim, dtype=dtype, device=device, generator=generator
    )
    descriptor_j = torch.randn(
        count, DESCRIPTOR_IRREPS.dim, dtype=dtype, device=device, generator=generator
    )
    displacement = torch.randn(
        count, 3, dtype=dtype, device=device, generator=generator
    )
    displacement = displacement / torch.linalg.vector_norm(
        displacement, dim=-1, keepdim=True
    )
    radii = 1.0 + 3.5 * torch.rand(
        count, 1, dtype=dtype, device=device, generator=generator
    )
    return descriptor_i, descriptor_j, displacement * radii


def _predict(
    model: torch.nn.Module,
    descriptor_i: torch.Tensor,
    descriptor_j: torch.Tensor,
    displacement: torch.Tensor,
) -> torch.Tensor:
    return model.predict_offsite(("A", "B"), descriptor_i, displacement, descriptor_j)


def _symmetry_metrics(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float | bool]:
    descriptor_i, descriptor_j, displacement = inputs
    transform = model.target_transform
    with torch.no_grad():
        reference = _predict(model, descriptor_i, descriptor_j, displacement)
        errors: dict[str, float | bool] = {}
        for label, determinant in (("proper", 1), ("improper", -1)):
            rotation = o3.rand_matrix(dtype=descriptor_i.dtype).to(descriptor_i.device)
            if determinant == -1:
                rotation = -rotation
            descriptor_action = o3_representation_matrix(DESCRIPTOR_IRREPS, rotation)
            target_action = transform.output_action(("A", "B"), rotation)
            actual = _predict(
                model,
                descriptor_i @ descriptor_action.T,
                descriptor_j @ descriptor_action.T,
                displacement @ rotation.T,
            )
            errors[f"{label}_o3_relative_error"] = _relative_error(
                actual, reference @ target_action.T
            )

        reverse = model.predict_offsite(
            ("B", "A"), descriptor_j, -displacement, descriptor_i
        )
        errors["heterogeneous_reversal_relative_error"] = _relative_error(
            reverse, transform.reverse(("A", "B"), reference)
        )
        same = model.predict_offsite(
            ("A", "A"), descriptor_i, displacement, descriptor_j
        )
        same_reverse = model.predict_offsite(
            ("A", "A"), descriptor_j, -displacement, descriptor_i
        )
        errors["homogeneous_reversal_relative_error"] = _relative_error(
            same_reverse, transform.reverse(("A", "A"), same)
        )

        onsite = model.predict_onsite("A", descriptor_i)
        rotation = -o3.rand_matrix(dtype=descriptor_i.dtype).to(descriptor_i.device)
        descriptor_action = o3_representation_matrix(DESCRIPTOR_IRREPS, rotation)
        target_action = transform.output_action(("A", "A"), rotation)
        rotated_onsite = model.predict_onsite("A", descriptor_i @ descriptor_action.T)
        errors["onsite_improper_o3_relative_error"] = _relative_error(
            rotated_onsite, onsite @ target_action.T
        )
        errors["onsite_hermiticity_relative_error"] = _relative_error(
            onsite, transform.reverse(("A", "A"), onsite)
        )

        first = model(
            descriptor_i,
            descriptor_j,
            displacement,
            "A",
            "B",
            {"onsite": False, "irrelevant_graph": torch.randn(19)},
        )
        second = model(
            descriptor_i,
            descriptor_j,
            displacement,
            "A",
            "B",
            {"onsite": False, "irrelevant_graph": object()},
        )
        errors["irrelevant_metadata_bitwise"] = bool(torch.equal(first, second))
        errors["output_dimension_exact"] = bool(
            first.shape[-1] == transform.schema("A-B").vector_dimension
        )
    return errors


def _randomize_ace_teacher(model: NativeACEPairMapper) -> None:
    with torch.no_grad():
        for name, value in model.named_buffers():
            if "weight_" in name:
                value.copy_(0.25 * torch.randn_like(value))


def _make_known_teacher(architecture: str, teacher: torch.nn.Module) -> None:
    """Restrict a teacher to a frozen, low-complexity equivariant polynomial."""
    if architecture == "m1":
        assert isinstance(teacher, NativeACEPairMapper)
        _randomize_ace_teacher(teacher)
        return
    assert isinstance(teacher, FullBlockNeuralPairMapper)
    kernel = teacher.offsite_kernels["A__B"]
    with torch.no_grad():
        for parameter in kernel.parameters():
            parameter.zero_()
        if architecture == "m0":
            kernel.linear.weight.normal_(mean=0.0, std=0.25)
        elif architecture == "m2":
            kernel.pre.weight.normal_(mean=0.0, std=0.25)
            kernel.output.weight.normal_(mean=0.0, std=0.25)
        elif architecture == "m3":
            kernel.coefficients[-1].bias[
                : min(8, kernel.tensor_product.weight_numel)
            ] = 0.25
        elif architecture == "m5":
            kernel.coefficients[-1].bias[0] = 1.0
            kernel.weight_basis[0, : min(8, kernel.tensor_product.weight_numel)] = 0.25
        elif architecture == "m7":
            local = kernel.local_kernel
            local.coefficients[-1].bias[
                : min(8, local.tensor_product.weight_numel)
            ] = 0.25
        else:  # pragma: no cover - argparse constrains this
            raise AssertionError(architecture)


def _prepare_known_student(
    architecture: str, student: torch.nn.Module, teacher: torch.nn.Module
) -> list[torch.nn.Parameter]:
    """Expose only the coefficients needed for the declared synthetic target."""
    if architecture == "m1":
        return []
    assert isinstance(student, FullBlockNeuralPairMapper)
    assert isinstance(teacher, FullBlockNeuralPairMapper)
    kernel = student.offsite_kernels["A__B"]
    teacher_kernel = teacher.offsite_kernels["A__B"]
    for parameter in kernel.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        if architecture == "m0":
            kernel.linear.weight.zero_()
            parameter = kernel.linear.weight
        elif architecture == "m2":
            kernel.load_state_dict(teacher_kernel.state_dict())
            kernel.output.weight.zero_()
            parameter = kernel.output.weight
        elif architecture == "m3":
            for value in kernel.parameters():
                value.zero_()
            parameter = kernel.coefficients[-1].bias
        elif architecture == "m5":
            for value in kernel.parameters():
                value.zero_()
            kernel.weight_basis.copy_(teacher_kernel.weight_basis)
            parameter = kernel.coefficients[-1].bias
        elif architecture == "m7":
            local = kernel.local_kernel
            for value in local.parameters():
                value.zero_()
            parameter = local.coefficients[-1].bias
        else:  # pragma: no cover - argparse constrains this
            raise AssertionError(architecture)
    parameter.requires_grad_(True)
    return [parameter]


def _synthetic_fit(
    architecture: str,
    student: torch.nn.Module,
    train_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    teacher = _make_mapper(
        architecture, next(student.buffers()).dtype, next(student.buffers()).device
    )
    _make_known_teacher(architecture, teacher)
    if architecture == "m1":
        assert isinstance(student, NativeACEPairMapper)
    teacher.eval()
    local_training = None
    with torch.no_grad():
        train_target = _predict(teacher, *train_inputs)
        validation_target = _predict(teacher, *validation_inputs)
        initial = _relative_rms(
            _predict(student, *validation_inputs), validation_target
        )
        if architecture == "m7":
            assert isinstance(student, FullBlockNeuralPairMapper)
            assert isinstance(teacher, FullBlockNeuralPairMapper)
            student_kernel = student.offsite_kernels["A__B"]
            teacher_kernel = teacher.offsite_kernels["A__B"]
            assert isinstance(student_kernel, BondFramePairKernel)
            assert isinstance(teacher_kernel, BondFramePairKernel)
            train_local = student_kernel.local_coordinates(
                train_inputs[0],
                train_inputs[2],
                train_inputs[1],
                student.bond_expansion,
                roll=None,
            )[:3]
            local_target = teacher_kernel.local_kernel(*train_local)
            local_training = (student_kernel, train_local, local_target)

    if architecture == "m1":
        student.fit_offsite(
            ("A", "B"),
            train_inputs[0],
            train_inputs[2],
            train_inputs[1],
            train_target,
        )
    else:
        parameters = _prepare_known_student(architecture, student, teacher)
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            if local_training is None:
                prediction = _predict(student, *train_inputs)
                target = train_target
            else:
                student_kernel, train_local, target = local_training
                prediction = student_kernel.local_kernel(*train_local)
            scale = torch.mean(target.square()).detach().clamp_min(1.0e-12)
            loss = torch.mean((prediction - target).square()) / scale
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        train_error = _relative_rms(_predict(student, *train_inputs), train_target)
        validation_error = _relative_rms(
            _predict(student, *validation_inputs), validation_target
        )
    return {
        "synthetic_initial_relative_rms": initial,
        "synthetic_train_relative_rms": train_error,
        "synthetic_validation_relative_rms": validation_error,
    }


def _m7_metrics(
    model: FullBlockNeuralPairMapper,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    descriptor_i, descriptor_j, displacement = inputs
    with torch.no_grad():
        reference = model.predict_offsite(
            ("A", "B"), descriptor_i, displacement, descriptor_j, roll=0.0
        )
        rolled = model.predict_offsite(
            ("A", "B"), descriptor_i, displacement, descriptor_j, roll=0.731
        )
        global_model = _make_mapper("m3", descriptor_i.dtype, descriptor_i.device)
        global_model.offsite_kernels["A__B"].load_state_dict(
            model.offsite_kernels["A__B"].local_kernel.state_dict()
        )
        global_result = _predict(global_model, descriptor_i, descriptor_j, displacement)
    return {
        "bond_roll_relative_error": _relative_error(rolled, reference),
        "local_global_low_l_relative_error": _relative_error(reference, global_result),
    }


def _benchmark(
    architecture: str,
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    steps: int,
    generator: torch.Generator,
) -> dict[str, float]:
    model = _make_mapper(architecture, torch.float32, device)
    inputs = _random_inputs(
        batch_size, dtype=torch.float32, device=device, generator=generator
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    includes_backward = any(parameter.requires_grad for parameter in model.parameters())
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        loss = _predict(model, *inputs).square().mean()
        if includes_backward:
            loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        loss = _predict(model, *inputs).square().mean()
        if includes_backward:
            loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "benchmark_includes_backward": includes_backward,
        "forward_backward_pairs_per_second": steps * batch_size / elapsed,
        "forward_backward_milliseconds_per_batch": 1000.0 * elapsed / steps,
        "peak_memory_bytes": float(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "convention": "mandala-stage4-mapper-correctness-v1",
        "architectures": args.architectures,
        "descriptor_irreps": str(DESCRIPTOR_IRREPS),
        "target_orbitals": {"A": "1s1p", "B": "1s1p"},
        "device": str(device),
        "seed": args.seed,
        "fit_samples": args.fit_samples,
        "fit_validation_samples": args.fit_validation_samples,
        "fit_steps": args.fit_steps,
        "fit_learning_rate": args.fit_learning_rate,
        "benchmark_batch_size": args.benchmark_batch_size,
        "benchmark_warmup": args.benchmark_warmup,
        "benchmark_steps": args.benchmark_steps,
        "acceptance": {
            "equivariance_relative_error": args.equivariance_tolerance,
            "reversal_relative_error": args.reversal_tolerance,
            "synthetic_validation_relative_rms": args.synthetic_fit_relative_rms,
            "m7_roll_relative_error": args.equivariance_tolerance,
            "m7_low_l_relative_error": args.equivariance_tolerance,
        },
        "selection_uses_test_hamiltonian": False,
    }
    config["manifest_hash"] = _hash(config)
    _atomic_json(args.output_dir / "config.json", config)

    reference_inputs = _random_inputs(
        8, dtype=torch.float64, device=device, generator=generator
    )
    train_inputs = _random_inputs(
        args.fit_samples, dtype=torch.float64, device=device, generator=generator
    )
    validation_inputs = _random_inputs(
        args.fit_validation_samples,
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    rows: list[dict[str, object]] = []
    for architecture in args.architectures:
        print(f"[{len(rows) + 1}/{len(args.architectures)}] {architecture}", flush=True)
        reference = _make_mapper(architecture, torch.float64, device)
        if architecture == "m1":
            assert isinstance(reference, NativeACEPairMapper)
            _randomize_ace_teacher(reference)
        symmetry = _symmetry_metrics(reference, reference_inputs)
        initial_m7 = (
            _m7_metrics(reference, reference_inputs)
            if architecture == "m7"
            else {
                "bond_roll_relative_error": "",
                "local_global_low_l_relative_error": "",
            }
        )
        float32_model = _make_mapper(architecture, torch.float32, device)
        float32_output = _predict(
            float32_model, *(value.float() for value in validation_inputs)
        )
        fitting = _synthetic_fit(
            architecture,
            reference,
            train_inputs,
            validation_inputs,
            steps=args.fit_steps,
            learning_rate=args.fit_learning_rate,
        )
        post_training = _symmetry_metrics(reference, reference_inputs)
        row: dict[str, object] = {
            "manifest_hash": config["manifest_hash"],
            "architecture": architecture,
            "parameter_count": sum(p.numel() for p in reference.parameters()),
            **symmetry,
            "float32_finite": bool(torch.isfinite(float32_output).all()),
            **fitting,
            "post_train_proper_o3_relative_error": post_training[
                "proper_o3_relative_error"
            ],
            "post_train_improper_o3_relative_error": post_training[
                "improper_o3_relative_error"
            ],
            **initial_m7,
        }
        if architecture == "m7":
            post_m7 = _m7_metrics(reference, reference_inputs)
            row.update(
                {
                    "post_train_bond_roll_relative_error": post_m7[
                        "bond_roll_relative_error"
                    ],
                    "post_train_local_global_low_l_relative_error": post_m7[
                        "local_global_low_l_relative_error"
                    ],
                }
            )
        else:
            row.update(
                {
                    "post_train_bond_roll_relative_error": "",
                    "post_train_local_global_low_l_relative_error": "",
                }
            )
        row.update(
            _benchmark(
                architecture,
                device=device,
                batch_size=args.benchmark_batch_size,
                warmup=args.benchmark_warmup,
                steps=args.benchmark_steps,
                generator=generator,
            )
        )
        equivariance_values = [
            float(row[name])
            for name in (
                "proper_o3_relative_error",
                "improper_o3_relative_error",
                "onsite_improper_o3_relative_error",
                "post_train_proper_o3_relative_error",
                "post_train_improper_o3_relative_error",
            )
        ]
        reversal_values = [
            float(row["heterogeneous_reversal_relative_error"]),
            float(row["homogeneous_reversal_relative_error"]),
            float(row["onsite_hermiticity_relative_error"]),
        ]
        row["passed"] = bool(
            max(equivariance_values) <= args.equivariance_tolerance
            and max(reversal_values) <= args.reversal_tolerance
            and row["irrelevant_metadata_bitwise"]
            and row["output_dimension_exact"]
            and row["float32_finite"]
            and float(row["synthetic_validation_relative_rms"])
            <= args.synthetic_fit_relative_rms
            and (
                architecture != "m7"
                or (
                    float(row["bond_roll_relative_error"])
                    <= args.equivariance_tolerance
                    and float(row["local_global_low_l_relative_error"])
                    <= args.equivariance_tolerance
                    and float(row["post_train_bond_roll_relative_error"])
                    <= args.equivariance_tolerance
                    and float(row["post_train_local_global_low_l_relative_error"])
                    <= args.equivariance_tolerance
                )
            )
        )
        rows.append(row)
        _atomic_json(args.output_dir / f"{architecture}.json", row)

    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "completed": True,
        "manifest_hash": config["manifest_hash"],
        "architecture_count": len(rows),
        "pass_count": sum(bool(row["passed"]) for row in rows),
        "passed": all(bool(row["passed"]) for row in rows),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    lines = [
        "# Stage 4 pair-mapper correctness",
        "",
        "No Hamiltonian target was used; fitting targets came from frozen synthetic equivariant teachers.",
        "",
        "| Mapper | Passed | Max initial O(3) error | Synthetic validation rel. RMS | pairs/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        max_o3 = max(
            float(row["proper_o3_relative_error"]),
            float(row["improper_o3_relative_error"]),
            float(row["onsite_improper_o3_relative_error"]),
        )
        lines.append(
            f"| {row['architecture'].upper()} | {row['passed']} | {max_o3:.3e} | "
            f"{float(row['synthetic_validation_relative_rms']):.3e} | "
            f"{float(row['forward_backward_pairs_per_second']):.1f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
