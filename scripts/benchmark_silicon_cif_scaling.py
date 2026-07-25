#!/usr/bin/env python
"""Benchmark target-free Silicon inference as the periodic structure grows."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402
from core.orbital_irrep_config import OrbitalIrrepConfig  # noqa: E402
from data.structure_inference import (  # noqa: E402
    build_model_input_from_structure,
    load_orbital_cfg_from_reference_info,
    load_structure_from_cif,
)
from net.e3gnn import E3GNN  # noqa: E402
from scripts.evaluate_checkpoint_materials import (  # noqa: E402
    _load_checkpoint,
    _restore_config,
    _sanitize_eval_config,
)


REPEATS = 10
EXPECTED_ATOM_COUNTS = (8, 64, 512, 4096, 32768, 262144)
DEFAULT_STRUCTURES_DIR = REPO_ROOT / "benchmark_data/silicon_scaling"
DEFAULT_OUTPUT = REPO_ROOT / "eval_outputs/silicon_cif_scaling_b200.txt"
CLUSTER_ACCURATE_CHECKPOINT = Path(
    "/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/"
    "silicon_perturbed_scales_hdo_compat_from_likely_sweep_22_47h5/"
    "sleek-sweep-8-restart-lr5em4-p120/best_model.pt"
)
LOCAL_ACCURATE_CHECKPOINT = (
    REPO_ROOT / "checkpoints/silicon_perturbed_hamiltonian/"
    "sleek-sweep-8-restart-lr2em3/best_model.pt"
)


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self._handle.write(message + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class ChunkProgressLogger:
    """Rate-limit chunk progress output without perturbing normal timings."""

    def __init__(self, logger: TeeLogger) -> None:
        self.logger = logger

    def __call__(self, stage: str, current: int, total: int) -> None:
        interval = max(1, total // 10)
        if current == 1 or current == total or current % interval == 0:
            self.logger.log(f"[chunked] {stage}: {current}/{total}")


@dataclass(frozen=True)
class TimingSummary:
    values: tuple[float, ...]
    minimum: float
    maximum: float
    mean: float
    stddev: float


def summarize(values: list[float]) -> TimingSummary:
    if len(values) != REPEATS:
        raise ValueError(f"Expected exactly {REPEATS} timings, got {len(values)}")
    return TimingSummary(
        values=tuple(values),
        minimum=min(values),
        maximum=max(values),
        mean=statistics.fmean(values),
        stddev=statistics.pstdev(values),
    )


def default_checkpoint() -> Path:
    if CLUSTER_ACCURATE_CHECKPOINT.is_file():
        return CLUSTER_ACCURATE_CHECKPOINT
    return LOCAL_ACCURATE_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CIF preparation, Mandala forward inference, and sparse block "
            "matrix construction for progressively larger Silicon supercells."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--edge-chunk-size",
        type=int,
        default=None,
        help=(
            "Use exact edge-streamed inference with this many edges per chunk. "
            "This combines model evaluation and matrix construction into one "
            "bounded-memory operation."
        ),
    )
    parser.add_argument(
        "--edge-store-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Persistent edge-state placement for --edge-chunk-size inference.",
    )
    parser.add_argument(
        "--edge-chunk-progress",
        action="store_true",
        help="Log coarse edge-chunk progress during bounded-memory inference.",
    )
    parser.add_argument(
        "--atom-counts",
        type=str,
        default=",".join(str(value) for value in EXPECTED_ATOM_COUNTS),
        help="Comma-separated subset of 8,64,512,4096,32768,262144.",
    )
    parser.add_argument(
        "--reference-info-path",
        type=Path,
        default=None,
        help=(
            "Optional OpenMX Si.out used only to define the orbital basis. Without "
            "it, the standard Silicon 2s2p1d basis is used."
        ),
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**3


def format_summary(name: str, summary: TimingSummary) -> str:
    raw = ", ".join(f"{value:.6f}" for value in summary.values)
    return (
        f"{name}: repeats={REPEATS} min={summary.minimum:.6f}s "
        f"max={summary.maximum:.6f}s avg={summary.mean:.6f}s "
        f"stddev={summary.stddev:.6f}s raw_seconds=[{raw}]"
    )


def parse_atom_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --atom-counts value: {value!r}") from exc
    if not counts or len(counts) != len(set(counts)):
        raise ValueError("--atom-counts must contain unique comma-separated integers")
    unknown = sorted(set(counts).difference(EXPECTED_ATOM_COUNTS))
    if unknown:
        raise ValueError(
            f"Unsupported atom counts {unknown}; available={EXPECTED_ATOM_COUNTS}"
        )
    return tuple(count for count in EXPECTED_ATOM_COUNTS if count in counts)


def expected_cif_paths(
    structures_dir: Path,
    atom_counts: tuple[int, ...],
) -> list[tuple[int, Path]]:
    result = []
    for atom_count in atom_counts:
        path = structures_dir / f"silicon_{atom_count:05d}_atoms.cif"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing benchmark CIF: {path}. Run "
                "scripts/generate_silicon_scaling_cifs.py first."
            )
        result.append((atom_count, path))
    return result


def time_repeated(
    operation: Callable[[], Any],
    *,
    device: torch.device,
) -> tuple[TimingSummary, Any]:
    values: list[float] = []
    last_result: Any = None
    for _ in range(REPEATS):
        if last_result is not None:
            del last_result
            last_result = None
            gc.collect()
        synchronize(device)
        started = time.perf_counter()
        result = operation()
        synchronize(device)
        values.append(time.perf_counter() - started)
        last_result = result
    return summarize(values), last_result


def warmup(operation: Callable[[], Any], count: int, device: torch.device) -> None:
    for _ in range(count):
        result = operation()
        synchronize(device)
        del result


def build_input(cif_path: Path, cfg, mapper, device: torch.device) -> dict[str, Any]:
    atoms, positions, box = load_structure_from_cif(
        cif_path,
        dtype=cfg.dtype,
        device=device,
    )
    return build_model_input_from_structure(
        atoms=atoms,
        positions=positions,
        box=box,
        cfg=cfg,
        mapper=mapper,
    )


def load_model(args: argparse.Namespace, device: torch.device):
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path)
    cfg = _sanitize_eval_config(_restore_config(checkpoint))
    cfg.benchmark = False
    cfg.log_activation_mag = False
    cfg.log_grad_norm = False

    if args.reference_info_path is None:
        orbital_cfg = OrbitalIrrepConfig.from_dict({"Si": ["2x0e", "2x1o", "1x2e"]})
    else:
        orbital_cfg = load_orbital_cfg_from_reference_info(
            args.reference_info_path.expanduser().resolve(),
            dtype=cfg.dtype,
        )
    mapper = BlockIrrepMapper(
        orbital_cfg,
        diagonal=False,
        device="cpu",
        dtype=cfg.dtype,
    )
    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return checkpoint_path, cfg, mapper, model


def benchmark_structure(
    *,
    atom_count: int,
    cif_path: Path,
    cfg,
    mapper,
    model: E3GNN,
    device: torch.device,
    warmup_runs: int,
    edge_chunk_size: int | None,
    edge_store_device: str,
    edge_chunk_progress: bool,
    logger: TeeLogger,
) -> None:
    logger.log()
    logger.log(f"=== atoms={atom_count} cif={cif_path} ===")

    prepare = partial(build_input, cif_path, cfg, mapper, device)

    warmup(prepare, 1, device)
    gc.collect()
    reset_peak_memory(device)
    preparation_summary, model_input = time_repeated(prepare, device=device)
    preparation_peak = peak_memory_gib(device)
    actual_atoms = len(model_input["atoms"])
    edge_count = int(model_input["edge_index"].shape[1])
    if actual_atoms != atom_count:
        raise ValueError(
            f"CIF {cif_path} contains {actual_atoms} atoms, expected {atom_count}"
        )
    logger.log(format_summary("data_preparation", preparation_summary))
    logger.log(f"graph: atoms={actual_atoms} directed_edges={edge_count}")
    if preparation_peak is not None:
        logger.log(f"data_preparation_peak_gpu_memory={preparation_peak:.6f} GiB")

    if edge_chunk_size is not None:

        # Bind the prepared input explicitly so static analysis and repeated
        # benchmark invocations both use this exact graph instance.
        def chunked_inference(input_payload=model_input):
            return model.predict_matrices_chunked(
                input_payload,
                edge_chunk_size=edge_chunk_size,
                edge_store_device=edge_store_device,
                output_device="cpu",
                physical=True,
                progress=ChunkProgressLogger(logger) if edge_chunk_progress else None,
            )

        with torch.inference_mode():
            warmup(chunked_inference, warmup_runs, device)
            reset_peak_memory(device)
            chunked_summary, matrices = time_repeated(chunked_inference, device=device)
        chunked_peak = peak_memory_gib(device)
        matrix_names = sorted(matrices)
        total_blocks = sum(
            int(blocks.shape[0])
            for matrix in matrices.values()
            for blocks in matrix.pair_blocks.values()
        )
        logger.log(
            format_summary("chunked_inference_and_matrix_construction", chunked_summary)
        )
        logger.log(f"matrices={matrix_names} total_matrix_blocks={total_blocks}")
        if chunked_peak is not None:
            logger.log(f"chunked_inference_peak_gpu_memory={chunked_peak:.6f} GiB")
        del matrices, model_input
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return

    forward = partial(model, model_input)

    with torch.inference_mode():
        warmup(forward, warmup_runs, device)
        reset_peak_memory(device)
        forward_summary, predictions = time_repeated(forward, device=device)
    forward_peak = peak_memory_gib(device)
    logger.log(format_summary("model_evaluation", forward_summary))
    if forward_peak is not None:
        logger.log(f"model_evaluation_peak_gpu_memory={forward_peak:.6f} GiB")

    construct = partial(
        model.predicted_irreps_to_block_matrices,
        predictions,
        model_input,
        physical=True,
    )
    with torch.inference_mode():
        warmup(construct, warmup_runs, device)
        reset_peak_memory(device)
        matrix_summary, matrices = time_repeated(construct, device=device)
    matrix_peak = peak_memory_gib(device)
    matrix_names = sorted(matrices)
    total_blocks = sum(
        int(blocks.shape[0])
        for matrix in matrices.values()
        for blocks in matrix.pair_blocks.values()
    )
    logger.log(format_summary("matrix_construction", matrix_summary))
    logger.log(f"matrices={matrix_names} total_matrix_blocks={total_blocks}")
    if matrix_peak is not None:
        logger.log(f"matrix_construction_peak_gpu_memory={matrix_peak:.6f} GiB")

    del matrices, predictions, model_input
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 1:
        raise ValueError("--warmup-runs must be at least 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    logger = TeeLogger(args.output)
    try:
        structures_dir = args.structures_dir.expanduser().resolve()
        atom_counts = parse_atom_counts(args.atom_counts)
        structures = expected_cif_paths(structures_dir, atom_counts)
        checkpoint_path, cfg, mapper, model = load_model(args, device)

        logger.log("=== Mandala Silicon CIF scaling benchmark ===")
        logger.log(f"checkpoint={checkpoint_path}")
        logger.log(f"structures_dir={structures_dir}")
        logger.log(f"output={logger.path}")
        logger.log(f"device={device}")
        if device.type == "cuda":
            logger.log(f"gpu={torch.cuda.get_device_name(device)}")
        logger.log(f"timed_repeats={REPEATS}")
        logger.log(f"warmup_runs={args.warmup_runs}")
        logger.log(f"edge_chunk_size={args.edge_chunk_size}")
        logger.log(f"edge_store_device={args.edge_store_device}")
        logger.log(f"edge_chunk_progress={args.edge_chunk_progress}")
        logger.log(f"matrix_targets={list(cfg.matrix_targets)}")
        logger.log(f"cutoff_radius_angstrom={float(cfg.cutoff_radius):.6f}")
        logger.log(f"dtype={cfg.dtype}")

        for atom_count, cif_path in structures:
            try:
                benchmark_structure(
                    atom_count=atom_count,
                    cif_path=cif_path,
                    cfg=cfg,
                    mapper=mapper,
                    model=model,
                    device=device,
                    warmup_runs=args.warmup_runs,
                    edge_chunk_size=args.edge_chunk_size,
                    edge_store_device=args.edge_store_device,
                    edge_chunk_progress=args.edge_chunk_progress,
                    logger=logger,
                )
            except Exception as exc:
                logger.log(f"FAILED atoms={atom_count}: {type(exc).__name__}: {exc}")
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        logger.log()
        logger.log(f"benchmark_complete output={logger.path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
