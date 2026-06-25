#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from net.checkpoint_compat import (  # noqa: E402
    initialize_model_from_checkpoint,
    load_checkpoint_payload,
    strip_mapper_keys,
)
from net.e3gnn import E3GNN  # noqa: E402
from scripts.dataset import (  # noqa: E402
    build_silicon_datasets,
    build_silicon_scales_datasets,
    build_siox_datasets,
    build_zncusnses_datasets,
    build_zncusnses_small_datasets,
)
from scripts.evaluate_checkpoint_materials import (  # noqa: E402
    _patch_config_unpickling,
    _restore_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check whether compatibility-mode initialization reproduces the "
            "same model and validation metrics as a strict direct reload."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-kind",
        type=str,
        required=True,
        choices=[
            "silicon",
            "silicon_scales",
            "siox",
            "ZnCuSnSeS_small",
            "ZnCuSnSeS",
        ],
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--scales", type=str, default="[1]")
    parser.add_argument("--num-train-per-scale", type=int, default=80)
    parser.add_argument("--num-val-per-scale", type=int, default=20)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--convention", type=str, default="e3nn")
    parser.add_argument("--snapshot-cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-device", type=str, default="cpu")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-val-samples", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _parse_scales(raw: str) -> list[int]:
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--scales must decode to a non-empty JSON list")
    return [int(x) for x in parsed]


def _build_dataset_bundle(args: argparse.Namespace, cfg) -> tuple[Any, Any, Any]:
    if args.snapshot_cache_dir is not None:
        cfg.snapshot_cache_dir = str(args.snapshot_cache_dir.expanduser().resolve())
    cfg.dataset_device = args.dataset_device
    if args.dataset_kind == "silicon":
        return build_silicon_datasets(
            data_path=args.data_path,
            cfg=cfg,
            num_train=args.num_train,
            num_val=args.num_val,
            seed=args.seed,
            convention=args.convention,
        )
    if args.dataset_kind == "siox":
        return build_siox_datasets(
            data_path=args.data_path,
            cfg=cfg,
            num_train=args.num_train,
            num_val=args.num_val,
            val_fraction=args.val_fraction,
            seed=args.seed,
            convention=args.convention,
        )
    if args.dataset_kind == "silicon_scales":
        return build_silicon_scales_datasets(
            data_path=args.data_path,
            cfg=cfg,
            scales=_parse_scales(args.scales),
            num_train_per_scale=args.num_train_per_scale,
            num_val_per_scale=args.num_val_per_scale,
            seed=args.seed,
            convention=args.convention,
        )
    if args.dataset_kind == "ZnCuSnSeS_small":
        return build_zncusnses_small_datasets(
            data_path=args.data_path,
            cfg=cfg,
            num_train=args.num_train,
            num_val=args.num_val,
            val_fraction=args.val_fraction,
            seed=args.seed,
            convention=args.convention,
        )
    if args.dataset_kind == "ZnCuSnSeS":
        return build_zncusnses_datasets(
            data_path=args.data_path,
            cfg=cfg,
            scales=_parse_scales(args.scales),
            num_train_per_scale=args.num_train_per_scale,
            num_val_per_scale=args.num_val_per_scale,
            seed=args.seed,
            convention=args.convention,
        )
    raise ValueError(f"Unsupported dataset_kind={args.dataset_kind!r}")


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(value, device) for value in obj)
    if hasattr(obj, "to"):
        try:
            return obj.to(device)
        except TypeError:
            return obj.to(device=device)
    return obj


def _capture_metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            detached = value.detach().cpu()
            if detached.numel() == 1:
                payload[key] = float(detached.item())
            else:
                payload[key] = detached
        else:
            payload[key] = value
    return payload


def _evaluate_shared_step(
    model: E3GNN,
    batch: tuple[dict[str, Any], dict[str, Any]],
    *,
    batch_idx: int,
) -> tuple[float | None, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def _log_dict(metrics: dict[str, Any], *args, **kwargs) -> None:
        captured.update(_capture_metric_payload(metrics))

    model.log_dict = _log_dict  # type: ignore[method-assign]
    with torch.no_grad():
        loss = model._shared_step(batch, batch_idx, stage="val")
    if loss is None:
        return None, captured
    return float(loss.detach().cpu().item()), captured


def _compare_state_dicts(
    direct_model: E3GNN,
    compat_model: E3GNN,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    direct_state = direct_model.state_dict()
    compat_state = compat_model.state_dict()
    keys_direct = set(direct_state)
    keys_compat = set(compat_state)
    shared = sorted(keys_direct.intersection(keys_compat))
    max_abs = 0.0
    max_key = None
    mismatched: list[dict[str, Any]] = []
    for key in shared:
        lhs = direct_state[key].detach().cpu()
        rhs = compat_state[key].detach().cpu()
        if lhs.shape != rhs.shape:
            mismatched.append(
                {
                    "key": key,
                    "reason": "shape",
                    "direct": tuple(lhs.shape),
                    "compat": tuple(rhs.shape),
                }
            )
            continue
        diff = torch.max(torch.abs(lhs - rhs)).item() if lhs.numel() else 0.0
        if diff > max_abs:
            max_abs = float(diff)
            max_key = key
        if not torch.allclose(lhs, rhs, atol=atol, rtol=rtol):
            mismatched.append(
                {"key": key, "reason": "value", "max_abs_diff": float(diff)}
            )
    return {
        "num_direct_keys": len(keys_direct),
        "num_compat_keys": len(keys_compat),
        "missing_in_compat": sorted(keys_direct - keys_compat),
        "missing_in_direct": sorted(keys_compat - keys_direct),
        "num_shared_keys": len(shared),
        "num_mismatched_shared_keys": len(mismatched),
        "max_abs_diff": max_abs,
        "max_abs_diff_key": max_key,
        "mismatched_shared_keys": mismatched[:50],
    }


def _compare_metric_maps(
    direct_metrics: dict[str, Any],
    compat_metrics: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    keys = sorted(set(direct_metrics) | set(compat_metrics))
    diffs: list[dict[str, Any]] = []
    max_abs = 0.0
    max_key = None
    for key in keys:
        if key not in direct_metrics or key not in compat_metrics:
            diffs.append(
                {
                    "key": key,
                    "reason": "missing",
                    "in_direct": key in direct_metrics,
                    "in_compat": key in compat_metrics,
                }
            )
            continue
        lhs = direct_metrics[key]
        rhs = compat_metrics[key]
        if isinstance(lhs, float) and isinstance(rhs, float):
            diff = abs(lhs - rhs)
            if diff > max_abs:
                max_abs = diff
                max_key = key
            if not math.isclose(lhs, rhs, abs_tol=atol, rel_tol=rtol):
                diffs.append(
                    {
                        "key": key,
                        "reason": "value",
                        "direct": lhs,
                        "compat": rhs,
                        "abs_diff": diff,
                    }
                )
            continue
        if torch.is_tensor(lhs) and torch.is_tensor(rhs):
            if lhs.shape != rhs.shape:
                diffs.append(
                    {
                        "key": key,
                        "reason": "shape",
                        "direct_shape": tuple(lhs.shape),
                        "compat_shape": tuple(rhs.shape),
                    }
                )
                continue
            diff = torch.max(torch.abs(lhs - rhs)).item() if lhs.numel() else 0.0
            if diff > max_abs:
                max_abs = float(diff)
                max_key = key
            if not torch.allclose(lhs, rhs, atol=atol, rtol=rtol):
                diffs.append(
                    {
                        "key": key,
                        "reason": "tensor",
                        "max_abs_diff": float(diff),
                    }
                )
            continue
        if lhs != rhs:
            diffs.append(
                {
                    "key": key,
                    "reason": "non_numeric",
                    "direct": lhs,
                    "compat": rhs,
                }
            )
    return {
        "num_keys": len(keys),
        "num_differences": len(diffs),
        "max_abs_diff": max_abs,
        "max_abs_diff_key": max_key,
        "differences": diffs[:100],
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _patch_config_unpickling()
    checkpoint = load_checkpoint_payload(checkpoint_path)
    cfg = _restore_config(checkpoint)
    cfg = deepcopy(cfg)
    cfg.snapshot_cache_dir = (
        None
        if args.snapshot_cache_dir is None
        else str(args.snapshot_cache_dir.expanduser().resolve())
    )
    cfg.dataset_device = args.dataset_device
    cfg.verbosity = 0

    print("--- Building dataset bundle for equivalence check ---")
    train_ds, val_ds, mapper = _build_dataset_bundle(args, cfg)
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset is empty.")
    device = torch.device(args.device)

    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = strip_mapper_keys(state_dict)

    direct_model = E3GNN(mapper=mapper, cfg=deepcopy(cfg)).to(device)
    compat_model = E3GNN(mapper=mapper, cfg=deepcopy(cfg)).to(device)
    direct_model.eval()
    compat_model.eval()

    print("--- Loading direct strict model ---")
    direct_model.load_state_dict(state_dict, strict=True)

    print("--- Loading compatibility model ---")
    compat_report = initialize_model_from_checkpoint(compat_model, checkpoint_path)

    state_report = _compare_state_dicts(
        direct_model,
        compat_model,
        atol=args.atol,
        rtol=args.rtol,
    )

    print(
        "--- State dict comparison --- "
        f"shared={state_report['num_shared_keys']} "
        f"mismatched={state_report['num_mismatched_shared_keys']} "
        f"max_abs_diff={state_report['max_abs_diff']:.6e} "
        f"key={state_report['max_abs_diff_key']}"
    )

    sample_reports: list[dict[str, Any]] = []
    num_samples = min(int(args.num_val_samples), len(val_ds))
    iterator = tqdm(range(num_samples), desc="Validation equivalence samples")
    for batch_idx in iterator:
        x, y = val_ds[batch_idx]
        batch = (_move_to_device(x, device), _move_to_device(y, device))
        direct_loss, direct_metrics = _evaluate_shared_step(
            direct_model,
            batch,
            batch_idx=batch_idx,
        )
        compat_loss, compat_metrics = _evaluate_shared_step(
            compat_model,
            batch,
            batch_idx=batch_idx,
        )
        metric_report = _compare_metric_maps(
            direct_metrics,
            compat_metrics,
            atol=args.atol,
            rtol=args.rtol,
        )
        loss_abs_diff = (
            None
            if direct_loss is None or compat_loss is None
            else abs(direct_loss - compat_loss)
        )
        sample_report = {
            "batch_idx": batch_idx,
            "direct_loss": direct_loss,
            "compat_loss": compat_loss,
            "loss_abs_diff": loss_abs_diff,
            "metric_report": metric_report,
        }
        sample_reports.append(sample_report)
        iterator.set_postfix_str(
            f"loss_diff={0.0 if loss_abs_diff is None else loss_abs_diff:.3e}, "
            f"metric_diffs={metric_report['num_differences']}"
        )

    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset_kind": args.dataset_kind,
        "data_path": str(args.data_path.expanduser().resolve()),
        "compatibility_report": compat_report.to_dict(),
        "state_report": state_report,
        "sample_reports": sample_reports,
    }

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"--- Wrote compatibility equivalence report to {output_path} ---")

    print("\n=== Compatibility Equivalence Summary ===")
    print(
        f"state mismatches: {state_report['num_mismatched_shared_keys']} / "
        f"{state_report['num_shared_keys']}"
    )
    for sample in sample_reports:
        print(
            f"batch {sample['batch_idx']}: "
            f"loss_abs_diff={sample['loss_abs_diff']} "
            f"metric_differences={sample['metric_report']['num_differences']}"
        )


if __name__ == "__main__":
    main()
