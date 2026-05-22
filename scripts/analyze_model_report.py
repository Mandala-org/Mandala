from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.wandb_sweep_core import (
    RunRecord,
    display_value,
    normalize_run_config,
)  # noqa: E402
from analysis.wandb_sweep_plotly import (  # noqa: E402
    build_model_detail_page,
    copy_evaluation_bundle_assets,
    discover_evaluation_manifests,
    find_manifest_for_checkpoint,
    find_manifest_for_run_id,
)
from net.common import Config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standalone HTML report for one trained model, reusing the sweep run-page renderer."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, default=None)
    source.add_argument("--run-url", type=str, default=None)
    source.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-kind", choices=["latest", "best", "final"], default="best"
    )
    parser.add_argument("--rank-metric", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-state", type=str, default="local")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--evaluation-cache-root", type=Path, default=Path("eval_cache")
    )
    parser.add_argument(
        "--include-plotlyjs",
        type=str,
        default="inline",
        choices=["cdn", "inline"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = discover_evaluation_manifests(args.evaluation_cache_root)
    model_info = _resolve_model_info(args, manifests)
    record = RunRecord(
        run_id=model_info["run_id"],
        name=model_info["run_name"],
        state=model_info["state"],
        score=model_info["score"],
        config=model_info["config"],
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else _default_output_dir(record.name or record.run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_page_index: dict[str, dict[str, Any]] = {record.run_id: {}}
    manifest = model_info.get("manifest")
    if manifest is not None:
        evaluation = copy_evaluation_bundle_assets(
            manifest,
            output_dir / "run_assets" / record.run_id,
            record.run_id,
            href_prefix="run_assets",
        )
        run_page_index[record.run_id]["evaluation"] = evaluation

    html_report = build_model_detail_page(
        record,
        include_plotlyjs=True if args.include_plotlyjs == "inline" else "cdn",
        run_page_index=run_page_index,
        rank_metric=model_info.get("rank_metric"),
        external_url=model_info.get("external_url"),
    )
    html_path = output_dir / "model_report.html"
    html_path.write_text(html_report, encoding="utf-8")
    print(f"Saved model report: {html_path}")


def _resolve_model_info(
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    if args.evaluation_dir is not None:
        manifest = _manifest_from_bundle_dir(args.evaluation_dir)
        checkpoint_path = _checkpoint_path_from_manifest(manifest)
        config = _config_from_checkpoint_or_empty(checkpoint_path)
        source = manifest.get("payload", {}).get("source", {})
        run_id = args.run_id or str(
            source.get("run_id") or _fallback_run_id(checkpoint_path)
        )
        run_name = args.run_name or str(
            source.get("run_name") or _fallback_run_name(checkpoint_path, run_id)
        )
        score = None
        if args.rank_metric is not None:
            score = None
        return {
            "run_id": run_id,
            "run_name": run_name,
            "state": args.run_state,
            "score": score,
            "config": config,
            "rank_metric": args.rank_metric,
            "external_url": source.get("run_url"),
            "manifest": manifest,
        }

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        config = _config_from_checkpoint_or_empty(checkpoint_path)
        manifest = find_manifest_for_checkpoint(checkpoint_path, manifests)
        source = (
            manifest.get("payload", {}).get("source", {})
            if manifest is not None
            else {}
        )
        run_id = args.run_id or str(
            source.get("run_id") or _fallback_run_id(checkpoint_path)
        )
        run_name = args.run_name or str(
            source.get("run_name") or _fallback_run_name(checkpoint_path, run_id)
        )
        return {
            "run_id": run_id,
            "run_name": run_name,
            "state": args.run_state,
            "score": None,
            "config": config,
            "rank_metric": args.rank_metric,
            "external_url": source.get("run_url"),
            "manifest": manifest,
        }

    assert args.run_url is not None
    entity, project, run_id = _parse_run_ref(args.run_url)
    run = _load_wandb_run(entity, project, run_id)
    summary = _mapping(getattr(run, "summary", {}))
    config = normalize_run_config(_mapping(getattr(run, "config", {})))
    checkpoint_path = _checkpoint_path_from_run_summary(summary, args.checkpoint_kind)
    if checkpoint_path is not None:
        checkpoint_config = _config_from_checkpoint_or_empty(checkpoint_path)
        if checkpoint_config:
            config = checkpoint_config
    score = None
    if args.rank_metric is not None and args.rank_metric in summary:
        try:
            score = float(summary[args.rank_metric])
        except Exception:
            score = None
    manifest = find_manifest_for_run_id(run_id, manifests)
    if manifest is None and checkpoint_path is not None:
        manifest = find_manifest_for_checkpoint(checkpoint_path, manifests)
    return {
        "run_id": args.run_id or run_id,
        "run_name": args.run_name or str(getattr(run, "name", run_id)),
        "state": (
            args.run_state
            if args.run_state != "local"
            else str(getattr(run, "state", ""))
        ),
        "score": score,
        "config": config,
        "rank_metric": args.rank_metric,
        "external_url": args.run_url,
        "manifest": manifest,
    }


def _manifest_from_bundle_dir(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "evaluation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No evaluation_manifest.json found in {bundle_dir}")
    import json

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest_path": manifest_path,
        "bundle_dir": bundle_dir,
        "payload": payload,
    }


def _checkpoint_path_from_manifest(manifest: dict[str, Any]) -> Path | None:
    payload = manifest.get("payload", {})
    source = payload.get("source", {})
    candidate = source.get("checkpoint_path") or payload.get("checkpoint")
    if not candidate:
        return None
    path = Path(str(candidate)).expanduser()
    return path.resolve() if path.exists() else path


def _checkpoint_path_from_run_summary(
    summary: dict[str, Any],
    checkpoint_kind: str,
) -> Path | None:
    key = {
        "latest": "checkpoint/latest_path",
        "best": "checkpoint/best_path",
        "final": "checkpoint/final_path",
    }[checkpoint_kind]
    candidate = summary.get(key)
    if not candidate:
        return None
    path = Path(str(candidate)).expanduser()
    return path.resolve() if path.exists() else path


def _config_from_checkpoint_or_empty(checkpoint_path: Path | None) -> dict[str, Any]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}
    checkpoint = _load_checkpoint(checkpoint_path)
    cfg = _restore_config(checkpoint)
    config_dict = dataclasses.asdict(cfg)
    return {str(key): display_value(value) for key, value in config_dict.items()}


def _patch_config_unpickling() -> None:
    if getattr(Config, "_mandala_legacy_unpickle_patch", False):
        return

    def __setstate__(self, state: Any) -> None:
        state_map: dict[str, Any] = {}
        if isinstance(state, tuple) and len(state) == 2:
            dict_state, slot_state = state
            if isinstance(dict_state, dict):
                state_map.update(dict_state)
            if isinstance(slot_state, dict):
                state_map.update(slot_state)
        elif isinstance(state, dict):
            state_map.update(state)

        defaults = Config()
        for name in Config.__dataclass_fields__:
            if name in state_map:
                object.__setattr__(self, name, state_map[name])
            else:
                object.__setattr__(self, name, getattr(defaults, name))

    Config.__setstate__ = __setstate__  # type: ignore[attr-defined]
    Config._mandala_legacy_unpickle_patch = True  # type: ignore[attr-defined]


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    _patch_config_unpickling()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(checkpoint)!r}")
    return checkpoint


def _restore_config(checkpoint: dict[str, Any]) -> Config:
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    cfg = hyper_parameters.get("cfg")
    if not isinstance(cfg, Config):
        raise ValueError(
            "Checkpoint does not contain a Config instance in hyper_parameters['cfg']."
        )
    cfg = copy.deepcopy(cfg)
    if isinstance(cfg.dtype, str):
        cfg.dtype = getattr(torch, cfg.dtype)
    if isinstance(cfg.matrix_targets, str):
        cfg.matrix_targets = [cfg.matrix_targets]
    return cfg


def _load_wandb_run(entity: str, project: str, run_id: str):
    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        raise SystemExit("wandb is required when using --run-url.") from exc
    api = wandb.Api()
    return api.run(f"{entity}/{project}/{run_id}")


def _parse_run_ref(ref: str) -> tuple[str, str, str]:
    from urllib.parse import urlparse

    if "://" in ref:
        parsed = urlparse(ref)
        parts = [part for part in parsed.path.split("/") if part]
    else:
        parts = [part for part in ref.split("/") if part]
    if len(parts) >= 4 and parts[2] == "runs":
        entity, project, _runs, run_id = parts[:4]
        return entity, project, run_id
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(
        "Expected a W&B run URL like https://wandb.ai/<entity>/<project>/runs/<run_id>"
    )


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    return dict(value)


def _fallback_run_id(checkpoint_path: Path | None) -> str:
    if checkpoint_path is None:
        return "local-model"
    return checkpoint_path.parent.name or checkpoint_path.stem


def _fallback_run_name(checkpoint_path: Path | None, run_id: str) -> str:
    if checkpoint_path is None:
        return run_id
    return checkpoint_path.parent.name or checkpoint_path.stem or run_id


def _default_output_dir(label: str) -> Path:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in label).strip("-")
    slug = "-".join(part for part in slug.split("-") if part) or "model-report"
    return Path("model_reports") / slug


if __name__ == "__main__":
    main()
