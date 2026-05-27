from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import wandb
except Exception as exc:  # pragma: no cover - informative failure
    raise SystemExit(
        "wandb is required for resolving run URLs. Install the project dependencies first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local checkpoint or W&B run with cached output bundles."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, default=None)
    source.add_argument("--run-url", type=str, default=None)
    parser.add_argument(
        "--report-run-id",
        type=str,
        default=None,
        help="Optional W&B run ID metadata for checkpoint-based evaluation bundles.",
    )
    parser.add_argument(
        "--report-run-name",
        type=str,
        default=None,
        help="Optional run name metadata for checkpoint-based evaluation bundles.",
    )
    parser.add_argument(
        "--report-run-url",
        type=str,
        default=None,
        help="Optional W&B run URL metadata for checkpoint-based evaluation bundles.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        default="best",
        choices=["latest", "best", "final"],
        help="Which checkpoint summary field to use when resolving --run-url.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["snapshot", "cif"],
    )
    parser.add_argument("--snapshot-path", type=Path, default=None)
    parser.add_argument("--matrix-path", type=Path, default=None)
    parser.add_argument("--info-path", type=Path, default=None)
    parser.add_argument("--band-info-path", type=Path, default=None)
    parser.add_argument("--special-points-json", type=str, default=None)
    parser.add_argument("--cif-path", type=Path, default=None)
    parser.add_argument("--reference-info-path", type=Path, default=None)
    parser.add_argument("--orbital-set", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--convention", type=str, default="e3nn")
    parser.add_argument("--max-atoms", type=int, default=6)
    parser.add_argument("--cif-max-atoms", type=int, default=8)
    parser.add_argument("--plot-clim", type=float, default=None)
    parser.add_argument("--hamiltonian-clim", type=float, default=0.05)
    parser.add_argument("--density-clim", type=float, default=0.1)
    parser.add_argument("--dos-sigma", type=float, default=0.2)
    parser.add_argument("--dos-bin-width", type=float, default=0.1)
    parser.add_argument("--tetra-batch-size", type=int, default=256)
    parser.add_argument(
        "--dos-method",
        type=str,
        default="tetrahedron",
        choices=["tetrahedron", "gaussian"],
    )
    parser.add_argument("--dos-kmesh", type=str, default="4x4x4")
    parser.add_argument("--dos-energy-min", type=float, default=-10.0)
    parser.add_argument("--dos-energy-max", type=float, default=15.0)
    parser.add_argument("--num-points", type=int, default=240)
    parser.add_argument("--path-string", type=str, default=None)
    parser.add_argument("--use-gt-overlap-for-eigs", action="store_true")
    parser.add_argument("--band-emin-ev", type=float, default=-8.0)
    parser.add_argument("--band-emax-ev", type=float, default=8.0)
    parser.add_argument("--band-line-alpha", type=float, default=0.2)
    parser.add_argument("--correlation-max-points", type=int, default=250000)
    parser.add_argument("--correlation-alpha", type=float, default=0.03)
    parser.add_argument("--correlation-sample-seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overlap-psd-cleanup", action="store_true")
    parser.add_argument("--overlap-jitter", action="store_true")
    parser.add_argument("--force-recompute-bands", action="store_true")
    parser.add_argument("--save-input", action="store_true")
    parser.add_argument("--plot-title", type=str, default=None)
    parser.add_argument("--cache-root", type=Path, default=Path("eval_cache"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore an existing matching cache bundle and recompute everything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_mod = _load_eval_module()
    resolved = _resolve_source(args)
    cache_dir = _resolve_cache_dir(args, resolved)
    manifest_path = cache_dir / "evaluation_manifest.json"
    if manifest_path.exists() and not args.force_refresh:
        print(f"Using cached evaluation bundle: {cache_dir}")
        print(f"Manifest: {manifest_path}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_args = _build_eval_namespace(
        args, resolved["checkpoint_path"], cache_dir, eval_mod
    )
    checkpoint = eval_mod._load_checkpoint(eval_args.checkpoint)
    cfg = eval_mod._restore_config(checkpoint)
    cfg.dataset_device = None
    cfg.snapshot_cache_dir = None

    if "hamiltonian" not in set(cfg.matrix_targets):
        raise ValueError(
            f"Checkpoint matrix_targets={cfg.matrix_targets!r} do not include hamiltonian."
        )

    if eval_args.mode == "snapshot":
        eval_mod._run_snapshot_case(eval_args, checkpoint, cfg)
    else:
        eval_mod._run_cif_case(eval_args, checkpoint, cfg)

    manifest = {
        "source": resolved,
        "mode": eval_args.mode,
        "checkpoint": str(eval_args.checkpoint),
        "output_dir": str(cache_dir),
        "cache_key": cache_dir.name,
        "settings": _manifest_settings(eval_args),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Saved evaluation manifest: {manifest_path}")


def _load_eval_module():
    module_path = REPO_ROOT / "scripts" / "evaluate_checkpoint_materials.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_checkpoint_materials_script", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _resolve_source(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        return {
            "kind": "checkpoint",
            "checkpoint_path": str(checkpoint_path),
            "label": checkpoint_path.stem,
            "run_id": args.report_run_id,
            "run_name": args.report_run_name,
            "run_url": args.report_run_url,
        }
    assert args.run_url is not None
    entity, project, run_id = _parse_run_ref(args.run_url)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    summary = getattr(run, "summary", {})
    summary_get = summary.get if hasattr(summary, "get") else dict(summary).get
    key_by_mode = {
        "latest": "checkpoint/latest_path",
        "best": "checkpoint/best_path",
        "final": "checkpoint/final_path",
    }
    summary_key = key_by_mode[args.checkpoint_kind]
    checkpoint_path = summary_get(summary_key)
    if not checkpoint_path:
        raise ValueError(
            f"W&B run {entity}/{project}/{run_id} does not expose {summary_key!r} in its summary"
        )
    checkpoint_path = str(Path(str(checkpoint_path)).expanduser().resolve())
    return {
        "kind": "wandb_run",
        "run_url": args.run_url,
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "run_name": getattr(run, "name", run_id),
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_path": checkpoint_path,
        "label": f"{getattr(run, 'name', run_id)}-{args.checkpoint_kind}",
    }


def _parse_run_ref(ref: str) -> tuple[str, str, str]:
    if "://" in ref:
        parsed = urlparse(ref)
        parts = [part for part in parsed.path.split("/") if part]
    else:
        parts = [part for part in ref.split("/") if part]

    if len(parts) >= 4 and parts[2] == "runs":
        entity, project, _, run_id = parts[:4]
    elif len(parts) >= 6 and parts[2] == "sweeps" and parts[4] == "runs":
        entity, project, _sweeps, _sweep_id, _runs, run_id = parts[:6]
    elif len(parts) == 3:
        entity, project, run_id = parts
    else:
        raise ValueError(
            "Expected a W&B run URL like "
            "'https://wandb.ai/<entity>/<project>/runs/<run_id>' or "
            "'https://wandb.ai/<entity>/<project>/sweeps/<sweep_id>/runs/<run_id>'"
        )
    return entity, project, run_id


def _resolve_cache_dir(args: argparse.Namespace, resolved: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    source_label = _slugify(resolved["label"])
    checkpoint_path = Path(resolved["checkpoint_path"])
    checkpoint_stat = checkpoint_path.stat()
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "checkpoint_size": checkpoint_stat.st_size,
        "settings": _settings_for_cache(args),
    }
    fingerprint = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return args.cache_root.expanduser().resolve() / source_label / fingerprint


def _build_eval_namespace(
    args: argparse.Namespace,
    checkpoint_path: str,
    output_dir: Path,
    eval_mod: Any,
) -> argparse.Namespace:
    path_string = args.path_string
    if path_string is None:
        path_string = eval_mod.analysis_eval.DEFAULT_PATH_STRING
    return argparse.Namespace(
        checkpoint=Path(checkpoint_path),
        mode=args.mode,
        snapshot_path=args.snapshot_path,
        matrix_path=args.matrix_path,
        info_path=args.info_path,
        band_info_path=args.band_info_path,
        special_points_json=args.special_points_json,
        cif_path=args.cif_path,
        reference_info_path=args.reference_info_path,
        orbital_set=args.orbital_set,
        output_dir=output_dir,
        device=args.device,
        convention=args.convention,
        max_atoms=args.max_atoms,
        cif_max_atoms=args.cif_max_atoms,
        plot_clim=args.plot_clim,
        hamiltonian_clim=args.hamiltonian_clim,
        density_clim=args.density_clim,
        dos_sigma=args.dos_sigma,
        dos_bin_width=args.dos_bin_width,
        tetra_batch_size=args.tetra_batch_size,
        dos_method=args.dos_method,
        dos_kmesh=args.dos_kmesh,
        dos_energy_min=args.dos_energy_min,
        dos_energy_max=args.dos_energy_max,
        num_points=args.num_points,
        path_string=path_string,
        use_gt_overlap_for_eigs=args.use_gt_overlap_for_eigs,
        band_emin_ev=args.band_emin_ev,
        band_emax_ev=args.band_emax_ev,
        band_line_alpha=args.band_line_alpha,
        correlation_max_points=args.correlation_max_points,
        correlation_alpha=args.correlation_alpha,
        correlation_sample_seed=args.correlation_sample_seed,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
        force_recompute_bands=args.force_recompute_bands,
        save_input=args.save_input,
        plot_title=args.plot_title,
    )


def _settings_for_cache(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "snapshot_path": str(args.snapshot_path) if args.snapshot_path else None,
        "matrix_path": str(args.matrix_path) if args.matrix_path else None,
        "info_path": str(args.info_path) if args.info_path else None,
        "band_info_path": str(args.band_info_path) if args.band_info_path else None,
        "special_points_json": args.special_points_json,
        "cif_path": str(args.cif_path) if args.cif_path else None,
        "reference_info_path": (
            str(args.reference_info_path) if args.reference_info_path else None
        ),
        "orbital_set": args.orbital_set,
        "convention": args.convention,
        "max_atoms": args.max_atoms,
        "cif_max_atoms": args.cif_max_atoms,
        "plot_clim": args.plot_clim,
        "hamiltonian_clim": args.hamiltonian_clim,
        "density_clim": args.density_clim,
        "dos_sigma": args.dos_sigma,
        "dos_bin_width": args.dos_bin_width,
        "tetra_batch_size": args.tetra_batch_size,
        "dos_method": args.dos_method,
        "dos_kmesh": args.dos_kmesh,
        "dos_energy_min": args.dos_energy_min,
        "dos_energy_max": args.dos_energy_max,
        "num_points": args.num_points,
        "path_string": args.path_string,
        "use_gt_overlap_for_eigs": args.use_gt_overlap_for_eigs,
        "band_emin_ev": args.band_emin_ev,
        "band_emax_ev": args.band_emax_ev,
        "band_line_alpha": args.band_line_alpha,
        "correlation_max_points": args.correlation_max_points,
        "correlation_alpha": args.correlation_alpha,
        "correlation_sample_seed": args.correlation_sample_seed,
        "chunk_size": args.chunk_size,
        "num_workers": args.num_workers,
        "overlap_psd_cleanup": args.overlap_psd_cleanup,
        "overlap_jitter": args.overlap_jitter,
        "save_input": args.save_input,
        "plot_title": args.plot_title,
    }


def _manifest_settings(eval_args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(eval_args).items()
        if key != "checkpoint"
    }


def _slugify(text: str) -> str:
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else "-")
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "evaluation"


if __name__ == "__main__":
    main()
