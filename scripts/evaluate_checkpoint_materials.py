#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis import evaluation as analysis_eval  # noqa: E402
from core.block_irrep_mapper import BlockIrrepMapper  # noqa: E402
from core.orbital_irrep_config import OrbitalIrrepConfig  # noqa: E402
from data.factory import DatasetFactory  # noqa: E402
from data.openmx_info_parser import parse_info_out  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402
from data.structure_inference import (  # noqa: E402
    build_model_input_from_structure,
    load_orbital_cfg_from_reference_info,
    load_structure_from_cif,
)
from net.common import Config  # noqa: E402
from net.e3gnn import E3GNN  # noqa: E402


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a snapshot or CIF structure with matrix, DOS, and band plots."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["snapshot", "cif"],
        help="Use a ground-truth snapshot comparison or prediction-only CIF evaluation.",
    )
    parser.add_argument("--snapshot-path", type=Path, default=None)
    parser.add_argument("--matrix-path", type=Path, default=None)
    parser.add_argument("--info-path", type=Path, default=None)
    parser.add_argument(
        "--band-info-path",
        type=Path,
        default=None,
        help="Optional separate OpenMX info file used only for resolving Band.kpath special points.",
    )
    parser.add_argument(
        "--special-points-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object mapping k-point labels to fractional coordinates, "
            'for example \'{"G":[0,0,0],"X":[0.5,0,0]}\'.'
        ),
    )
    parser.add_argument("--cif-path", type=Path, default=None)
    parser.add_argument("--reference-info-path", type=Path, default=None)
    parser.add_argument("--orbital-set", type=str, default=None)
    parser.add_argument(
        "--analysis-cutoff-radius",
        type=float,
        default=None,
        help=(
            "Optional post-inference cutoff applied to both predictions and ground truth "
            "before analysis plots and metrics are computed."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--convention", type=str, default="e3nn")
    parser.add_argument("--max-atoms", type=int, default=6)
    parser.add_argument("--cif-max-atoms", type=int, default=8)
    parser.add_argument("--plot-clim", type=float, default=None)
    parser.add_argument("--hamiltonian-clim", type=float, default=0.05)
    parser.add_argument("--density-clim", type=float, default=0.1)
    parser.add_argument("--dos-sigma", type=float, default=0.2)
    parser.add_argument("--dos-bin-width", type=float, default=0.1)
    parser.add_argument(
        "--tetra-batch-size",
        type=int,
        default=256,
        help="Number of tetrahedra processed per DOS batch.",
    )
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
    parser.add_argument(
        "--path-string", type=str, default=analysis_eval.DEFAULT_PATH_STRING
    )
    parser.add_argument(
        "--use-gt-overlap-for-eigs",
        action="store_true",
        help="Use the ground-truth overlap matrix for DOS and band-structure eigensolves in snapshot mode.",
    )
    parser.add_argument("--band-emin-ev", type=float, default=-8.0)
    parser.add_argument("--band-emax-ev", type=float, default=8.0)
    parser.add_argument("--band-line-alpha", type=float, default=0.2)
    parser.add_argument("--correlation-max-points", type=int, default=250000)
    parser.add_argument("--correlation-alpha", type=float, default=0.03)
    parser.add_argument("--correlation-sample-seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--overlap-psd-cleanup",
        action="store_true",
        help="Enable overlap PSD cleanup before generalized eigensolves.",
    )
    parser.add_argument(
        "--overlap-jitter",
        action="store_true",
        help="Allow diagonal jitter retries if overlap Cholesky fails.",
    )
    parser.add_argument("--force-recompute-bands", action="store_true")
    parser.add_argument("--save-input", action="store_true")
    parser.add_argument("--plot-title", type=str, default=None)
    return parser.parse_args()


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


def _resolve_snapshot_cache_dir(cfg: Config, output_dir: Path) -> str | None:
    cache_dir = getattr(cfg, "snapshot_cache_dir", None)
    if not cache_dir:
        fallback = output_dir / "snapshot_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)

    cache_path = Path(cache_dir)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        test_file = cache_path / ".mandala_write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return str(cache_path)
    except OSError:
        fallback = output_dir / "snapshot_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"--- snapshot_cache_dir={cache_path} is not writable here; using local cache {fallback} ---"
        )
        return str(fallback)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def _parse_special_points_json(value: str | None) -> dict[str, list[float]] | None:
    if value is None:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("--special-points-json must decode to a non-empty object")

    special_points: dict[str, list[float]] = {}
    for key, coords in payload.items():
        if not isinstance(key, str):
            raise ValueError("special-point labels must be strings")
        if not isinstance(coords, list) or len(coords) != 3:
            raise ValueError(
                f"special point {key!r} must map to a length-3 coordinate list"
            )
        special_points[key] = [float(coord) for coord in coords]
    return special_points


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(value, device) for value in obj]
        return type(obj)(moved)
    if hasattr(obj, "to"):
        try:
            return obj.to(device)
        except TypeError:
            return obj.to(device=device)
    return obj


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _cpu_copy(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        copied = [_cpu_copy(val) for val in value]
        return type(value)(copied)
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _discover_snapshot_paths(
    snapshot_path: Path | None, matrix_path: Path | None, info_path: Path | None
) -> tuple[Path, Path]:
    if matrix_path is not None and info_path is not None:
        return matrix_path, info_path
    if matrix_path is not None or info_path is not None:
        raise ValueError("Provide both --matrix-path and --info-path together.")
    if snapshot_path is None:
        raise ValueError("--snapshot-path is required in snapshot mode.")
    if snapshot_path.is_file():
        matrix_candidate = snapshot_path
        info_candidate = _discover_info_file(snapshot_path.parent)
        if info_candidate is None:
            raise FileNotFoundError(
                f"Could not infer info file next to matrix file: {snapshot_path}"
            )
        return matrix_candidate, info_candidate
    matrix_candidate = _discover_matrix_file(snapshot_path)
    info_candidate = _discover_info_file(snapshot_path)
    if not matrix_candidate.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_candidate}")
    if info_candidate is None or not info_candidate.exists():
        raise FileNotFoundError(
            f"Info file not found under snapshot path: {snapshot_path}"
        )
    return matrix_candidate, info_candidate


def _discover_matrix_file(snapshot_dir: Path) -> Path:
    for name in ("Si_DM", "HS.out"):
        candidate = snapshot_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer matrix file under snapshot path: {snapshot_dir}"
    )


def _discover_info_file(snapshot_dir: Path) -> Path | None:
    preferred = [
        "info.dat",
        "info.txt",
        "ZnCuSeS.out",
        "SiO2.out",
    ]
    for name in preferred:
        candidate = snapshot_dir / name
        if candidate.exists():
            return candidate
    for candidate in sorted(snapshot_dir.glob("*.out")):
        if candidate.name in {"HS.out", "log.out"}:
            continue
        return candidate
    return None


def _resolve_orbital_cfg(args: argparse.Namespace, cfg: Config) -> OrbitalIrrepConfig:
    if args.reference_info_path is not None:
        return load_orbital_cfg_from_reference_info(
            args.reference_info_path, dtype=cfg.dtype
        )
    if args.orbital_set is not None:
        payload = yaml.safe_load(args.orbital_set)
        if not isinstance(payload, dict):
            raise ValueError(
                "--orbital-set must parse to a mapping like '{Si: 2s2p1d}'"
            )
        return OrbitalIrrepConfig.from_dict(payload)
    raise ValueError(
        "Need either --reference-info-path or --orbital-set to define the orbital basis."
    )


def _build_snapshot_from_matrices(
    mats: dict[str, Any], *, positions, box, info=None
) -> Snapshot:
    return Snapshot(
        mats["hamiltonian"],
        mats["overlap"],
        mats["density"],
        positions=positions,
        box=box,
        info=info,
    )


def _filter_block_matrix_by_distance_analysis(
    block_matrix,
    *,
    positions: torch.Tensor,
    box: torch.Tensor | None,
    cutoff_radius: float,
):
    mask_dict: dict[str, torch.Tensor] = {}
    for key, edges in block_matrix.pair_edges.items():
        sx, sy, sz = edges[0], edges[1], edges[2]
        src, dst = edges[3], edges[4]
        shift = torch.stack([sx, sy, sz], dim=1).to(positions.dtype)
        if box is not None:
            disp = positions[dst] - positions[src] + shift @ box
        else:
            disp = positions[dst] - positions[src]
        dist = torch.linalg.norm(disp, dim=1)
        mask_dict[key] = dist <= float(cutoff_radius)
    return block_matrix._apply_edge_mask(mask_dict, drop_empty=True)


def _apply_analysis_cutoff_to_snapshot(
    snapshot: Snapshot,
    analysis_cutoff_radius: float | None,
    cfg: Config,
) -> Snapshot:
    if analysis_cutoff_radius is None:
        return snapshot
    trained_cutoff = getattr(cfg, "cutoff_radius", None)
    if trained_cutoff is not None and analysis_cutoff_radius > float(trained_cutoff):
        raise ValueError(
            "analysis cutoff must not exceed the trained cutoff: "
            f"analysis_cutoff_radius={analysis_cutoff_radius} > trained_cutoff={trained_cutoff}"
        )
    if snapshot.positions is None:
        raise ValueError("analysis cutoff requires snapshot positions")

    cutoff_radius = float(analysis_cutoff_radius)
    filtered_h = _filter_block_matrix_by_distance_analysis(
        snapshot.hamiltonian,
        positions=snapshot.positions,
        box=snapshot.box,
        cutoff_radius=cutoff_radius,
    )
    filtered_s = _filter_block_matrix_by_distance_analysis(
        snapshot.overlap,
        positions=snapshot.positions,
        box=snapshot.box,
        cutoff_radius=cutoff_radius,
    )
    filtered_d = _filter_block_matrix_by_distance_analysis(
        snapshot.density,
        positions=snapshot.positions,
        box=snapshot.box,
        cutoff_radius=cutoff_radius,
    )
    return Snapshot(
        filtered_h,
        filtered_s,
        filtered_d,
        positions=snapshot.positions,
        forces=snapshot.forces,
        box=snapshot.box,
        stress=snapshot.stress,
        matrix_path=snapshot.matrix_path,
        info_path=snapshot.info_path,
        cutoff_radius=cutoff_radius,
        cfg=snapshot.cfg,
        info=snapshot.info,
    )


def _maybe_apply_analysis_cutoff(
    snapshot: Snapshot, analysis_cutoff_radius: float | None, cfg: Config
) -> Snapshot:
    return _apply_analysis_cutoff_to_snapshot(
        snapshot,
        analysis_cutoff_radius,
        cfg,
    )


def _run_snapshot_case(
    args: argparse.Namespace, checkpoint: dict[str, Any], cfg: Config
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path, args.matrix_path, args.info_path
    )

    cfg.dataset_device = None
    cfg.snapshot_cache_dir = _resolve_snapshot_cache_dir(cfg, output_dir)
    factory = DatasetFactory(cfg, convention=args.convention)
    factory.add_snapshot(matrix_path, info_path, purpose="train")
    dataset, _, mapper = factory.create()
    x, y = dataset[0]
    device = _resolve_device(args.device)
    x = _move_to_device(x, device)
    y = _move_to_device(y, device)

    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions_irreps = model(x)
        if "overlap" in predictions_irreps:
            predictions_irreps["overlap"] = (
                analysis_eval.clean_predicted_overlap_irreps(
                    predictions_irreps["overlap"], model.mapper
                )
            )
        pred_mats = {
            name: analysis_eval.symmetrize_block_matrix(pred.to_blocks(model.mapper))
            for name, pred in predictions_irreps.items()
        }

    gt_mats = {name: y[name] for name in ("hamiltonian", "density", "overlap")}
    info = parse_info_out(info_path)
    positions = x["positions"]
    box = x["box"]
    gt_snapshot = _build_snapshot_from_matrices(
        gt_mats, positions=positions, box=box, info=info
    )
    band_info_path = (
        args.band_info_path if args.band_info_path is not None else info_path
    )
    special_points_override = _parse_special_points_json(args.special_points_json)
    resolved_path_string, special_points = analysis_eval.resolve_band_path(
        band_info_path, args.path_string, special_points_override
    )
    gt_snapshot = _maybe_apply_analysis_cutoff(
        gt_snapshot, args.analysis_cutoff_radius, cfg
    )
    gt_mats = {
        name: gt_snapshot[name] for name in ("hamiltonian", "density", "overlap")
    }

    density_for_eigs = pred_mats.get("density", gt_mats["density"])
    overlap_for_eigs = (
        gt_mats["overlap"] if args.use_gt_overlap_for_eigs else pred_mats.get("overlap")
    )
    if overlap_for_eigs is None:
        raise ValueError(
            "Checkpoint does not predict overlap and --use-gt-overlap-for-eigs was not set."
        )
    pred_band_snapshot = _build_snapshot_from_matrices(
        {
            "hamiltonian": pred_mats["hamiltonian"],
            "density": density_for_eigs,
            "overlap": overlap_for_eigs,
        },
        positions=positions,
        box=box,
        info=info,
    )
    pred_band_snapshot = _maybe_apply_analysis_cutoff(
        pred_band_snapshot, args.analysis_cutoff_radius, cfg
    )

    title = args.plot_title or matrix_path.parent.name
    ham_clim = (
        args.hamiltonian_clim
        if args.hamiltonian_clim is not None
        else (args.plot_clim if args.plot_clim is not None else 0.05)
    )
    density_clim = (
        args.density_clim
        if args.density_clim is not None
        else (args.plot_clim if args.plot_clim is not None else 0.1)
    )
    analysis_eval.save_comparison_plot(
        pred_mats["hamiltonian"],
        gt_mats["hamiltonian"],
        output_dir / "hamiltonian_first_atoms_comparison.png",
        title=f"Hamiltonian comparison: {title}",
        max_atoms=args.max_atoms,
        clim=ham_clim,
    )
    analysis_eval.save_hamiltonian_interactive_heatmap_payload(
        pred_mats["hamiltonian"],
        gt_mats["hamiltonian"],
        positions=positions,
        box=box,
        output_path=output_dir / "hamiltonian_interactive_heatmaps.pt",
        default_clim=ham_clim,
        max_nodes=6,
    )
    analysis_eval.save_snapshot_3d_error_payload(
        pred_mats["hamiltonian"],
        gt_mats["hamiltonian"],
        positions=positions,
        box=box,
        output_path=output_dir / "snapshot_3d_error_payload.pt",
    )
    analysis_eval.save_correlation_plot(
        pred_mats["hamiltonian"],
        gt_mats["hamiltonian"],
        output_dir / "hamiltonian_correlation.png",
        title=f"Hamiltonian correlation: {title}",
        max_points=args.correlation_max_points,
        alpha=args.correlation_alpha,
        seed=args.correlation_sample_seed,
    )
    analysis_eval.save_block_error_scatter_data(
        pred_mats["hamiltonian"],
        gt_mats["hamiltonian"],
        positions=positions,
        box=box,
        output_path=output_dir / "hamiltonian_block_error_metrics.pt",
    )
    if "density" in pred_mats:
        analysis_eval.save_comparison_plot(
            pred_mats["density"],
            gt_mats["density"],
            output_dir / "density_first_atoms_comparison.png",
            title=f"Density comparison: {title}",
            max_atoms=args.max_atoms,
            clim=density_clim,
        )
        analysis_eval.save_correlation_plot(
            pred_mats["density"],
            gt_mats["density"],
            output_dir / "density_correlation.png",
            title=f"Density correlation: {title}",
            max_points=args.correlation_max_points,
            alpha=args.correlation_alpha,
            seed=args.correlation_sample_seed,
        )
        analysis_eval.save_block_error_scatter_data(
            pred_mats["density"],
            gt_mats["density"],
            positions=positions,
            box=box,
            output_path=output_dir / "density_block_error_metrics.pt",
        )
    else:
        print(
            "--- Density prediction unavailable; skipping density comparison plot ---"
        )
    if "overlap" in pred_mats:
        analysis_eval.save_correlation_plot(
            pred_mats["overlap"],
            gt_mats["overlap"],
            output_dir / "overlap_correlation.png",
            title=f"Overlap correlation: {title}",
            max_points=args.correlation_max_points,
            alpha=args.correlation_alpha,
            seed=args.correlation_sample_seed,
        )

    num_electrons_true = float(gt_snapshot.get_number_of_electrons().item())
    num_electrons_pred = (
        num_electrons_true
        if args.use_gt_overlap_for_eigs
        else float(pred_band_snapshot.get_number_of_electrons().item())
    )
    if args.dos_method == "tetrahedron":
        dos_metrics = analysis_eval.save_tetrahedron_dos_comparison_plot(
            gt_snapshot,
            pred_band_snapshot,
            output_dir / "dos_comparison.png",
            kmesh_spec=args.dos_kmesh,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            energy_min=args.dos_energy_min,
            energy_max=args.dos_energy_max,
            title=f"DOS comparison: {title}",
            error_output_path=output_dir / "dos_error.png",
            overlap_psd_cleanup=args.overlap_psd_cleanup,
            overlap_jitter=args.overlap_jitter,
            bin_width=args.dos_bin_width,
            tetra_batch_size=args.tetra_batch_size,
            cache_path_true=output_dir / "tetrahedron_dos_cache_gt.pt",
            cache_path_pred=output_dir / "tetrahedron_dos_cache_pred.pt",
        )
    else:
        dos_metrics = analysis_eval.save_dos_comparison_plot(
            pred_mats["hamiltonian"],
            pred_band_snapshot.overlap,
            gt_mats["hamiltonian"],
            gt_mats["overlap"],
            num_electrons_true,
            num_electrons_pred,
            output_dir / "dos_comparison.png",
            sigma=args.dos_sigma,
            bin_width=args.dos_bin_width,
            energy_min=args.dos_energy_min,
            energy_max=args.dos_energy_max,
            title=f"DOS comparison: {title}",
            error_output_path=output_dir / "dos_error.png",
            overlap_psd_cleanup=args.overlap_psd_cleanup,
            overlap_jitter=args.overlap_jitter,
        )

    gt_band = analysis_eval.compute_or_load_band_structure(
        gt_snapshot,
        analysis_eval.band_cache_path(
            output_dir,
            kind="gt",
            use_gt_overlap_for_eigs=args.use_gt_overlap_for_eigs,
        ),
        path_string=resolved_path_string,
        special_points=special_points,
        num_points=args.num_points,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
        force_recompute=args.force_recompute_bands,
    )
    pred_band = analysis_eval.compute_or_load_band_structure(
        pred_band_snapshot,
        analysis_eval.band_cache_path(
            output_dir,
            kind="pred",
            use_gt_overlap_for_eigs=args.use_gt_overlap_for_eigs,
        ),
        path_string=resolved_path_string,
        special_points=special_points,
        num_points=args.num_points,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
        force_recompute=args.force_recompute_bands,
    )
    analysis_eval.save_band_structure_comparison_plot(
        gt_band,
        pred_band,
        output_dir / "band_structure_comparison.png",
        title=f"Band structure comparison: {title}",
        emin_ev=args.band_emin_ev,
        emax_ev=args.band_emax_ev,
        line_alpha=args.band_line_alpha,
    )

    for name, block in pred_mats.items():
        block.save(output_dir / f"pred_{name}.pt")
    print("mode: snapshot")
    print("checkpoint:", args.checkpoint)
    print("matrix_path:", matrix_path)
    print("info_path:", info_path)
    print("output_dir:", output_dir)
    print("dos_method:", args.dos_method)
    print("dos_kmesh:", args.dos_kmesh)
    print("dos_metrics:", dos_metrics)


def _run_cif_case(
    args: argparse.Namespace, checkpoint: dict[str, Any], cfg: Config
) -> None:
    if args.cif_path is None:
        raise ValueError("--cif-path is required in cif mode.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    orbital_cfg = _resolve_orbital_cfg(args, cfg)
    mapper = BlockIrrepMapper(
        orbital_cfg,
        diagonal=False,
        device="cpu",
        dtype=cfg.dtype,
    )
    atoms, positions, box = load_structure_from_cif(
        args.cif_path,
        dtype=cfg.dtype,
        device=device,
    )
    if args.reference_info_path is None:
        raise ValueError(
            "CIF mode requires --reference-info-path so the OpenMX Band.kpath can be reused."
        )
    band_info_path = (
        args.band_info_path
        if args.band_info_path is not None
        else args.reference_info_path
    )
    special_points_override = _parse_special_points_json(args.special_points_json)
    resolved_path_string, special_points = analysis_eval.resolve_band_path(
        band_info_path, args.path_string, special_points_override
    )
    x = build_model_input_from_structure(
        atoms=atoms,
        positions=positions,
        box=box,
        cfg=cfg,
        mapper=mapper,
    )
    model = E3GNN(mapper=mapper, cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions_irreps = model(x)
        if "overlap" in predictions_irreps:
            predictions_irreps["overlap"] = (
                analysis_eval.clean_predicted_overlap_irreps(
                    predictions_irreps["overlap"], model.mapper
                )
            )
        pred_mats = {
            name: analysis_eval.symmetrize_block_matrix(pred.to_blocks(model.mapper))
            for name, pred in predictions_irreps.items()
        }

    title = args.plot_title or args.cif_path.stem
    ham_clim = (
        args.hamiltonian_clim
        if args.hamiltonian_clim is not None
        else (args.plot_clim if args.plot_clim is not None else 0.05)
    )
    density_clim = (
        args.density_clim
        if args.density_clim is not None
        else (args.plot_clim if args.plot_clim is not None else 0.1)
    )
    analysis_eval.save_prediction_plot(
        pred_mats["hamiltonian"],
        output_dir / "hamiltonian_first_atoms_prediction.png",
        title=f"Hamiltonian prediction: {title}",
        max_atoms=args.cif_max_atoms,
        clim=ham_clim,
    )
    if "density" in pred_mats:
        analysis_eval.save_prediction_plot(
            pred_mats["density"],
            output_dir / "density_first_atoms_prediction.png",
            title=f"Density prediction: {title}",
            max_atoms=args.cif_max_atoms,
            clim=density_clim,
        )
    else:
        raise ValueError(
            "CIF evaluation currently requires a predicted density matrix for DOS/band plots."
        )

    pred_snapshot_for_eigs = _build_snapshot_from_matrices(
        {
            "hamiltonian": pred_mats["hamiltonian"],
            "density": pred_mats["density"],
            "overlap": pred_mats.get("overlap"),
        },
        positions=positions,
        box=box,
    )
    pred_snapshot_for_eigs = _maybe_apply_analysis_cutoff(
        pred_snapshot_for_eigs, args.analysis_cutoff_radius, cfg
    )
    if pred_snapshot_for_eigs.overlap is None:
        raise ValueError(
            "CIF evaluation needs a predicted overlap matrix for DOS/band plots."
        )
    num_electrons_pred = float(pred_snapshot_for_eigs.get_number_of_electrons().item())
    if args.dos_method == "tetrahedron":
        analysis_eval.save_tetrahedron_dos_prediction_plot(
            pred_snapshot_for_eigs,
            output_dir / "dos_prediction.png",
            kmesh_spec=args.dos_kmesh,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            energy_min=args.dos_energy_min,
            energy_max=args.dos_energy_max,
            num_electrons=num_electrons_pred,
            title=f"DOS prediction: {title}",
            overlap_psd_cleanup=args.overlap_psd_cleanup,
            overlap_jitter=args.overlap_jitter,
            bin_width=args.dos_bin_width,
            tetra_batch_size=args.tetra_batch_size,
            cache_path=output_dir / "tetrahedron_dos_cache_pred.pt",
        )
    else:
        analysis_eval.save_dos_prediction_plot(
            pred_mats["hamiltonian"],
            pred_snapshot_for_eigs.overlap,
            output_dir / "dos_prediction.png",
            sigma=args.dos_sigma,
            bin_width=args.dos_bin_width,
            energy_min=args.dos_energy_min,
            energy_max=args.dos_energy_max,
            num_electrons=num_electrons_pred,
            title=f"DOS prediction: {title}",
            overlap_psd_cleanup=args.overlap_psd_cleanup,
            overlap_jitter=args.overlap_jitter,
        )
    pred_band = analysis_eval.compute_or_load_band_structure(
        pred_snapshot_for_eigs,
        analysis_eval.band_cache_path(
            output_dir,
            kind="pred",
            use_gt_overlap_for_eigs=False,
        ),
        path_string=resolved_path_string,
        special_points=special_points,
        num_points=args.num_points,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
        force_recompute=args.force_recompute_bands,
    )
    analysis_eval.save_band_structure_prediction_plot(
        pred_band,
        output_dir / "band_structure_prediction.png",
        title=f"Band structure prediction: {title}",
        emin_ev=args.band_emin_ev,
        emax_ev=args.band_emax_ev,
        line_alpha=args.band_line_alpha,
    )
    torch.save(
        {
            "atoms": list(atoms),
            "positions": positions.detach().cpu(),
            "box": box.detach().cpu(),
            "orbital_cfg": orbital_cfg.to_dict(),
            "checkpoint": str(args.checkpoint),
            "cif_path": str(args.cif_path),
            "matrix_targets": list(cfg.matrix_targets),
        },
        output_dir / "structure_metadata.pt",
    )
    if args.save_input:
        torch.save(_cpu_copy(x), output_dir / "model_input.pt")
    for name, block in pred_mats.items():
        block.save(output_dir / f"pred_{name}.pt")
    print("mode: cif")
    print("checkpoint:", args.checkpoint)
    print("cif_path:", args.cif_path)
    print("output_dir:", output_dir)
    print("dos_method:", args.dos_method)
    print("dos_kmesh:", args.dos_kmesh)


def main() -> None:
    args = setup_argparse()
    checkpoint = _load_checkpoint(args.checkpoint)
    cfg = _restore_config(checkpoint)
    cfg.dataset_device = None
    cfg.snapshot_cache_dir = None

    if "hamiltonian" not in set(cfg.matrix_targets):
        raise ValueError(
            f"Checkpoint matrix_targets={cfg.matrix_targets!r} do not include hamiltonian."
        )

    if args.mode == "snapshot":
        _run_snapshot_case(args, checkpoint, cfg)
    else:
        _run_cif_case(args, checkpoint, cfg)


if __name__ == "__main__":
    main()
