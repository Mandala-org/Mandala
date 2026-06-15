#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis import evaluation as analysis_eval  # noqa: E402
from data.kspace_snapshot import build_band_path  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402


def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and plot the ground-truth band structure for a snapshot."
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path("data/big/silicon/300K"),
        help="Snapshot directory containing Si_DM and info.dat.",
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=None,
        help="Optional explicit matrix path. Overrides --snapshot-path discovery.",
    )
    parser.add_argument(
        "--info-path",
        type=Path,
        default=None,
        help="Optional explicit info path. Overrides --snapshot-path discovery.",
    )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/ground_truth_silicon_300K_band_structure"),
        help="Directory for the plot and serialized band-structure payload.",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        help="Matrix convention used when loading the snapshot.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=240,
        help="Number of interpolated k-points along the path.",
    )
    parser.add_argument(
        "--path-string",
        type=str,
        default=None,
        help="Band path string in fractional reciprocal coordinates.",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default="Ground-truth band structure",
        help="Figure title.",
    )
    parser.add_argument(
        "--emin-ev",
        type=float,
        default=-10.0,
        help="Lower plot bound in eV after subtracting the Fermi level.",
    )
    parser.add_argument(
        "--emax-ev",
        type=float,
        default=10.0,
        help="Upper plot bound in eV after subtracting the Fermi level.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Number of k-points processed per chunk during the band calculation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes used for band-structure chunk solving.",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.8,
        help="Opacity of the band lines in the plot.",
    )
    parser.add_argument(
        "--dos-sigma",
        type=float,
        default=0.1,
        help="Gaussian broadening sigma in eV for the DOS plot.",
    )
    parser.add_argument(
        "--dos-bin-width",
        type=float,
        default=0.05,
        help="Energy grid spacing in eV for the tetrahedron DOS.",
    )
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
        choices=["tetrahedron", "kmesh-average", "gaussian"],
        help="DOS construction method.",
    )
    parser.add_argument(
        "--dos-kmesh",
        type=str,
        default="4x4x4",
        help="Uniform k-point grid for DOS averaging, e.g. 4x4x4.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached band_structure.pt and recompute the expensive band structure.",
    )
    parser.add_argument(
        "--overlap-psd-cleanup",
        action="store_true",
        help="Enable overlap PSD cleanup before the generalized eigensolve.",
    )
    parser.add_argument(
        "--overlap-jitter",
        action="store_true",
        help="Allow diagonal jitter retries if the overlap Cholesky fails.",
    )
    return parser.parse_args()


def _discover_snapshot_paths(
    snapshot_path: Path, matrix_path: Path | None, info_path: Path | None
) -> tuple[Path, Path]:
    if matrix_path is not None and info_path is not None:
        return matrix_path, info_path
    if matrix_path is not None or info_path is not None:
        raise ValueError("Provide both --matrix-path and --info-path together.")
    matrix_candidate = snapshot_path / "Si_DM"
    info_candidate = snapshot_path / "info.dat"
    if not matrix_candidate.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_candidate}")
    if not info_candidate.exists():
        raise FileNotFoundError(f"Info file not found: {info_candidate}")
    return matrix_candidate, info_candidate


def _log(message: str) -> None:
    print(message, flush=True)


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    return f"{minutes}m {remainder:.1f}s"


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


def main() -> None:
    t0 = time.perf_counter()
    args = setup_argparse()
    matrix_path, info_path = _discover_snapshot_paths(
        args.snapshot_path,
        args.matrix_path,
        args.info_path,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("=== Band structure evaluation starting ===")
    _log(f"matrix_path: {matrix_path}")
    _log(f"info_path: {info_path}")
    _log(f"output_dir: {output_dir}")
    _log(f"path_string: {args.path_string}")
    _log(f"num_points: {args.num_points}")
    _log(f"chunk_size: {args.chunk_size}")
    _log(f"num_workers: {args.num_workers}")
    _log(f"line_alpha: {args.line_alpha}")
    _log(f"dos_method: {args.dos_method}")
    _log(f"dos_kmesh: {args.dos_kmesh}")
    _log(f"dos_sigma_ev: {args.dos_sigma}")
    _log(f"dos_bin_width_ev: {args.dos_bin_width}")

    t_load = time.perf_counter()
    _log("[1/6] Loading snapshot ...")
    snapshot = Snapshot.from_openmx(
        matrix_path,
        info_path,
        convention=args.convention,
    )
    _log(
        "[1/6] Loaded snapshot in "
        f"{_format_seconds(time.perf_counter() - t_load)} "
        f"(atoms={len(snapshot.hamiltonian.atoms)}, basis={snapshot.hamiltonian.basis})"
    )

    t_path = time.perf_counter()
    _log("[2/6] Building k-path ...")
    special_points_override = _parse_special_points_json(args.special_points_json)
    path_string, special_points = analysis_eval.resolve_band_path(
        box=snapshot.box,
        info_path=args.band_info_path,
        requested_path_string=args.path_string,
        special_points_override=special_points_override,
    )
    (
        _resolved_path_string,
        fractional_kpoints,
        kpoints_abs,
        linear_k,
        tick_positions,
        tick_labels,
    ) = build_band_path(
        snapshot.box,
        path=path_string,
        special_points=special_points,
        npoints=args.num_points,
    )
    _log(
        "[2/6] Built k-path in "
        f"{_format_seconds(time.perf_counter() - t_path)} "
        f"(num_kpoints={kpoints_abs.shape[0]}, labels={tick_labels})"
    )

    t_shifts = time.perf_counter()
    _log("[3/6] Collecting translation shifts ...")
    shifts = snapshot.get_translation_shifts().to(device=snapshot.box.device)
    _log(
        "[3/6] Collected shifts in "
        f"{_format_seconds(time.perf_counter() - t_shifts)} "
        f"(num_shifts={shifts.shape[0]})"
    )

    t_dos = time.perf_counter()
    _log("[4/6] Computing DOS and Fermi level ...")
    if args.dos_method == "tetrahedron":
        grid_ev, dos, num_electrons, dos_electron_target, fermi_level_ev = (
            analysis_eval.compute_tetrahedron_dos_and_fermi(
                snapshot,
                kmesh_spec=args.dos_kmesh,
                chunk_size=args.chunk_size,
                num_workers=args.num_workers,
                psd_cleanup=args.overlap_psd_cleanup,
                allow_jitter=args.overlap_jitter,
                bin_width=args.dos_bin_width,
                tetra_batch_size=args.tetra_batch_size,
                cache_path=output_dir / "tetrahedron_dos_cache.pt",
            )
        )
        _log("[4/6] Using tetrahedron DOS on a uniform k-mesh.")
    elif args.dos_method == "kmesh-average":
        grid_ev, dos, num_electrons, dos_electron_target, fermi_level_ev = (
            analysis_eval.compute_kmesh_average_dos_and_fermi(
                snapshot,
                kmesh_spec=args.dos_kmesh,
                chunk_size=args.chunk_size,
                num_workers=args.num_workers,
                psd_cleanup=args.overlap_psd_cleanup,
                allow_jitter=args.overlap_jitter,
                dos_sigma_ev=args.dos_sigma,
            )
        )
        _log("[4/6] Using k-mesh-averaged eigenvalue DOS.")
    else:
        grid_ev, dos, num_electrons, dos_electron_target, fermi_level_ev = (
            analysis_eval.compute_gaussian_dos_from_eigenvalues_and_fermi(
                snapshot,
                psd_cleanup=args.overlap_psd_cleanup,
                allow_jitter=args.overlap_jitter,
                dos_sigma_ev=args.dos_sigma,
            )
        )
    dos_path = output_dir / "dos.png"
    analysis_eval.save_dos_plot(
        grid_ev,
        dos,
        output_path=dos_path,
        title=f"DOS: {args.plot_title}",
        num_electrons=num_electrons,
        fermi_level_ev=fermi_level_ev,
        energy_min=args.emin_ev,
        energy_max=args.emax_ev,
    )
    _log(
        "[4/6] Computed DOS in "
        f"{_format_seconds(time.perf_counter() - t_dos)} "
        f"(N_e={num_electrons:.3f}, DOS_target={dos_electron_target:.3f}, E_F={fermi_level_ev:.3f} eV)"
    )

    dos_reference = None
    dos_reference_path = info_path.with_name(f"{info_path.stem}.DOS.Tetrahedron")
    if dos_reference_path.exists():
        _log(f"[4/6] Loading DOS reference from {dos_reference_path} ...")
        dos_reference = analysis_eval.load_dos_reference(dos_reference_path)

    cache_path = analysis_eval.band_cache_path(
        output_dir, kind="snapshot", use_gt_overlap_for_eigs=False
    )
    t_band = time.perf_counter()
    _log("[5/6] Computing chunked band structure ...")
    band_structure = analysis_eval.compute_or_load_band_structure(
        snapshot,
        cache_path,
        path_string=path_string,
        special_points=special_points,
        num_points=args.num_points,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        overlap_psd_cleanup=args.overlap_psd_cleanup,
        overlap_jitter=args.overlap_jitter,
        force_recompute=args.force_recompute,
    )
    _log(
        "[5/6] Loaded/computed band structure in "
        f"{_format_seconds(time.perf_counter() - t_band)} "
        f"(num_bands={band_structure.eigenvalues.shape[1]})"
    )
    _log(f"[5/6] Band structure cache at {cache_path}")

    plot_path = output_dir / "band_structure.png"
    t_plot = time.perf_counter()
    _log("[6/6] Saving plot and serialized payload ...")
    analysis_eval.save_band_structure_plot(
        band_structure,
        plot_path,
        title=args.plot_title,
        emin_ev=args.emin_ev,
        emax_ev=args.emax_ev,
        line_alpha=args.line_alpha,
    )
    band_dos_path = output_dir / "band_structure_with_dos.png"
    analysis_eval.save_band_and_dos_plot(
        band_structure,
        dos_grid=grid_ev,
        dos=dos,
        dos_reference=dos_reference,
        output_path=band_dos_path,
        title=f"{args.plot_title} (band + DOS)",
        emin_ev=args.emin_ev,
        emax_ev=args.emax_ev,
        line_alpha=args.line_alpha,
        num_electrons=num_electrons,
        fermi_level_ev=fermi_level_ev,
    )
    _log("[6/6] Saved outputs in " f"{_format_seconds(time.perf_counter() - t_plot)}")

    _log("=== Band structure evaluation finished ===")
    _log(f"plot_path: {plot_path}")
    _log(f"band_dos_path: {band_dos_path}")
    _log(f"dos_path: {dos_path}")
    _log(f"num_kpoints: {int(band_structure.eigenvalues.shape[0])}")
    _log(f"num_bands: {int(band_structure.eigenvalues.shape[1])}")
    if fermi_level_ev is not None:
        _log(f"fermi_level_ev: {fermi_level_ev:.6f}")
    _log(f"num_electrons: {num_electrons:.6f}")
    if dos_electron_target is not None:
        _log(f"dos_electron_target: {dos_electron_target:.6f}")
    _log(f"total_runtime: {_format_seconds(time.perf_counter() - t0)}")


if __name__ == "__main__":
    main()
