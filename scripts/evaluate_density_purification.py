"""Evaluate frozen density-prediction artifacts; no inference or model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from analysis.density_purification import evaluate_purification  # noqa: E402
from data.block_matrix import BlockMatrix  # noqa: E402
from data.snapshot import Snapshot  # noqa: E402
from net.common import Config  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    torch.set_num_threads(args.threads)
    cases = json.loads(args.manifest.read_text())["cases"]
    if not cases:
        raise ValueError("The evaluation manifest must contain at least one case.")
    results = []
    reference_cache = {}
    for case in cases:
        directory = Path(case["prediction_dir"])
        density_path = directory / "pred_density.pt"
        if not density_path.exists():
            raise FileNotFoundError(f"Density predictions required: {density_path}")
        scale = case["saved_density_scale"]
        if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("saved_density_scale must be finite and positive.")
        d = scale * BlockMatrix.load(density_path)
        h_path = directory / "pred_hamiltonian.pt"
        h = BlockMatrix.load(h_path) if h_path.exists() else None
        reference_key = (case["matrix_path"], case["info_path"])
        if reference_key not in reference_cache:
            reference_cache[reference_key] = Snapshot.from_openmx(
                *reference_key, convention="e3nn", cfg=Config(verbosity=0)
            )
        reference = reference_cache[reference_key]
        for overlap_source in case["overlap_sources"]:
            if overlap_source == "reference":
                s = reference.overlap
            elif overlap_source == "predicted":
                s = BlockMatrix.load(directory / "pred_overlap.pt")
            else:
                raise ValueError(f"Unknown overlap source {overlap_source!r}")
            print(f"Evaluating {case['label']} / {overlap_source} overlap", flush=True)
            with torch.no_grad():
                result = evaluate_purification(
                    d,
                    s,
                    reference.density,
                    reference.overlap,
                    reference.hamiltonian,
                    predicted_hamiltonian=h,
                    chunk_size=args.chunk_size,
                    spin_degeneracy=case["spin_degeneracy"],
                )
            input_paths = [
                density_path,
                Path(case["matrix_path"]),
                Path(case["info_path"]),
            ]
            cif_path = Path(case["info_path"]).with_suffix(".cif")
            if cif_path.exists():
                input_paths.append(cif_path)
            if h_path.exists():
                input_paths.append(h_path)
            if overlap_source == "predicted":
                input_paths.append(directory / "pred_overlap.pt")
            result.update(
                case=case,
                overlap_source=overlap_source,
                input_sha256={str(p): sha256(p) for p in input_paths},
            )
            results.append(result)
            print(
                [
                    (r["iteration"], r["physical_density"]["mae"])
                    for r in result["rows"]
                ],
                flush=True,
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        formula="X=P(SD); Y=P(DX); Z=P(YX); D_next=3Y-2Z",
        recurrence="Raw polynomial iterates; global Hermitian projection for evaluation only",
        dtype="float64",
        threads=args.threads,
        manifest_sha256=sha256(args.manifest),
        source_sha256={
            path: sha256(ROOT / path)
            for path in (
                "src/core/sparse_math.py",
                "src/core/density_purification.py",
                "src/analysis/density_purification.py",
                "src/data/openmx_parser.py",
                "src/data/snapshot.py",
                "scripts/evaluate_density_purification.py",
            )
        },
        results=results,
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# McWeeny density purification: perturbed Silicon",
        "",
        "Saved trained-model predictions, without re-inference or retraining. "
        "All 0/1/2/3 steps were specified before evaluation. SiOx excluded at the user's request.",
        "",
        "Each multiplication is truncated to the original density edges, including periodic shifts. "
        "This is an approximation to the untruncated McWeeny polynomial. Raw iterates feed the "
        "next step; global reverse-pair Hermitian projection is applied only for the reported "
        "physical metrics. Legacy saved density predictions are explicitly multiplied by "
        "the manifest's saved_density_scale (0.5 for these old doubled-target checkpoints) "
        "to match the corrected OpenMX loader, 0.5(D + Dᵀ). No per-iteration normalization "
        "or electron-count correction is applied.",
        "",
        "Density errors are element-weighted over the union of reference/prediction periodic "
        "edges, treating missing blocks as zero. Band energy is g Tr(D H), with explicit "
        "spin degeneracy g=2 for these nonmagnetic Silicon cases; reference H isolates the density contribution, while predicted "
        "H measures the joint H/D prediction. Energies in the tables are absolute errors in eV "
        "per cell, not band eigenvalue errors.",
        "",
    ]
    improved = sum(
        row["physical_density"]["mae"] < result["rows"][0]["physical_density"]["mae"]
        for result in results
        for row in result["rows"][1:]
    )
    lines[2:2] = [
        f"Result: {improved} of {3 * len(results)} nonzero-step variants improve density MAE "
        "over their respective unpurified baseline. See the reference-density control "
        "alongside every comparison.",
        "",
    ]
    for result in results:
        lines += [
            f"## {result['case']['label']} — {result['overlap_source']} overlap",
            "",
            f"{result['atoms']} atoms, {result['edges']} directed edges; reference Tr(DS) = "
            f"{result['reference_occupation_trace']:.8g} (g={result['spin_degeneracy']}). "
            f"Topology setup {result['plan_seconds']:.3f} s, paths SD/DD "
            f"{result['sd_paths']}/{result['dd_paths']}.",
            "",
            "| Steps | Density MAE | Density RMSE | MAE improvement | ΔE (reference H), eV | ΔE (predicted H), eV | Tr(DS) | Reference-control MAE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in result["rows"]:
            lines.append(
                f"| {row['iteration']} | {row['physical_density']['mae']:.7g} | "
                f"{row['physical_density']['rmse']:.7g} | {row['density_mae_improvement_percent']:.3f}% | "
                f"{row['band_energy_ref_h_abs_error_ev']:.7g} | "
                f"{row.get('band_energy_pred_h_abs_error_ev', float('nan')):.7g} | "
                f"{row['occupation_trace']:.7g} | {row['reference_control_density']['mae']:.7g} |"
            )
        lines += [
            "",
            f"Source: `{result['case']['prediction_dir']}`; reference `{result['case']['matrix_path']}`.",
            "",
        ]
    lines += [
        "## Interpretation and scope",
        "",
        "The polynomial f(x)=3x²−2x³ targets occupations 0 and 1. It does not conserve "
        "Tr(DS); f(x)<0 for x>1.5. The corrected loader retains the native spin=0 "
        "density normalization, independently of the spin factor used for observable reporting. "
        "The reference-density control measures how the exact same truncated "
        "operation changes ground truth itself.",
        "",
        "These are available local snapshot artifacts, not a complete validation/test-set "
        "benchmark. Established H/S/D checkpoints were included without selecting models "
        "or iteration counts using these results. The supplied manifest and artifact hashes "
        "identify every input. Per-iterate raw errors, Hermiticity residuals, timings, and "
        "electron-count errors are retained in results.json.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
