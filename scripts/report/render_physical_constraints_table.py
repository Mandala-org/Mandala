#!/usr/bin/env python3
"""Render the historical paper audit, without importing the ML package.

Selected aggregate keys: silicon_historical/joint_predicted_reference_support;
sio2_historical/predicted_H_reference_SD_reference_support. These are reference-
support controls (discard prediction-only blocks), with Hermitian projection
inside diagnose_kpoint. They are NOT overlap repair or occupation clipping.
reference_full is the paired baseline. Zn uses only historical_summary.json's
matrix_errors_hermitian_common_support/hamiltonian, never spectral diagnostics.
No raw_hermiticity or matrix_errors_raw_common_support values are reported.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

DEFAULT = (
    Path(__file__).resolve().parents[2] / "docs/data/physical_constraints_20261104"
)
SELECTION = {
    "silicon": "joint_predicted_reference_support",
    "sio2": "predicted_H_reference_SD_reference_support",
}
# Worst means minimum for lower spectral bounds and success fractions.
DIAGNOSTICS = {
    "S": [
        ("overlap.positive_definite", "min"),
        ("overlap.well_conditioned_positive_definite", "min"),
        ("overlap.cholesky_failed_without_jitter", "max"),
        ("overlap.lambda_min", "min"),
        ("overlap.lambda_max", "max"),
        ("overlap.condition_number", "max"),
        ("overlap.nonpositive_eigenvalues", "max"),
    ],
    "D": [
        ("density.occupation_min", "min"),
        ("density.occupation_max", "max"),
        ("density.negative_count", "max"),
        ("density.overoccupied_count", "max"),
        ("electron_count_absolute_error", "max"),
        ("density.zero_temperature_idempotency_relative", "max"),
    ],
    "H": [
        ("consistency.commutator_relative", "max"),
        ("finite_temperature_density_relative_frobenius", "max"),
    ],
}


LABELS = {
    "overlap.positive_definite": "S positive definite",
    "overlap.well_conditioned_positive_definite": "S usable metric",
    "overlap.cholesky_failed_without_jitter": "Cholesky failure",
    "overlap.lambda_min": "S min eigenvalue",
    "overlap.lambda_max": "S max eigenvalue",
    "overlap.condition_number": "S condition number",
    "overlap.nonpositive_eigenvalues": "S nonpositive eigenvalue count",
    "density.occupation_min": "Min occupation",
    "density.occupation_max": "Max occupation",
    "density.negative_count": "Negative occupation count",
    "density.overoccupied_count": "Excess occupation count",
    "electron_count_absolute_error": "Electron count absolute error",
    "density.zero_temperature_idempotency_relative": "Zero-T idempotency residual",
    "consistency.commutator_relative": "Stationarity residual",
    "finite_temperature_density_relative_frobenius": "Fermi-Dirac residual",
}


def label(key):
    return LABELS.get(key, key.replace("projected common-support MAE", "Element MAE"))


def operator_label(material, operator):
    return operator + (
        " (reference)" if material == "sio2" and operator in ("S", "D") else ""
    )


def stats(scenario, key, worst, structures, contributing_structures=None):
    value = scenario.get("metrics", {}).get(key, scenario.get(key))
    per_structure = key in (
        "electron_count_absolute_error",
        "finite_temperature_density_relative_frobenius",
    )
    value = value or {"count": 0, "mean": None, worst: None}
    return dict(
        mean=value["mean"],
        worst=value[worst],
        count=value["count"],
        unit_of_count="structures" if per_structure else "kpoints",
        total_structures=structures,
        total_kpoints=scenario["kpoint_count"],
        contributing_structures=(
            value["count"] if per_structure else contributing_structures
        ),
    )


def build(directory):
    rows, sources, denominators = [], [], {}
    for material in ("silicon", "sio2", "zncusnses"):
        path = directory / material / "historical_summary.json"
        summary = json.loads(path.read_text())
        if (
            summary["status"] != "complete"
            or len(summary["results"]) != summary["structures"]
        ):
            raise ValueError(f"Incomplete historical summary: {path}")
        sources.append(str(path.relative_to(directory)))
        # Per-structure, common-support, Hermitian-projected element errors.
        for name, operator in [
            ("hamiltonian", "H"),
            ("overlap", "S"),
            ("density", "D"),
        ]:
            values = [
                r["matrix_errors_hermitian_common_support"][name]
                for r in summary["results"]
                if name in r["matrix_errors_hermitian_common_support"]
            ]
            if not values:
                continue
            unit = "meV" if operator == "H" else "AO units"
            errors = [
                v["mae_ev"] * 1000 if operator == "H" else v["mae"] for v in values
            ]
            rows.append(
                dict(
                    material=material,
                    operator=operator,
                    diagnostic=f"projected common-support MAE ({unit})",
                    reference=dict(
                        mean=0.0,
                        worst=0.0,
                        count=len(errors),
                        unit_of_count="structures",
                        total_structures=len(errors),
                        total_kpoints=0,
                        contributing_structures=len(errors),
                    ),
                    predicted=dict(
                        mean=mean(errors),
                        worst=max(errors),
                        count=len(errors),
                        unit_of_count="structures",
                        total_structures=len(errors),
                        total_kpoints=0,
                        contributing_structures=len(errors),
                    ),
                    scenario="matrix_errors_hermitian_common_support",
                    worst_direction="max",
                )
            )
        if material == "zncusnses":
            continue
        path = directory / material / f"{material}_aggregate.json"
        sources.append(str(path.relative_to(directory)))
        evaluations = json.loads(path.read_text())["evaluations"]
        selected = [
            v for k, v in evaluations.items() if k.endswith(f"/{material}_historical")
        ]
        if len(selected) != 1:
            raise ValueError(f"Ambiguous historical aggregate: {path}")
        evaluation = selected[0]
        selected_scenario = evaluation["scenarios"][SELECTION[material]]
        denominators[material] = dict(
            structures=evaluation["structures"],
            kpoints=selected_scenario["kpoint_count"],
            invalid_metric_kpoints=selected_scenario["invalid_kpoint_count"],
            structures_with_invalid_metric=selected_scenario[
                "structures_with_invalid_metric"
            ],
        )

        def metric_stats(label, key, direction):
            contributing = sum(
                r["scenarios"][label]["summary"]["metrics"].get(key, {}).get("count", 0)
                > 0
                for r in summary["results"]
            )
            return stats(
                evaluation["scenarios"][label],
                key,
                direction,
                evaluation["structures"],
                contributing,
            )

        for operator, diagnostics in DIAGNOSTICS.items():
            for key, direction in diagnostics:
                rows.append(
                    dict(
                        material=material,
                        operator=operator,
                        diagnostic=key,
                        reference=metric_stats("reference_full", key, direction),
                        predicted=metric_stats(SELECTION[material], key, direction),
                        scenario=SELECTION[material],
                        worst_direction=direction,
                    )
                )
    return dict(
        schema_version=1,
        sources=sources,
        scenarios=SELECTION,
        denominators=denominators,
        cleanup_status="Reference-support control and Hermitian projection only; "
        "no overlap repair, fixed-charge occupation projection, or post-cleanup aggregates exist.",
        rows=rows,
    )


def fmt(value):
    return "—" if value is None else f"{value:.6g}"


def cells(row):
    output = [
        row["material"],
        operator_label(row["material"], row["operator"]),
        label(row["diagnostic"]),
    ]
    for which in ("reference", "predicted"):
        s = row[which]
        output.extend([fmt(s["mean"]) if s else "—", fmt(s["worst"]) if s else "—"])
        output.append(
            f"{s['count']} {s['unit_of_count']}; {s['contributing_structures']}/{s['total_structures']} structures; {s['total_kpoints']} total kpoints"
            if s
            else "0 valid; unavailable"
        )
    return output


def operator_panels(result):
    """Keep material/operator rows and diagnostic columns in compact panels.

    Every mean/worst cell pairs reference / prediction; exact contributing
    sample and structure counts remain in the detailed denominator table.
    """
    panels = []
    for operator in ("H", "S", "D"):
        rows = [r for r in result["rows"] if r["operator"] == operator]
        keys = list(dict.fromkeys(r["diagnostic"] for r in rows))
        # Element errors stand alone, so Zn never acquires a spectral row.
        groups = [keys[:1]] + [keys[i : i + 2] for i in range(1, len(keys), 2)]
        for keys_in_panel in groups:
            header = ["Material", "Operator"]
            for key in keys_in_panel:
                header += [label(key) + " mean (R / P)", label(key) + " worst (R / P)"]
            table = []
            for material in ("silicon", "sio2", "zncusnses"):
                selected = {
                    r["diagnostic"]: r
                    for r in rows
                    if r["material"] == material and r["diagnostic"] in keys_in_panel
                }
                if not selected:
                    continue
                row = [material, operator_label(material, operator)]
                for key in keys_in_panel:
                    entry = selected.get(key)
                    for stat in ("mean", "worst"):
                        row.append(
                            " / ".join(
                                fmt(entry[side][stat]) if entry and entry[side] else "—"
                                for side in ("reference", "predicted")
                            )
                        )
                table.append(row)
            panels.append((header, table))
    return panels


def render(result, directory):
    intro = (
        "Historical July inference only. Selected scenarios: "
        + "; ".join(f"{k}: `{v}`" for k, v in SELECTION.items())
        + ".\n\n"
        + result["cleanup_status"]
        + "\n\n"
        "These spectral values use Hermitian-projected matrices on reference support, "
        "a control requiring reference sparsity, not independent deployment. SiO2 H uses "
        "reference S/D; silicon uses predicted H/S/D. Zn reports H elements only. "
        "Missing post-cleanup results are unavailable, not replaced with raw predictions.\n\n"
        "Means are over all contributing k points (not means of structure means); "
        "electron-count absolute errors use exact Hermitian real-space traces per structure; "
        "Fermi–Dirac residuals use one fit per structure. Worst is min for lower bounds "
        "and positive-definite success, max otherwise. Counts exclude invalid metrics; "
        "total denominators retain them. Contributing structure counts are recovered from "
        "historical per-structure summaries. Element MAEs are equally "
        "weighted structure means, with worst structure MAE; k points do not apply (0). "
        "Reference element error is zero by definition, not a physical residual.\n\n"
    )
    header = [
        "Material",
        "Operator",
        "Diagnostic",
        "Ref mean",
        "Ref worst",
        "Ref n / totals",
        "Pred mean",
        "Pred worst",
        "Pred n / totals",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(cells(r)) + " |" for r in result["rows"])
    lookup = {(r["material"], r["diagnostic"]): r for r in result["rows"]}
    si = lookup["silicon", "overlap.positive_definite"]["predicted"]
    fd = lookup["sio2", "finite_temperature_density_relative_frobenius"]
    paragraph = (
        f"The silicon reference-support projected overlap is positive definite at only "
        f"{si['mean']:.2%} of {si['count']} sampled k points; support restriction plus "
        "hermitization therefore does not establish a valid learned metric. No silicon "
        "structure has an available joint Fermi–Dirac residual. Silicon density and "
        f"stationarity means cover only {lookup['silicon', 'density.occupation_min']['predicted']['count']} usable k points, whereas reference means "
        "cover all 8192; this is not a paired whole-mesh comparison. For SiO2, the mean "
        f"Fermi–Dirac residual is {fd['predicted']['mean']:.6g} versus "
        f"{fd['reference']['mean']:.6g} for reference H/S/D, with reference S/D retained "
        "in the prediction control. These finite-mesh consistency tests do not certify "
        "SCF convergence or Brillouin-zone-wide validity; no result after overlap or "
        "occupation repair is present. Nonzero zero-temperature idempotency residuals "
        "alone do not demonstrate a violation at 300 K.\n"
    )
    wide = [
        "Each compact cell pairs reference / prediction (R / P). Mean and worst "
        "are separate columns; exact valid counts and total denominators follow below."
    ]
    for panel_header, panel_rows in operator_panels(result):
        wide += [
            "",
            "| " + " | ".join(panel_header) + " |",
            "| " + " | ".join(["---"] * len(panel_header)) + " |",
        ]
        wide += ["| " + " | ".join(row) + " |" for row in panel_rows]
    (directory / "summary_projected.md").write_text(
        intro
        + "\n".join(wide)
        + "\n\n"
        + paragraph
        + "\nDetailed values and denominators:\n\n"
        + "\n".join(lines)
        + "\n"
    )

    def escape(s):
        return (
            s.replace("—", "--")
            .replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
        )

    tex = ["% " + line for line in intro.splitlines() if line]
    tex += [
        "% Count columns: n / contributing structures / total structures / total kpoints."
    ]
    tex += [
        r"\begin{tabular}{lllrrlrrl}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in result["rows"]:
        tex_cells = cells(row)
        for side, index in (("reference", 5), ("predicted", 8)):
            value = row[side]
            tex_cells[index] = (
                f"{value['count']}/{value['contributing_structures']}/"
                f"{value['total_structures']}/{value['total_kpoints']}"
            )
        tex.append(" & ".join(escape(c) for c in tex_cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex += [
        "% Alternative compact panels: reference / prediction in each cell.",
        "% Counts and total denominators are in the detailed table above.",
    ]
    for panel_header, panel_rows in operator_panels(result):
        tex += [
            r"\begin{tabular}{ll" + "r" * (len(panel_header) - 2) + "}",
            r"\toprule",
            " & ".join(escape(c) for c in panel_header) + r" \\",
            r"\midrule",
        ]
        tex += [" & ".join(escape(c) for c in row) + r" \\" for row in panel_rows]
        tex += [r"\bottomrule", r"\end{tabular}"]
    (directory / "summary_projected.tex").write_text("\n".join(tex) + "\n")
    (directory / "summary_projected.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    render(build(args.data_dir), args.data_dir)


if __name__ == "__main__":
    main()
