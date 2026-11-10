#!/usr/bin/env python3
"""Render the historical paper audit and overlap-only repair, without ML imports.

Selected aggregate keys: silicon_historical/joint_predicted_reference_support;
sio2_historical/predicted_H_reference_SD_reference_support. These are reference-
support controls (discard prediction-only blocks), with Hermitian projection
inside diagnose_kpoint. Keep these unrepaired values alongside silicon's
joint_predicted_reference_support_overlap_repaired ("after overlap repair").
That additional scenario floors only S^h(k) eigenvalues at 1e-6, without an
upper clamp, jitter or occupation projection. Its real-space errors and traces
retain the original S. Mesh trace errors are separately labelled. Repair does
not apply to the H-only SiO2 and Zn models. Reference columns remain unchanged;
repair-statistic reference columns use the one-structure overlap control only.
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
REPAIRED = "joint_predicted_reference_support_overlap_repaired"
REPAIR_DIAGNOSTICS = [
    ("overlap_repair.relative_frobenius", "max"),
    ("overlap_repair.floored_eigenvalues", "max"),
    ("overlap_repair.lambda_min_before", "min"),
    ("overlap_repair.lambda_min_after", "min"),
]
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
    "electron_count_absolute_error": "Electron count absolute error (real-space, unrepaired S)",
    "mesh_electron_count_absolute_error": "Electron count absolute error (k-mesh)",
    "density.zero_temperature_idempotency_relative": "Zero-T idempotency residual",
    "consistency.commutator_relative": "Stationarity residual",
    "finite_temperature_density_relative_frobenius": "Fermi-Dirac residual",
    "overlap_repair.relative_frobenius": "Overlap correction relative Frobenius norm",
    "overlap_repair.floored_eigenvalues": "Floored eigenvalue count",
    "overlap_repair.lambda_min_before": "Min eigenvalue before repair",
    "overlap_repair.lambda_min_after": "Min eigenvalue after repair",
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
        "mesh_electron_count_absolute_error",
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

        def metric_stats(label, key, direction, evaluation=evaluation, summary=summary):
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
        if material == "silicon":
            repaired_paths = [
                directory / material / "overlap_repaired_aggregate.json",
                directory / material / "overlap_repaired_summary.json",
                directory / material / "overlap_repair_corrections.json",
            ]
            sources.extend(str(p.relative_to(directory)) for p in repaired_paths)
            repaired_evaluations = json.loads(repaired_paths[0].read_text())[
                "evaluations"
            ]
            if len(repaired_evaluations) != 1:
                raise ValueError("Ambiguous repaired aggregate")
            repaired_evaluation = next(iter(repaired_evaluations.values()))
            repaired_summary = json.loads(repaired_paths[1].read_text())
            corrections = json.loads(repaired_paths[2].read_text())
            if (
                repaired_evaluation["scenarios"]["reference_full"]
                != evaluation["scenarios"]["reference_full"]
            ):
                raise ValueError("Overlap repair changed the reference baseline")
            if (
                repaired_summary["status"] != "complete"
                or repaired_summary["structures"] != evaluation["structures"]
            ):
                raise ValueError("Incomplete repaired summary")
            scenario = repaired_evaluation["scenarios"][REPAIRED]
            denominators[material]["after_overlap_repair"] = dict(
                structures=repaired_evaluation["structures"],
                kpoints=scenario["kpoint_count"],
                invalid_metric_kpoints=scenario["invalid_kpoint_count"],
                structures_with_invalid_metric=scenario[
                    "structures_with_invalid_metric"
                ],
            )
            for operator, diagnostics in DIAGNOSTICS.items():
                for key, direction in diagnostics:
                    rows.append(
                        dict(
                            material=material,
                            operator=operator,
                            diagnostic=key,
                            reference=metric_stats("reference_full", key, direction),
                            predicted=metric_stats(
                                REPAIRED,
                                key,
                                direction,
                                repaired_evaluation,
                                repaired_summary,
                            ),
                            scenario=REPAIRED,
                            worst_direction=direction,
                        )
                    )
            # The reference repair control covers one structure, not all 16.
            control = corrections["reference_control"]["summary"]
            for key, direction in REPAIR_DIAGNOSTICS:
                control_key = key.removeprefix("overlap_repair.")
                rows.append(
                    dict(
                        material=material,
                        operator="S",
                        diagnostic=key,
                        reference=stats(control, control_key, direction, 1, 1),
                        predicted=metric_stats(
                            REPAIRED,
                            key,
                            direction,
                            repaired_evaluation,
                            repaired_summary,
                        ),
                        scenario=REPAIRED,
                        worst_direction=direction,
                    )
                )
            # Reference mesh traces come from the unchanged historical records.
            reference_errors = [
                abs(r["scenarios"]["reference_full"]["mesh_electron_count_error"])
                for r in summary["results"]
            ]
            reference_mesh = stats(
                dict(
                    kpoint_count=evaluation["scenarios"]["reference_full"][
                        "kpoint_count"
                    ],
                    mesh_electron_count_absolute_error=dict(
                        count=len(reference_errors),
                        mean=mean(reference_errors),
                        max=max(reference_errors),
                    ),
                ),
                "mesh_electron_count_absolute_error",
                "max",
                evaluation["structures"],
            )
            for scenario_label in (SELECTION[material], REPAIRED):
                rows.append(
                    dict(
                        material=material,
                        operator="D",
                        diagnostic="mesh_electron_count_absolute_error",
                        reference=reference_mesh,
                        predicted=metric_stats(
                            scenario_label,
                            "mesh_electron_count_absolute_error",
                            "max",
                            repaired_evaluation,
                            repaired_summary,
                        ),
                        scenario=scenario_label,
                        worst_direction="max",
                    )
                )
    return dict(
        schema_version=2,
        sources=sources,
        scenarios=SELECTION,
        additional_scenarios={"silicon": REPAIRED},
        denominators=denominators,
        cleanup_status="Silicon after overlap repair: floor S^h(k) eigenvalues at 1e-6, "
        "with no upper clamp or jitter. No occupation projection was applied: "
        "the data use finite-temperature (300 K Fermi-Dirac) occupations, "
        "so a zero-temperature idempotent target is inappropriate. "
        "SiO2 and ZnCu2Sn(SeS)2 predict H only; overlap repair does not apply to them.",
        rows=rows,
    )


def fmt(value):
    return "—" if value is None else f"{value:.6g}"


def cells(row):
    output = [
        display_material(row),
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


def display_material(row):
    return row["material"] + (
        " (after overlap repair)" if row["scenario"] == REPAIRED else ""
    )


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
        repair_keys = [key for key in keys if key.startswith("overlap_repair.")]
        keys = [key for key in keys if key not in repair_keys]
        groups = [keys[:1]] + [keys[i : i + 2] for i in range(1, len(keys), 2)]
        groups += [repair_keys[i : i + 2] for i in range(0, len(repair_keys), 2)]
        for keys_in_panel in groups:
            header = ["Material", "Operator"]
            for key in keys_in_panel:
                header += [label(key) + " mean (R / P)", label(key) + " worst (R / P)"]
            table = []
            for material, repaired in (
                ("silicon", False),
                ("silicon", True),
                ("sio2", False),
                ("zncusnses", False),
            ):
                selected = {
                    r["diagnostic"]: r
                    for r in rows
                    if r["material"] == material
                    and r["diagnostic"] in keys_in_panel
                    and (r["scenario"] == REPAIRED) == repaired
                }
                if not selected:
                    continue
                row = [
                    display_material(next(iter(selected.values()))),
                    operator_label(material, operator),
                ]
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
        + f"; silicon after overlap repair: `{REPAIRED}`"
        + ".\n\n"
        + result["cleanup_status"]
        + "\n\n"
        "These spectral values use Hermitian-projected matrices on reference support, "
        "a control requiring reference sparsity, not independent deployment. SiO2 H uses "
        "reference S/D; silicon uses predicted H/S/D. Zn reports H elements only. "
        "Only silicon has an additional after-overlap-repair scenario.\n\n"
        "Means are over all contributing k points (not means of structure means); "
        "real-space electron-count errors use exact Hermitian traces with unrepaired S; "
        "separately labelled k-mesh errors use the scenario's metric and a normalized "
        "mesh average per structure. Repair-statistic reference columns cover the "
        "512 k points of structure 0 only (0 floored eigenvalues). "
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
    lookup = {
        (r["material"], r["diagnostic"]): r
        for r in result["rows"]
        if r["scenario"] != REPAIRED
    }
    repaired_lookup = {
        r["diagnostic"]: r for r in result["rows"] if r["scenario"] == REPAIRED
    }
    si = lookup["silicon", "overlap.positive_definite"]["predicted"]
    after = {key: row["predicted"] for key, row in repaired_lookup.items()}
    before = {
        key: row["predicted"]
        for (material, key), row in lookup.items()
        if material == "silicon"
    }
    correction_count = after["overlap_repair.floored_eigenvalues"]
    total_floored = round(correction_count["mean"] * correction_count["count"])
    fd = lookup["sio2", "finite_temperature_density_relative_frobenius"]
    paragraph = (
        f"For silicon, overlap repair raises positive-definite success from {si['mean']:.2%} "
        f"to {after['overlap.positive_definite']['mean']:.2%} across {si['count']} k points; "
        f"Cholesky failures fall from {before['overlap.cholesky_failed_without_jitter']['mean']:.2%} "
        f"to {after['overlap.cholesky_failed_without_jitter']['mean']:.2%}. "
        f"The worst minimum eigenvalue changes from {before['overlap.lambda_min']['worst']:.6g} "
        f"to {after['overlap.lambda_min']['worst']:.6g}. The relative overlap correction "
        f"has mean {after['overlap_repair.relative_frobenius']['mean']:.6g} and maximum "
        f"{after['overlap_repair.relative_frobenius']['worst']:.6g}, flooring {total_floored} "
        f"eigenvalues in total ({correction_count['mean']:.6g} per k point on average). "
        f"Density and stationarity diagnostics now cover {after['density.occupation_min']['count']} "
        f"k points versus {before['density.occupation_min']['count']} before repair; "
        "these whole-result means have different contributing samples. "
        f"Mean minimum/maximum occupations change from {before['density.occupation_min']['mean']:.6g}/"
        f"{before['density.occupation_max']['mean']:.6g} to {after['density.occupation_min']['mean']:.6g}/"
        f"{after['density.occupation_max']['mean']:.6g}; the repaired worst extrema are "
        f"{after['density.occupation_min']['worst']:.6g} and {after['density.occupation_max']['worst']:.6g}. "
        f"The mean stationarity residual changes from {before['consistency.commutator_relative']['mean']:.6g} "
        f"to {after['consistency.commutator_relative']['mean']:.6g}. A joint Fermi–Dirac residual "
        f"is now available for all {after['finite_temperature_density_relative_frobenius']['count']} "
        f"structures (previously 0), with mean {after['finite_temperature_density_relative_frobenius']['mean']:.6g}. "
        f"The k-mesh electron-count absolute error changes from mean "
        f"{before['mesh_electron_count_absolute_error']['mean']:.6g} to "
        f"{after['mesh_electron_count_absolute_error']['mean']:.6g}; the exact real-space "
        f"error stays {before['electron_count_absolute_error']['mean']:.6g}, and element MAEs "
        "are unchanged. Occupation-bound violations remain. No occupation projection "
        "was applied because these are finite-temperature (300 K Fermi-Dirac) data; "
        "the zero-T idempotency residual remains informational. SiO2 and ZnCu2Sn(SeS)2 "
        "predict H only, so overlap repair does not apply. For SiO2, the mean "
        f"Fermi–Dirac residual is {fd['predicted']['mean']:.6g} versus "
        f"{fd['reference']['mean']:.6g} for reference H/S/D, with reference S/D retained "
        "in the prediction control. These finite-mesh consistency tests do not certify "
        "SCF convergence or Brillouin-zone-wide validity. Nonzero zero-temperature idempotency residuals "
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
