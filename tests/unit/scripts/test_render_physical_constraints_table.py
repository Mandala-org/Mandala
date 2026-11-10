"""Check selection and denominators against the committed per-structure evidence."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "physical_table", ROOT / "scripts/report/render_physical_constraints_table.py"
)
physical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(physical)


class PhysicalTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = physical.build(physical.DEFAULT)

    def test_aggregate_weights_and_worst_match_structure_records(self):
        for material, label in physical.SELECTION.items():
            summary = json.loads(
                (physical.DEFAULT / material / "historical_summary.json").read_text()
            )
            for row in self.result["rows"]:
                if row["material"] != material or row["scenario"] != label:
                    continue
                key = row["diagnostic"]
                if (
                    key
                    not in summary["results"][0]["scenarios"][label]["summary"][
                        "metrics"
                    ]
                ):
                    continue  # Structure-level FD/charge summaries have separate definitions.
                for side, scenario in [
                    ("reference", "reference_full"),
                    ("predicted", label),
                ]:
                    entries = [
                        r["scenarios"][scenario]["summary"]["metrics"].get(key)
                        for r in summary["results"]
                    ]
                    entries = [e for e in entries if e and e["count"]]
                    with self.subTest(material=material, diagnostic=key, side=side):
                        count = sum(e["count"] for e in entries)
                        if not count:
                            self.assertEqual(row[side]["count"], 0)
                            self.assertIsNone(row[side]["mean"])
                            continue
                        self.assertEqual(row[side]["count"], count)
                        self.assertAlmostEqual(
                            row[side]["mean"],
                            sum(e["mean"] * e["count"] for e in entries) / count,
                            delta=max(1e-10, abs(row[side]["mean"]) * 1e-10),
                        )
                        worst = min if row["worst_direction"] == "min" else max
                        self.assertEqual(
                            row[side]["worst"],
                            worst(e[row["worst_direction"]] for e in entries),
                        )
                        self.assertEqual(
                            row[side]["contributing_structures"], len(entries)
                        )

    def test_no_raw_or_zn_spectral_results_and_missing_fd_retained(self):
        self.assertTrue(all("raw" not in r["diagnostic"] for r in self.result["rows"]))
        zn = [r for r in self.result["rows"] if r["material"] == "zncusnses"]
        self.assertEqual(len(zn), 1)
        self.assertEqual(zn[0]["operator"], "H")
        fd = next(
            r
            for r in self.result["rows"]
            if r["material"] == "silicon"
            and r["diagnostic"] == "finite_temperature_density_relative_frobenius"
        )
        self.assertEqual(fd["predicted"]["count"], 0)
        self.assertEqual(fd["predicted"]["total_kpoints"], 8192)
        self.assertIsNone(fd["predicted"]["mean"])
        self.assertEqual(fd["reference"]["count"], 16)
        self.assertIn(
            "No occupation projection was applied", self.result["cleanup_status"]
        )

    def test_repaired_counts_references_and_element_errors(self):
        original = {
            r["diagnostic"]: r
            for r in self.result["rows"]
            if r["material"] == "silicon"
            and r["scenario"] == physical.SELECTION["silicon"]
        }
        repaired = {
            r["diagnostic"]: r
            for r in self.result["rows"]
            if r["scenario"] == physical.REPAIRED
        }
        for key, row in repaired.items():
            if key in original:
                self.assertEqual(row["reference"], original[key]["reference"])
            self.assertEqual(row["predicted"]["contributing_structures"], 16)
            self.assertEqual(row["predicted"]["total_kpoints"], 8192)
        self.assertEqual(repaired["overlap.positive_definite"]["predicted"]["mean"], 1)
        self.assertEqual(
            repaired["overlap.cholesky_failed_without_jitter"]["predicted"]["mean"], 0
        )
        self.assertEqual(repaired["density.occupation_min"]["predicted"]["count"], 8192)
        self.assertEqual(
            repaired["finite_temperature_density_relative_frobenius"]["predicted"][
                "count"
            ],
            16,
        )
        self.assertLess(repaired["density.occupation_min"]["predicted"]["worst"], 0)
        self.assertGreater(repaired["density.occupation_max"]["predicted"]["worst"], 2)
        self.assertEqual(
            repaired["electron_count_absolute_error"]["predicted"],
            original["electron_count_absolute_error"]["predicted"],
        )
        for key, _ in physical.REPAIR_DIAGNOSTICS:
            self.assertEqual(repaired[key]["reference"]["count"], 512)
            self.assertEqual(repaired[key]["reference"]["total_structures"], 1)
        self.assertEqual(
            repaired["overlap_repair.floored_eigenvalues"]["reference"]["mean"], 0
        )
        self.assertFalse(any("MAE" in key for key in repaired))
        self.assertFalse(
            any(
                r["scenario"] == physical.REPAIRED and r["material"] != "silicon"
                for r in self.result["rows"]
            )
        )

    def test_repair_aggregate_matches_every_structure_and_correction(self):
        directory = physical.DEFAULT / "silicon"
        summary = json.loads((directory / "overlap_repaired_summary.json").read_text())
        corrections = json.loads(
            (directory / "overlap_repair_corrections.json").read_text()
        )
        total = 0
        for structure in corrections["structures"]:
            self.assertEqual(len(structure["kpoints"]), 512)
            self.assertTrue(
                all(r["lambda_min_after"] > 0 for r in structure["kpoints"])
            )
            total += sum(r["floored_eigenvalues"] for r in structure["kpoints"])
        self.assertEqual(total, corrections["total_floored_eigenvalues"])
        for row in self.result["rows"]:
            if row["scenario"] != physical.REPAIRED:
                continue
            key = row["diagnostic"]
            entries = [
                r["scenarios"][physical.REPAIRED]["summary"]["metrics"].get(key)
                for r in summary["results"]
            ]
            entries = [e for e in entries if e]
            if not entries:
                continue
            count = sum(e["count"] for e in entries)
            self.assertEqual(count, row["predicted"]["count"])
            self.assertAlmostEqual(
                sum(e["mean"] * e["count"] for e in entries) / count,
                row["predicted"]["mean"],
                delta=max(1e-10, abs(row["predicted"]["mean"]) * 1e-10),
            )

    def test_rendered_artifacts_are_current_and_label_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            physical.render(self.result, directory)
            for extension in ("json", "md", "tex"):
                name = "summary_projected." + extension
                self.assertEqual(
                    (directory / name).read_text(),
                    (physical.DEFAULT / name).read_text(),
                )
            text = (directory / "summary_projected.md").read_text()
            self.assertIn("after overlap repair", text)
            self.assertIn("No occupation projection was applied", text)
            self.assertIn("300 K", text)


if __name__ == "__main__":
    unittest.main()
