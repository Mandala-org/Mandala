"""Check selection and denominators against the committed per-structure evidence."""

import importlib.util
import json
from pathlib import Path
import unittest

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
        self.assertIn("no overlap repair", self.result["cleanup_status"])


if __name__ == "__main__":
    unittest.main()
