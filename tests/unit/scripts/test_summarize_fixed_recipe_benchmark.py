"""Small synthetic runs; no torch, GPU, network, or package installation needed."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_summary", ROOT / "scripts/report/summarize_fixed_recipe_benchmark.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "logs").mkdir()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def write_csv(self, arm, version, rows):
        path = (
            self.root
            / "results"
            / arm
            / "metrics"
            / f"version_{version}"
            / "metrics.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted(set().union(*(r.keys() for r in rows)))
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_versions_units_missing_test_and_frozen_endpoint(self):
        arm = "mandala_graphene"
        directory = self.root / "results" / arm
        self.write_json(directory / "segment_1.json", {"epochs": 350})
        self.write_json(directory / "segment_2.json", {"epochs": 700})
        self.write_csv(
            arm,
            0,
            [
                {
                    "epoch": 348,
                    "step": 10,
                    "val/hamiltonian_mae": 999,
                    "val/hamiltonian_mae_elementwise": 0.004,
                    "val/hamiltonian_mae_elementwise_projected": 0.003,
                },
                {"epoch": 350, "step": 11, "val/hamiltonian_mae_elementwise": 0.003},
            ],
        )
        self.write_csv(
            arm,
            1,
            [
                {"epoch": 350, "step": 11, "val/hamiltonian_mae_elementwise": 0.002},
                {"epoch": 699, "step": 12, "val/hamiltonian_mae_elementwise": 0.001},
                {"epoch": 699, "step": 12, "train/hamiltonian_mae_elementwise": 0.0001},
                {"epoch": 700, "step": 13, "val/hamiltonian_mae_elementwise": 0.00001},
            ],
        )
        # Repeated tqdm lines are not independent epochs. Last elapsed time includes val.
        (self.root / "logs" / f"{arm}_1_123.out").write_text(
            "Epoch 348: 100%|xx| 2/2 [01:00<00:00]\r"
            "Epoch 348: 100%|xx| 2/2 [02:00<00:00]\n"
            "Epoch 348: 100%|xx| 2/2 [02:00<00:00]\n"
            "Epoch 349: 90%|xx| 1/2 [09:00<01:00]\n"
        )
        (self.root / "logs" / f"{arm}_2_124.out").write_text(
            "Epoch 699: 100%|xx| 2/2 [04:00<00:00]\n"
            "Epoch 700: 100%|xx| 2/2 [50:00<00:00]\n"
        )
        self.write_json(self.root / "HORIZON_FREEZE.json", {"decision": "90 h"})
        result = benchmark.summarize(self.root, True, "2026-11-10T00:00:00Z")
        a = result["arms"][arm]
        raw = a["validation"]["raw"]
        self.assertEqual(a["epochs_at_90h"], 700)
        self.assertEqual(raw["best"]["epoch"], 700)
        self.assertEqual(raw["best"]["mae_mev"], 1)
        self.assertEqual(raw["last"]["mae_mev"], 1)
        self.assertEqual(raw["logged_epochs"], 3)
        self.assertEqual(raw["matched_epochs"]["350"]["epoch"], 349)  # tie: earlier
        self.assertEqual(a["mean_minutes_per_epoch"], 3)
        self.assertEqual(a["timing_epoch_count"], 2)
        self.assertIsNone(a["terminal_test"])
        self.assertIsNone(result["arms"]["deeph_e3_graphene"]["validation"]["raw"])
        text = benchmark.markdown(result)
        self.assertIn("interim snapshot", text)
        self.assertIn("pending", text)
        self.assertEqual(
            len(text.split("| Arm |")[1].splitlines()[0].split("|")),
            len(text.split("| mandala_graphene |")[1].splitlines()[0].split("|")),
        )

    def test_hartree_conversion_and_terminal_views(self):
        arm = "deeph_e3_mos2"
        directory = self.root / "results" / arm
        self.write_csv(
            arm, 0, [{"epoch": 173, "val/hamiltonian_mae_elementwise_hartree": 0.001}]
        )
        self.write_json(
            directory / "terminal_test.json",
            dict(
                epoch=174,
                metrics={
                    "raw": {"structure_mean_mae_mev": 4},
                    "projected": {"structure_mean_mae_mev": 3},
                },
            ),
        )
        result = benchmark.summarize(self.root)
        a = result["arms"][arm]
        self.assertAlmostEqual(a["validation"]["raw"]["last"]["mae_mev"], 27.2114)
        self.assertFalse(
            a["validation"]["raw"]["matched_epochs"]["175"]["target_reached"]
        )
        self.assertEqual(a["terminal_test"]["mae_mev"], {"raw": 4.0, "projected": 3.0})
        self.assertIsNone(a["validation"]["projected"])
        self.assertIn("†", benchmark.markdown(result))

    def test_refuses_ambiguous_legacy_or_nonfinite_metric(self):
        self.write_csv("mace_h_mos2", 0, [{"epoch": 0, "val/hamiltonian_mae": 0.1}])
        with self.assertRaisesRegex(ValueError, "No common elementwise"):
            benchmark.summarize(self.root)
        self.write_csv(
            "mace_h_mos2",
            0,
            [{"epoch": 0, "val/hamiltonian_mae_elementwise_mev": "nan"}],
        )
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            benchmark.summarize(self.root)
        with self.assertRaisesRegex(ValueError, "schema"):
            benchmark.terminal_scores({"mae": 0.1})


if __name__ == "__main__":
    unittest.main()
