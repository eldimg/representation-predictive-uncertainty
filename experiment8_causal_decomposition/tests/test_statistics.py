from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from _support import ROOT  # noqa: F401
import analyze_model
from analyze_model import holm_adjust
from common import atomic_write_json


class StatisticsTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_bounded(self) -> None:
        adjusted = holm_adjust({"C1": 0.001, "C2": 0.02, "C3": 0.01, "C4": 0.5})
        self.assertAlmostEqual(adjusted["C1"], 0.004)
        self.assertAlmostEqual(adjusted["C3"], 0.03)
        self.assertAlmostEqual(adjusted["C2"], 0.04)
        self.assertAlmostEqual(adjusted["C4"], 0.5)
        self.assertTrue(all(0 <= value <= 1 for value in adjusted.values()))

    def test_end_to_end_analysis_on_synthetic_1800_rows(self) -> None:
        condition_values = {
            "nl_exp7_anchor": 1.20,
            "procedural_prose": 1.00,
            "controlled_nl": 0.80,
            "pseudocode": 0.60,
            "python_canonical": 0.40,
            "python_opaque": 0.405,
        }
        rows = []
        for task_index in range(100):
            task_id = f"task_{task_index:03d}"
            family = f"family_{task_index // 2:03d}"
            for condition, value in condition_values.items():
                for variant in range(1, 4):
                    rows.append(
                        {
                            "trajectory_id": f"{task_id}_{condition}_{variant}",
                            "task_id": task_id,
                            "family": family,
                            "condition": condition,
                            "variant": variant,
                            "trajectory_valid": True,
                            "mean_entropy_top20": value,
                            "mean_entropy_valid_vocab": value + 0.01,
                            "semantic_step_balanced_entropy_top20": (
                                None if condition == "nl_exp7_anchor" else value + 0.02
                            ),
                            "step_mapped_token_fraction": (
                                None if condition == "nl_exp7_anchor" else 0.99
                            ),
                        }
                    )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame(rows).to_csv(directory / "trajectories.csv", index=False)
            atomic_write_json(
                directory / "config.json",
                {
                    "model": "Qwen/Qwen3-4B",
                    "model_slug": "qwen3",
                    "model_commit_hash": "synthetic",
                },
            )
            original_signflip = analyze_model.N_SIGNFLIP
            original_bootstrap = analyze_model.N_BOOTSTRAP
            analyze_model.N_SIGNFLIP = 1_000
            analyze_model.N_BOOTSTRAP = 1_000
            try:
                output = analyze_model.analyze(directory)
            finally:
                analyze_model.N_SIGNFLIP = original_signflip
                analyze_model.N_BOOTSTRAP = original_bootstrap
            self.assertEqual(output["contrasts"]["C1"]["result_status"], "ROBUST")
            self.assertEqual(output["contrasts"]["C2"]["result_status"], "ROBUST")
            self.assertEqual(output["contrasts"]["C3"]["result_status"], "ROBUST")
            self.assertTrue(output["contrasts"]["C4"]["equivalent"])
            self.assertEqual(
                output["contrasts"]["C4"]["c4_interpretation"],
                "statistically_detectable_but_practically_small",
            )


if __name__ == "__main__":
    unittest.main()
