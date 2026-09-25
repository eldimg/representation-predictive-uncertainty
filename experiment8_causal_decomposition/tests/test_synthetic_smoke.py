from __future__ import annotations

import unittest

from _support import ROOT
from common import load_json


class SyntheticSmokeTests(unittest.TestCase):
    def test_all_three_model_smokes_are_valid(self) -> None:
        for slug in ("qwen3", "phi3", "mistral"):
            payload = load_json(ROOT / "debug" / f"smoke_{slug}.json")
            self.assertEqual(payload["kind"], "synthetic_debug_only")
            self.assertEqual(payload["synthetic_task_count"], 5)
            self.assertEqual(payload["trajectory_count"], 30)
            self.assertTrue(payload["all_valid"])
            derived = [
                row for row in payload["records"] if row["condition"] != "nl_exp7_anchor"
            ]
            self.assertTrue(
                all(row["semantic_step_balanced_entropy_top20"] is not None for row in derived)
            )


if __name__ == "__main__":
    unittest.main()
