from __future__ import annotations

import unittest

from _support import ROOT  # noqa: F401
from common import IR_DERIVED_CONDITIONS, VALIDATION_FILE, load_json


class TokenAlignmentTests(unittest.TestCase):
    def test_all_model_condition_coverage_gates_pass(self) -> None:
        report = load_json(VALIDATION_FILE)
        coverage = report["tokenizer_coverage"]
        self.assertEqual(set(coverage), {"qwen3", "phi3", "mistral"})
        for model in coverage.values():
            self.assertTrue(model["all_conditions_pass"])
            self.assertEqual(set(model["conditions"]), set(IR_DERIVED_CONDITIONS))
            for condition in model["conditions"].values():
                self.assertGreaterEqual(condition["step_mapped_token_fraction"], 0.90)


if __name__ == "__main__":
    unittest.main()
