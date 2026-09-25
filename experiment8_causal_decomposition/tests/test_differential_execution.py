from __future__ import annotations

import unittest

from _support import ROOT  # noqa: F401
from common import VALIDATION_FILE, load_json


class DifferentialExecutionTests(unittest.TestCase):
    def test_all_program_pairs_passed(self) -> None:
        report = load_json(VALIDATION_FILE)
        self.assertTrue(report["differential_execution"]["enabled"])
        self.assertTrue(report["differential_execution"]["passed"])
        self.assertEqual(report["counts"]["differential_pairs"], 300)
        self.assertEqual(report["counts"]["differential_passed"], 300)
        self.assertGreaterEqual(report["differential_execution"]["total_cases"], 900)
        self.assertEqual(report["differential_execution"]["failed_programs"], [])


if __name__ == "__main__":
    unittest.main()
