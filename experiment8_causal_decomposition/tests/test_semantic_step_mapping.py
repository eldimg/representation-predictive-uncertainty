from __future__ import annotations

import unittest

from _support import ROOT  # noqa: F401
from common import map_token_to_semantic_step


class SemanticStepMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spans = [
            {"semantic_step_id": "S001", "start_char": 0, "end_char": 10},
            {"semantic_step_id": "S002", "start_char": 10, "end_char": 20},
        ]

    def test_full_containment_maps_once(self) -> None:
        self.assertEqual(map_token_to_semantic_step(2, 8, self.spans), ("mapped", "S001"))

    def test_boundary_overlap_is_excluded(self) -> None:
        self.assertEqual(
            map_token_to_semantic_step(8, 12, self.spans),
            ("boundary_overlap", None),
        )

    def test_gap_is_unmapped(self) -> None:
        spans = [{"semantic_step_id": "S001", "start_char": 0, "end_char": 5}]
        self.assertEqual(map_token_to_semantic_step(6, 8, spans), ("unmapped", None))


if __name__ == "__main__":
    unittest.main()
