from __future__ import annotations

import collections
import unittest

from _support import ROOT  # noqa: F401
from common import CONDITIONS, RENDERED_FILE, TASKS_FILE, load_json


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rendered = load_json(RENDERED_FILE)
        cls.tasks = {task["id"]: task for task in load_json(TASKS_FILE)}

    def test_exact_counts(self) -> None:
        records = self.rendered["records"]
        self.assertEqual(len(records), 1800)
        counts = collections.Counter(
            (record["task_id"], record["condition"]) for record in records
        )
        self.assertEqual(len(counts), 600)
        self.assertTrue(all(value == 3 for value in counts.values()))

    def test_frozen_anchors_are_byte_identical(self) -> None:
        for record in self.rendered["records"]:
            task = self.tasks[record["task_id"]]
            index = record["variant"] - 1
            if record["condition"] == "nl_exp7_anchor":
                self.assertEqual(record["reference_text"], task["nl"][index])
            if record["condition"] == "python_canonical":
                self.assertEqual(record["reference_text"], task["python"][index])

    def test_interface_present_in_derived_conditions(self) -> None:
        for record in self.rendered["records"]:
            if record["condition"] == "nl_exp7_anchor":
                continue
            first_line = record["reference_text"].splitlines()[0]
            self.assertTrue(first_line.strip())
            if record["condition"] in {"procedural_prose", "controlled_nl", "pseudocode"}:
                canonical_name = record["source_python"].split("def ", 1)[1].split("(", 1)[0]
                self.assertIn(canonical_name, first_line)

    def test_semantic_spans_non_overlapping(self) -> None:
        for record in self.rendered["records"]:
            spans = sorted(record["semantic_step_spans"], key=lambda span: span["start_char"])
            previous_end = -1
            for span in spans:
                self.assertGreaterEqual(span["start_char"], previous_end)
                self.assertGreater(span["end_char"], span["start_char"])
                self.assertLessEqual(span["end_char"], len(record["reference_text"]))
                previous_end = span["end_char"]


if __name__ == "__main__":
    unittest.main()
