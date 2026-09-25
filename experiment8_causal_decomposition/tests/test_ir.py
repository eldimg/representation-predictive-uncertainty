from __future__ import annotations

import ast
import unittest

from _support import ROOT  # noqa: F401
from common import EXPECTED_TASKS_SHA256, IR_FILE, TASKS_FILE, load_json, sha256_file
from ir import dict_to_ast


class ProgramIRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = load_json(IR_FILE)

    def test_frozen_source_hash(self) -> None:
        self.assertEqual(sha256_file(TASKS_FILE), EXPECTED_TASKS_SHA256)

    def test_program_count_and_unique_ids(self) -> None:
        programs = self.payload["programs"]
        self.assertEqual(len(programs), 300)
        self.assertEqual(len({program["program_id"] for program in programs}), 300)

    def test_every_program_has_interface_and_body(self) -> None:
        for program in self.payload["programs"]:
            self.assertTrue(program["interface"]["function_name"])
            self.assertGreaterEqual(len(program["interface"]["parameters"]), 1)
            self.assertGreater(program["body"]["statement_count"], 0)
            tree = dict_to_ast(program["ast"])
            self.assertIsInstance(tree, ast.Module)
            self.assertIsInstance(tree.body[0], ast.FunctionDef)

    def test_ids_are_stable_and_owned(self) -> None:
        for program in self.payload["programs"]:
            trace_ids = [node["trace_node_id"] for node in program["trace_nodes"]]
            step_ids = [step["semantic_step_id"] for step in program["semantic_steps"]]
            self.assertEqual(len(trace_ids), len(set(trace_ids)))
            self.assertEqual(len(step_ids), len(set(step_ids)))
            self.assertEqual(step_ids[0], "S001")
            self.assertEqual(program["semantic_steps"][0]["kind"], "FUNCTION_INTERFACE")
            self.assertTrue(
                all(node["owner_semantic_step_id"] in step_ids for node in program["trace_nodes"])
            )


if __name__ == "__main__":
    unittest.main()
