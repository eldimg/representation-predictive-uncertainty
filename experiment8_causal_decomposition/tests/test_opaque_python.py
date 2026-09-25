from __future__ import annotations

import ast
import unittest

from _support import ROOT  # noqa: F401
from common import RENDERED_FILE, load_json
from opaque_renamer import normalized_ast_equivalent


class OpaquePythonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = load_json(RENDERED_FILE)["records"]
        cls.canonical = {
            record["program_id"]: record
            for record in records
            if record["condition"] == "python_canonical"
        }
        cls.opaque = {
            record["program_id"]: record
            for record in records
            if record["condition"] == "python_opaque"
        }

    def test_all_pairs_compile_and_are_alpha_equivalent(self) -> None:
        self.assertEqual(set(self.canonical), set(self.opaque))
        self.assertEqual(len(self.canonical), 300)
        for program_id in self.canonical:
            canonical = self.canonical[program_id]["reference_text"]
            opaque_record = self.opaque[program_id]
            opaque = opaque_record["reference_text"]
            compile(canonical, "<canonical>", "exec")
            compile(opaque, "<opaque>", "exec")
            self.assertTrue(
                normalized_ast_equivalent(
                    canonical, opaque, opaque_record["opacity"]["mapping"]
                ),
                program_id,
            )

    def test_attributes_and_globals_are_preserved(self) -> None:
        for program_id, opaque_record in self.opaque.items():
            opacity = opaque_record["opacity"]
            self.assertEqual(opacity["canonical_globals"], opacity["opaque_globals"])
            mapping = opacity["mapping"]
            canonical_tree = ast.parse(self.canonical[program_id]["reference_text"])
            opaque_tree = ast.parse(opaque_record["reference_text"])
            canonical_attributes = [
                node.attr for node in ast.walk(canonical_tree) if isinstance(node, ast.Attribute)
            ]
            opaque_attributes = [
                node.attr for node in ast.walk(opaque_tree) if isinstance(node, ast.Attribute)
            ]
            self.assertEqual(canonical_attributes, opaque_attributes)
            self.assertEqual(len(mapping), len(set(mapping.values())))


if __name__ == "__main__":
    unittest.main()
