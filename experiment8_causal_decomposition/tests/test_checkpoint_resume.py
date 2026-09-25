from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from _support import ROOT  # noqa: F401
from run_experiment8 import load_completed, repair_partial_tokens


class CheckpointResumeTests(unittest.TestCase):
    def test_partial_token_rows_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trajectories = directory / "trajectories.csv"
            tokens = directory / "tokens.csv"
            with trajectories.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["trajectory_id", "value"])
                writer.writeheader()
                writer.writerow({"trajectory_id": "complete", "value": 1})
            with tokens.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["trajectory_id", "token"])
                writer.writeheader()
                writer.writerow({"trajectory_id": "complete", "token": "a"})
                writer.writerow({"trajectory_id": "partial", "token": "b"})
            completed = load_completed(trajectories)
            self.assertEqual(completed, {"complete"})
            repair_partial_tokens(tokens, completed)
            with tokens.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{"trajectory_id": "complete", "token": "a"}])


if __name__ == "__main__":
    unittest.main()
