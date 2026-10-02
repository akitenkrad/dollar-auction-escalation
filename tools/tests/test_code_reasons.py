import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from dollar_auction_tools.code_reasons import agreement, export_reasons


class CodeReasonsTests(unittest.TestCase):
    def test_export_and_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            child = root / "child"
            parent.mkdir()
            child.mkdir()
            (parent / "run.json").write_text(json.dumps({"run_uid": "PARENT"}))
            (child / "run.json").write_text(
                json.dumps({"run_uid": "CHILD", "lineage": {"parent_run_uid": "PARENT"}})
            )
            (child / "config.json").write_text(
                json.dumps({"parameters": {"framing": "named", "opponent": "human"}})
            )
            artifacts = child / "artifacts"
            artifacts.mkdir()
            transcripts = [
                {"trial": 0, "turn": 1, "raw_action": "bid", "reason": "I should continue"},
                {"trial": 0, "turn": 2, "raw_action": "calc", "reason": None},
                {"trial": 1, "turn": 1, "raw_action": "drop", "reason": "Stop now"},
            ]
            (artifacts / "transcripts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in transcripts)
            )
            exported = root / "reasons.csv"
            export_reasons(parent, exported)
            with exported.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run_uid"], "CHILD")
            self.assertEqual(rows[0]["reason"], "I should continue")
            for code in ["sunk_cost", "rivalry", "foresight", "game_recall"]:
                self.assertEqual(rows[0][code], "")

            coder_a = root / "a.csv"
            coder_b = root / "b.csv"
            fieldnames = list(rows[0])
            for path, values in [(coder_a, [(1, 0), (0, 1)]), (coder_b, [(1, 0), (1, 1)])]:
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for row, (sunk, rivalry) in zip(rows, values):
                        row = dict(row)
                        row.update(
                            sunk_cost=sunk,
                            rivalry=rivalry,
                            foresight=0,
                            game_recall=0,
                        )
                        writer.writerow(row)
            output = StringIO()
            with redirect_stdout(output):
                kappas = agreement(coder_a, coder_b)
            self.assertEqual(set(kappas), {"sunk_cost", "rivalry", "foresight", "game_recall"})
            self.assertIn("sunk_cost", output.getvalue())


if __name__ == "__main__":
    unittest.main()
