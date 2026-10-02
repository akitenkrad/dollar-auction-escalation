import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dollar_auction_tools.analyze import analyze_sweep, sweep_sources


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = self.root / "pilot_parent"
        self.parent.mkdir()
        (self.parent / "run.json").write_text(
            json.dumps({"run_uid": "PARENT-UID", "run_slug": "pilot_parent"})
        )
        self.children = []
        for index, (framing, opponent, b) in enumerate(
            [
                ("named", "rational_ai", 100),
                ("disguised", "human", 250),
                ("named", "human", 250),
                ("disguised", "rational_ai", 100),
            ]
        ):
            child = self.root / f"pilot_child_{index}"
            child.mkdir()
            uid = f"CHILD-{index}"
            (child / "run.json").write_text(
                json.dumps(
                    {
                        "run_uid": uid,
                        "run_slug": child.name,
                        "lineage": {"parent_run_uid": "PARENT-UID"},
                    }
                )
            )
            (child / "config.json").write_text(
                json.dumps(
                    {
                        "parameters": {
                            "model": "model-a" if index < 2 else "model-b",
                            "framing": framing,
                            "opponent": opponent,
                            "b": b,
                            "s": 100,
                            "kind": "main",
                            "max_bids": 50,
                        }
                    }
                )
            )
            events = []
            for trial in range(8):
                outcome = "no_bid" if trial == 0 else ("invalid" if trial == 1 else "p1_win")
                t = 0 if trial == 0 else (2 if trial == 1 else 1 + trial % 3)
                h = 0 if trial == 0 else (7 if trial == 1 else t + 4)
                censored = trial == 7
                unit = f"trial-{trial}"
                common = {
                    "run_uid": uid,
                    "unit_id": unit,
                    "trial": trial,
                    "model": "model-a" if index < 2 else "model-b",
                    "framing": framing,
                    "opponent": opponent,
                    "b": b,
                    "s": 100,
                }
                events.append(
                    {
                        **common,
                        "schema": "terminal",
                        "t": t,
                        "T": t,
                        "outcome": outcome,
                        "censored": censored,
                        "budget": t if censored else b,
                        "h": h,
                        "turn": trial + 1,
                    }
                )
                x1 = 0 if trial == 0 else (1 if b == 100 else 50 + trial % 4)
                drop_anchor = 1 if b == 100 else 52
                events.append(
                    {
                        **common,
                        "schema": "x.dollar-auction-escalation.trial",
                        "t": t,
                        "T": t,
                        "first_end": t <= 1,
                        "first_bid_deviation": x1 - drop_anchor,
                        "waste": x1 / 100,
                        "paraphrase_id": trial % 3,
                        "turn_capped": censored,
                        "waste_is_lower_bound": censored,
                    }
                )
                if trial == 1:
                    events.append(
                        {
                            **common,
                            "schema": "x.dollar-auction-escalation.violation",
                            "kind": "format",
                        }
                    )
            (child / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events)
            )
            (child / "metrics.csv").write_text(
                "run_uid,step,step_unit,scope,name,value\n"
                f"{uid},,,run,tokens_in,{100 + index}\n"
                f"{uid},,,run,tokens_out,{20 + index}\n"
                f"{uid},,,run,n_calls,8\n"
                f"{uid},,,run,n_calls_missing_prompt_tokens,1\n"
            )
            self.children.append(child)

    def tearDown(self):
        self.temp.cleanup()

    def test_main_and_sensitivity_outputs_are_complete_and_runs_are_immutable(self):
        before = {
            path: path.read_bytes()
            for child in self.children
            for path in child.iterdir()
            if path.is_file()
        }
        out = self.root / "analysis"
        analyze_sweep(self.parent, out=out)

        required = [
            "trials.csv",
            "km_survival.csv",
            "cox_ph.csv",
            "logistic_first_end.csv",
            "first_bid_summary.csv",
            "sensitivity.csv",
            "violation_rates.csv",
            "usage.csv",
            "km_curves.png",
            "first_bid.png",
            "summary.md",
        ]
        for name in required:
            self.assertTrue((out / name).exists(), name)

        trials = pd.read_csv(out / "trials.csv")
        self.assertEqual(int(trials["no_bid"].sum()), 4)
        self.assertEqual(int(trials["logistic_denominator"].sum()), len(trials) - 4)
        self.assertIn("x1_deviation_drop_tie", trials.columns)
        self.assertIn("x1_deviation_mixed_tie", trials.columns)

        sensitivity = pd.read_csv(out / "sensitivity.csv")
        row = sensitivity[(sensitivity["scenario"] == "invalid_to_cap") & sensitivity["invalid"]].iloc[0]
        self.assertEqual(
            row["duration"], min(row["T"] + (row["b"] - row["h"]), row["max_bids"])
        )
        self.assertTrue(bool(row["censored_sensitivity"]))
        self.assertFalse(bool(row["first_end_sensitivity"]))
        trials_at_cap = trials[trials["turn_capped"].astype(bool)]
        self.assertFalse(trials_at_cap.empty)
        self.assertTrue(trials_at_cap["waste_is_lower_bound"].astype(bool).all())

        for csv_path in out.glob("*.csv"):
            frame = pd.read_csv(csv_path)
            self.assertIn("input_run_uids", frame.columns, csv_path.name)
            self.assertTrue(frame["input_run_uids"].astype(str).str.contains("CHILD-0").all())
        self.assertIn("CHILD-0", (out / "summary.md").read_text())

        after = {path: path.read_bytes() for path in before}
        self.assertEqual(before, after)

    def test_resume_chain_keeps_finished_cells_and_uses_newest_matching_config(self):
        for index, child in enumerate(self.children):
            meta = json.loads((child / "run.json").read_text())
            meta["config_hash"] = f"HASH-{index}"
            (child / "run.json").write_text(json.dumps(meta))
            (child / "status.json").write_text(json.dumps({"state": "finished"}))
        resumed = self.root / "resumed_parent"
        resumed.mkdir()
        (resumed / "run.json").write_text(
            json.dumps({"run_uid": "RESUMED", "lineage": {"resumed_from": "PARENT-UID"}})
        )
        replacement = self.root / "replacement_child"
        replacement.mkdir()
        (replacement / "run.json").write_text(
            json.dumps(
                {
                    "run_uid": "REPLACEMENT",
                    "config_hash": "HASH-1",
                    "lineage": {"parent_run_uid": "RESUMED"},
                }
            )
        )
        (replacement / "status.json").write_text(json.dumps({"state": "finished"}))
        children, parents = sweep_sources(resumed)
        self.assertEqual(parents, ["RESUMED", "PARENT-UID"])
        self.assertEqual(len(children), 4)
        self.assertIn(replacement.resolve(), children)
        self.assertNotIn(self.children[1].resolve(), children)


if __name__ == "__main__":
    unittest.main()
