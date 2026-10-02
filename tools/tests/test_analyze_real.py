import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dollar_auction_tools.analyze import analyze_sweep


class RealSmokeAnalyzeTests(unittest.TestCase):
    def test_real_shaped_sweep_analyzes_without_mutating_inputs(self):
        fixture = Path(__file__).parent / "fixtures" / "real_smoke"
        parent = fixture / "pilot_20261002_105527_1583ad74_3877"
        before = {
            path.relative_to(fixture): path.read_bytes()
            for path in fixture.rglob("*")
            if path.is_file()
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            analyze_sweep(parent, out=output)

            expected = {
                "analysis.json",
                "summary.md",
                "trials.csv",
                "km_survival.csv",
                "cox_ph.csv",
                "logistic_first_end.csv",
                "first_bid_summary.csv",
                "sensitivity.csv",
                "sensitivity_models.csv",
                "violation_rates.csv",
                "usage.csv",
                "quiz_solve_rates.csv",
                "qre_lambda.csv",
                "km_curves.png",
                "first_bid.png",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)

            trials = pd.read_csv(output / "trials.csv")
            anchors = trials.groupby("b")[
                ["x1_star_drop_tie", "x1_star_mixed_tie"]
            ].first()
            self.assertEqual(tuple(anchors.loc[100]), (1, 1))
            self.assertEqual(tuple(anchors.loc[250]), (52, 50))

            censored = trials[trials["censored"]]
            self.assertEqual(
                set(zip(censored["b"], censored["T"])),
                {(100, 68), (100, 36), (250, 104)},
            )
            self.assertFalse(censored["event_observed"].any())
            self.assertTrue((censored["duration"] == censored["T"]).all())

            violations = pd.read_csv(output / "violation_rates.csv")
            format_rows = violations[violations["kind"] == "format"].set_index("b")
            self.assertEqual(int(format_rows.loc[100, "n_violations"]), 0)
            self.assertEqual(int(format_rows.loc[250, "n_violations"]), 1)

            self.assertTrue((output / "summary.md").read_text().strip())

        after = {
            path.relative_to(fixture): path.read_bytes()
            for path in fixture.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
