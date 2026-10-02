import json
import tempfile
import unittest
from pathlib import Path

from dollar_auction_tools.solver_ref import generate_vectors, proposition_first_bid, solve


class SolverReferenceTests(unittest.TestCase):
    def test_proposition_3_1_full_grid(self) -> None:
        for s in (3, 5, 10):
            for b in range(1, 4 * s):
                x1, second_reply = solve(s, b)
                self.assertEqual(x1, proposition_first_bid(s, b), (s, b))
                self.assertIsNone(second_reply, (s, b))

    def test_anchor_values(self) -> None:
        self.assertEqual(solve(100, 100), (1, None))
        self.assertEqual(solve(100, 250), (52, None))

    def test_generator_writes_the_cross_check_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.json"
            generate_vectors(path)
            rows = json.loads(path.read_text())
        self.assertEqual(len(rows), sum(4 * s for s in (3, 5, 10)) + 2)
        self.assertEqual(rows[-1], {"s": 100, "b": 250, "x1": 52, "second_reply": None})


if __name__ == "__main__":
    unittest.main()
