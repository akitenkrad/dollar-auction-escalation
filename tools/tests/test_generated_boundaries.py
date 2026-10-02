import unittest

from dollar_auction_tools.solver_ref import proposition_first_bid, solve


class GeneratedBoundaryTests(unittest.TestCase):
    def test_invalid_value_and_budget_raise_value_error(self) -> None:
        for s, b in ((1, 1), (0, 1), (3, 0), (3, -1)):
            with self.subTest(s=s, b=b):
                with self.assertRaises(ValueError):
                    solve(s, b)
                with self.assertRaises(ValueError):
                    proposition_first_bid(s, b)


if __name__ == "__main__":
    unittest.main()
