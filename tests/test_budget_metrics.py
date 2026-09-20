"""Tests for budget_label, precision_at_budgets and normalized_aurc.

Run: python3 -m unittest discover -s tests -v
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import localization_metrics as M


class TestBudgetLabel(unittest.TestCase):
    def test_labels(self):
        self.assertEqual([M.budget_label(b) for b in [0.005, 0.01, 0.02, 0.1, 0.3, 0.5]],
                         ["0p5", "1", "2", "10", "30", "50"])


class TestPrecision(unittest.TestCase):
    def test_precision_from_recall(self):
        fractions = np.array([0.01, 0.1])
        total, n_pos = 10_000, 200
        recalls = np.array([0.25, 0.5])  # hits 50 of 100 probed, 100 of 1000 probed
        q = M.precision_at_budgets(recalls, fractions, n_pos=n_pos, total_pixels=total)
        np.testing.assert_allclose(q, [0.5, 0.1])

    def test_zero_carriers(self):
        q = M.precision_at_budgets(np.array([0.0]), np.array([0.01]), n_pos=0, total_pixels=100)
        np.testing.assert_allclose(q, [0.0])


class TestNormalizedAURC(unittest.TestCase):
    def test_uniform_gives_half_bmax(self):
        f = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        self.assertAlmostEqual(M.normalized_aurc(f, f, 0.1), 0.05, places=12)
        self.assertAlmostEqual(M.normalized_aurc(f, f, 0.5), 0.25, places=12)

    def test_perfect_recall_gives_one(self):
        f = np.array([0.05, 0.1])
        # ramp from (0,0) to (0.05,1) then flat: area = 0.025 + 0.05 = 0.075 -> /0.1 = 0.75
        self.assertAlmostEqual(M.normalized_aurc(np.array([1.0, 1.0]), f, 0.1), 0.75, places=12)

    def test_order_independent_and_validated(self):
        f = np.array([0.1, 0.01, 0.05]); r = np.array([0.6, 0.2, 0.4])
        self.assertAlmostEqual(M.normalized_aurc(r, f, 0.1), M.normalized_aurc(r[::-1], f[::-1], 0.1))
        with self.assertRaises(ValueError):
            M.normalized_aurc(r, f, 0.07)


if __name__ == "__main__":
    unittest.main()
