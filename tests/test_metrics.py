"""Property tests for the localization metrics. Run: python3 -m unittest discover -s tests -v"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import localization_metrics as M


class TestAveragePrecision(unittest.TestCase):
    def test_worked_example(self):
        # scores .9 .8 .7 .6, labels 1 0 1 0 -> AP = 0.5*1 + 0.5*(2/3)
        ap = M.average_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0]))
        self.assertAlmostEqual(ap, 0.5 * 1.0 + 0.5 * (2.0 / 3.0), places=12)

    def test_perfect_and_random(self):
        self.assertAlmostEqual(M.average_precision(np.array([4, 3, 2, 1]), np.array([1, 1, 0, 0])), 1.0, places=12)
        rng = np.random.default_rng(0)
        labels = rng.random(100_000) < 0.05
        ap = M.average_precision(rng.random(100_000), labels)
        self.assertAlmostEqual(ap, 0.05, delta=0.01)  # AP of a random ranker ~ prevalence

    def test_no_positives_is_nan(self):
        self.assertTrue(np.isnan(M.average_precision(np.array([1.0, 2.0]), np.array([0, 0]))))


class TestRocAuc(unittest.TestCase):
    def test_perfect_and_chance(self):
        self.assertAlmostEqual(M.roc_auc(np.array([4, 3, 2, 1]), np.array([1, 1, 0, 0])), 1.0, places=12)
        rng = np.random.default_rng(1)
        labels = rng.random(50_000) < 0.1
        self.assertAlmostEqual(M.roc_auc(rng.random(50_000), labels), 0.5, delta=0.02)


class TestRecallAndLift(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.n = 200_000
        self.labels = self.rng.random(self.n) < 0.05

    def test_uniform_recall_approx_fraction(self):
        recalls = M.recall_at_budgets(self.rng.random(self.n), self.labels,
                                      np.array([0.01, 0.1, 0.5]), rng=self.rng)
        for r, f in zip(recalls, [0.01, 0.1, 0.5]):
            self.assertAlmostEqual(r, f, delta=0.01)

    def test_recall_monotone_in_budget(self):
        scores = self.rng.random(self.n)  # any score map
        fr = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
        recalls = M.recall_at_budgets(scores, self.labels, fr, rng=self.rng)
        self.assertTrue(np.all(np.diff(recalls) >= -1e-12), recalls)

    def test_lift_is_recall_over_fraction(self):
        recalls = np.array([0.2, 0.3])
        fr = np.array([0.1, 0.1])
        np.testing.assert_allclose(M.lift_at_budgets(recalls, fr), [2.0, 3.0])

    def test_informative_scores_beat_random(self):
        # Scores correlated with labels -> recall@10% well above 0.1.
        scores = self.labels.astype(float) + 0.3 * self.rng.random(self.n)
        recalls = M.recall_at_budgets(scores, self.labels, np.array([0.1]), rng=self.rng)
        self.assertGreater(recalls[0], 0.5)


class TestBootstrapCI(unittest.TestCase):
    def test_ci_contains_mean_and_drops_nan(self):
        vals = np.r_[np.full(100, 0.5), np.nan]
        mean, lo, hi = M.bootstrap_ci(vals, n_boot=500, seed=0)
        self.assertAlmostEqual(mean, 0.5, places=6)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)


if __name__ == "__main__":
    unittest.main()
