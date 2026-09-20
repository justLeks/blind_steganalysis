"""Paired Wilcoxon/Holm tests over per-image benchmark rows.

Run: python3 -m unittest discover -s tests -v
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from run_paired_tests import holm_adjust, paired_rows, wilcoxon_p


def _table(n=60, shift=0.05, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random(n) * 0.2
    names = [f"img{i}" for i in range(n)]

    def block(values):
        return {name: {"average_precision": float(v)} for name, v in zip(names, values)}

    return {
        ("HUGO", 0.01, "texture_energy_stego"): block(base + shift),
        ("HUGO", 0.01, "same"): block(base + shift),
        ("HUGO", 0.01, "worse"): block(base),
        ("HUGO", 0.01, "noisy"): block(base + shift + rng.normal(0, 0.001, n)),
    }


class TestPaired(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])  # step-down, monotone

    def test_wilcoxon_edge_cases(self):
        self.assertEqual(wilcoxon_p(np.zeros(10)), (1.0, 10))
        p, n = wilcoxon_p(np.r_[np.full(30, 0.1), np.nan])
        self.assertLess(p, 0.001)
        self.assertEqual(n, 30)

    def test_paired_rows(self):
        rows = paired_rows(_table(), "texture_energy_stego", ["same", "worse", "noisy"],
                           ["average_precision"], bootstrap=200, seed=0)
        by = {r["baseline"]: r for r in rows}
        self.assertEqual(by["same"]["p_raw"], 1.0)
        self.assertAlmostEqual(by["same"]["median_diff"], 0.0)
        self.assertLess(by["worse"]["p_holm"], 1e-6)
        self.assertAlmostEqual(by["worse"]["median_diff"], 0.05, places=9)
        self.assertGreater(by["noisy"]["p_holm"], 0.05)
        self.assertEqual(by["worse"]["n"], 60)
        self.assertEqual(set(rows[0]), {"method", "alpha", "metric", "target", "baseline", "n", "median_diff",
                                        "mean_diff", "mean_diff_ci_lo", "mean_diff_ci_hi", "p_raw", "p_holm"})


if __name__ == "__main__":
    unittest.main()
