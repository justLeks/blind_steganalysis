"""Benchmark per-image rows carry fractional budget labels, precision@B and nAURC columns.

Run: python3 -m unittest discover -s tests -v
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

import run_localization_benchmark as RB
from embedding import embed_and_log
from read_changes import iter_record_paths


class TestBenchmarkColumns(unittest.TestCase):
    def test_precision_and_naurc_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); src = root / "covers"; src.mkdir()
            rng = np.random.default_rng(5)
            Image.fromarray(rng.integers(0, 256, size=(32, 32), dtype=np.uint8), mode="L").save(src / "c0.png")
            cdir = root / "stego" / "hugo_alpha0p4"
            embed_and_log(src_dir=src, dst_dir=cdir, method="HUGO", alpha=0.4, seed=1)
            rp = iter_record_paths(cdir)[0]
            budgets = [0.005, 0.01, 0.1, 0.5]
            labels = [RB.M.budget_label(b) for b in budgets]
            self.assertEqual(labels, ["0p5", "1", "10", "50"])
            task = (rp, "hugo_alpha0p4", "HUGO", 0.4, ["uniform", "texture_energy_stego"], budgets, labels,
                    src, cdir, 12345, [0.1, 0.5])
            rows = {r["localizer"]: r for r in RB._evaluate_record(task)}
            u, t = rows["uniform"], rows["texture_energy_stego"]
            for lab in labels:
                self.assertIn(f"precision_at_{lab}", t)
            self.assertAlmostEqual(u["naurc_10"], 0.05, places=9)
            self.assertAlmostEqual(u["naurc_50"], 0.25, places=9)
            b = RB.M.budget_pixels(32 * 32, 0.1)
            self.assertAlmostEqual(t["precision_at_10"], t["recall_at_10"] * t["carrier_pixels"] / b, places=9)
            self.assertGreaterEqual(t["naurc_10"], 0.0)
            self.assertLessEqual(t["naurc_50"], 1.0)
            summary = RB.summarize(list(rows.values()), labels, 50, 0, [0.1, 0.5])
            self.assertIn("mean_precision_at_0p5", summary[0])
            self.assertIn("naurc_10_ci_hi", summary[0])


if __name__ == "__main__":
    unittest.main()
