"""Probing runner: K=1 row schema unchanged, K>1 aggregates, workers bit-identical to sequential.

Run: python3 -m unittest discover -s tests -v
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

import run_probing_experiment as RP
from embedding import embed_and_log
from read_changes import iter_record_paths, load_change_record

LEGACY_KEYS = [
    "config_id", "record_name", "record_path", "cover_relative_path", "stego_relative_path", "method",
    "alpha", "record_seed", "probe_seed", "probe_distribution", "probe_image_source", "image_height",
    "image_width", "total_pixels", "carrier_pixels", "used_pixels", "used_pixel_fraction",
    "used_pixel_percentage", "probe_budget_fraction", "probe_budget_percentage", "probe_budget_pixels",
    "unique_probed_pixels", "guessed_carrier_pixels", "precision_hit_rate", "carrier_recall",
    "incremental_probe_pixels", "incremental_guessed_pixels", "incremental_precision_hit_rate",
]


def _fixture(root):
    src = root / "covers"; src.mkdir()
    rng = np.random.default_rng(3)
    for i in range(2):
        Image.fromarray(rng.integers(0, 256, size=(32, 32), dtype=np.uint8), mode="L").save(src / f"c{i}.png")
    out = root / "stego"
    embed_and_log(src_dir=src, dst_dir=out, method="HUGO", alpha=0.4, seed=1)
    return src, out


def _cfg(src, repeats):
    return dict(distribution="texture_energy", distribution_params={"gamma": 1.0, "floor": 1e-6},
                image_source="stego", cover_root=src, budget_fractions=[0.05, 0.2, 0.5], repeats=repeats)


class TestRepeats(unittest.TestCase):
    def test_k1_schema_unchanged_and_k3_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = _fixture(pathlib.Path(tmp))
            rec = load_change_record(iter_record_paths(out)[0])
            rows1 = RP.probe_record(rec, out, 512, 99, **_cfg(src, 1))
            self.assertEqual(list(rows1[0].keys()), LEGACY_KEYS)
            rows3 = RP.probe_record(rec, out, 512, 99, **_cfg(src, 3))
            for k in LEGACY_KEYS + ["repeats", "sd_carrier_recall", "min_carrier_recall", "max_carrier_recall",
                                    "sd_guessed_carrier_pixels"]:
                self.assertIn(k, rows3[0])
            for r1, r3 in zip(rows1, rows3):
                self.assertEqual(r3["repeats"], 3)
                self.assertLessEqual(r3["min_carrier_recall"], r3["carrier_recall"])
                self.assertLessEqual(r3["carrier_recall"], r3["max_carrier_recall"])
                self.assertEqual(r3["probe_seed"], r1["probe_seed"])  # draw 0 of K=3 is the K=1 draw
                self.assertLessEqual(r3["min_carrier_recall"], r1["carrier_recall"])
                self.assertLessEqual(r1["carrier_recall"], r3["max_carrier_recall"])

    def test_workers_match_sequential(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = _fixture(pathlib.Path(tmp))
            cfg = _cfg(src, 2)
            tasks = [(rp, out, 512, 99, cfg) for rp in iter_record_paths(out)]
            seq = [RP._probe_task(t) for t in tasks]
            par = RP.run_tasks(tasks, workers=2)
            self.assertEqual(seq, par)
            with self.assertRaises(ValueError):
                RP.run_tasks(tasks, workers=0)


if __name__ == "__main__":
    unittest.main()
