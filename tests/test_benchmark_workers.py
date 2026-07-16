"""Property test: parallel benchmark (workers>1) is bit-identical to the sequential path.

Run: python3 -m unittest discover -s tests -v
Exercises conseal (numba JIT) in the parent and each spawned worker; may take ~30-60 s on first run.
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from embedding import embed_and_log
from run_probing_experiment import config_id
import run_localization_benchmark as B


def _make_covers(src_dir: pathlib.Path, count: int = 3, size: int = 64) -> None:
    rng = np.random.default_rng(20260716)
    for i in range(count):
        array = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
        Image.fromarray(array, mode="L").save(src_dir / f"cover_{i:02d}.png")


def _run_benchmark(cover_root, steg_root, output_root, workers: int) -> None:
    args = B.build_parser().parse_args(
        [
            "--steganogram-root", str(steg_root),
            "--cover-root", str(cover_root),
            "--output-root", str(output_root),
            "--methods", "HUGO",
            "--alphas", "0.4",
            "--bootstrap", "50",
            "--workers", str(workers),
        ]
    )
    B.evaluate(args)


class TestParallelBenchmarkBitExact(unittest.TestCase):
    def test_workers_match_sequential(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            covers = tmp_path / "covers"
            covers.mkdir()
            _make_covers(covers)
            steg_root = tmp_path / "steganograms"
            embed_and_log(
                src_dir=covers,
                dst_dir=steg_root / config_id("HUGO", 0.4),
                method="HUGO",
                alpha=0.4,
                seed=12345,
            )

            out_seq = tmp_path / "bench_seq"
            out_par = tmp_path / "bench_par"
            _run_benchmark(covers, steg_root, out_seq, workers=1)
            _run_benchmark(covers, steg_root, out_par, workers=2)

            for name in ("per_image.csv", "summary.csv"):
                seq_text = (out_seq / name).read_text(encoding="utf-8")
                par_text = (out_par / name).read_text(encoding="utf-8")
                self.assertEqual(seq_text, par_text, f"{name} differs between workers=1 and workers=2")
            # Sanity: the run actually produced rows for all 12 localizers x 3 images.
            n_rows = len((out_seq / "per_image.csv").read_text(encoding="utf-8").strip().splitlines()) - 1
            self.assertEqual(n_rows, 12 * 3)


if __name__ == "__main__":
    unittest.main()
