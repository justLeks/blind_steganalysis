"""Property test: parallel embedding (workers>1) is bit-identical to the sequential path.

Run: python3 -m unittest discover -s tests -v
Exercises conseal (numba JIT) in the parent and each spawned worker; may take ~30-60 s on first run.
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from embedding import embed_and_log


def _make_covers(src_dir: pathlib.Path, count: int = 3, size: int = 64) -> None:
    """Write tiny deterministic grayscale covers (textured, so HUGO has non-flat costs)."""
    rng = np.random.default_rng(20260716)
    for i in range(count):
        array = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
        Image.fromarray(array, mode="L").save(src_dir / f"cover_{i:02d}.png")


class TestParallelEmbeddingBitExact(unittest.TestCase):
    def test_workers_match_sequential(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            src = tmp_path / "covers"
            src.mkdir()
            _make_covers(src)

            dst_seq = tmp_path / "out_seq"
            dst_par = tmp_path / "out_par"
            records_seq = embed_and_log(
                src_dir=src, dst_dir=dst_seq, method="HUGO", alpha=0.4, seed=123, workers=1
            )
            records_par = embed_and_log(
                src_dir=src, dst_dir=dst_par, method="HUGO", alpha=0.4, seed=123, workers=2
            )

            # Manifest: same order, same content.
            self.assertEqual(records_seq, records_par)
            manifest_seq = json.loads((dst_seq / "manifest.json").read_text(encoding="utf-8"))
            manifest_par = json.loads((dst_par / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_seq, manifest_par)
            self.assertEqual(len(records_seq), 3)

            # Every change record and stego image is bit-identical.
            for record in records_seq:
                npz_seq = np.load(dst_seq / record["change_record"])
                npz_par = np.load(dst_par / record["change_record"])
                self.assertEqual(sorted(npz_seq.files), sorted(npz_par.files))
                for key in npz_seq.files:
                    np.testing.assert_array_equal(npz_seq[key], npz_par[key])
                stego_seq = np.asarray(Image.open(dst_seq / record["stego_image"]))
                stego_par = np.asarray(Image.open(dst_par / record["stego_image"]))
                np.testing.assert_array_equal(stego_seq, stego_par)
                # The embedding must have actually changed pixels for the comparison to be meaningful.
                self.assertGreater(int(npz_seq["num_changed"]), 0)

    def test_invalid_workers_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            src = tmp_path / "covers"
            src.mkdir()
            _make_covers(src, count=1)
            with self.assertRaises(ValueError):
                embed_and_log(src_dir=src, dst_dir=tmp_path / "out", workers=0)


if __name__ == "__main__":
    unittest.main()
