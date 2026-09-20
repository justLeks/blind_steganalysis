"""--cover-list support: embed only listed covers, filter change records by cover name.

Run: python3 -m unittest discover -s tests -v
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
from read_changes import filter_record_paths_by_cover_list, iter_record_paths


def _make_covers(src, count=3, size=32):
    rng = np.random.default_rng(1)
    for i in range(count):
        Image.fromarray(rng.integers(0, 256, size=(size, size), dtype=np.uint8), mode="L").save(src / f"c{i}.png")


class TestCoverList(unittest.TestCase):
    def test_embed_only_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); src = root / "covers"; src.mkdir(); _make_covers(src)
            records = embed_and_log(src_dir=src, dst_dir=root / "out", method="HUGO", alpha=0.4,
                                    seed=1, cover_list={"c0.png", "c2.png"})
            self.assertEqual([r["cover_image"] for r in records], ["c0.png", "c2.png"])
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(len(manifest), 2)
            paths = iter_record_paths(root / "out")
            kept = filter_record_paths_by_cover_list(paths, {"c2.png"})
            self.assertEqual([p.name for p in kept], ["c2_changes.npz"])
            self.assertEqual(filter_record_paths_by_cover_list(paths, None), paths)

    def test_missing_listed_cover_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); src = root / "covers"; src.mkdir(); _make_covers(src)
            with self.assertRaises(FileNotFoundError):
                embed_and_log(src_dir=src, dst_dir=root / "out", method="HUGO", alpha=0.4,
                              seed=1, cover_list={"c0.png", "nope.png"})


if __name__ == "__main__":
    unittest.main()
