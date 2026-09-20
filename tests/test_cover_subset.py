"""Seeded cover-subset selection: deterministic, unique, existing names; cover-list round trip.

Run: python3 -m unittest discover -s tests -v
"""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from select_cover_subset import select_subset, write_subset
from embedding import read_cover_list


class TestSubset(unittest.TestCase):
    def test_deterministic_unique_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for i in range(20):
                (root / f"{i + 1:05d}.tif").write_bytes(b"")
            (root / "notes.txt").write_text("ignore me")
            a = select_subset(root, n=5, seed=1)
            b = select_subset(root, n=5, seed=1)
            c = select_subset(root, n=5, seed=2)
            self.assertEqual(a, b)
            self.assertNotEqual(a, c)
            self.assertEqual(len(set(a)), 5)
            self.assertEqual(a, sorted(a))
            self.assertTrue(all((root / name).exists() and name.endswith(".tif") for name in a))
            out = root / "subset.txt"
            write_subset(a, out)
            self.assertEqual(sorted(read_cover_list(out)), a)
            with self.assertRaises(ValueError):
                select_subset(root, n=21, seed=1)


if __name__ == "__main__":
    unittest.main()
