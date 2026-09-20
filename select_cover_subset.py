"""Draw a seeded random subset of cover file names and write it as a cover list.

python3 select_cover_subset.py --cover-root ALASKA_v2_TIFF_512_GrayScale_10K --n 1000 --seed 12345 \
    --out data/subsets/alaska10k_n1000_seed12345.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embedding import IMAGE_EXTENSIONS

DEFAULT_OUT = Path("data/subsets/alaska10k_n1000_seed12345.txt")


def select_subset(cover_root: Path, n: int, seed: int) -> list[str]:
    names = sorted(p.name for p in Path(cover_root).iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if n < 1 or n > len(names):
        raise ValueError(f"n must be in [1, {len(names)}], got {n}.")
    idx = np.random.default_rng(seed).choice(len(names), size=n, replace=False)
    return sorted(names[i] for i in idx)


def write_subset(names: list[str], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(names) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cover-root", type=Path, default=Path("ALASKA_v2_TIFF_512_GrayScale_10K"))
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    names = select_subset(args.cover_root, args.n, args.seed)
    write_subset(names, args.out)
    print(f"Wrote {len(names)} names to {args.out}")


if __name__ == "__main__":
    main()
