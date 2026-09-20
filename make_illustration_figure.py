"""Fig. 1: cover crop, carrier mask, and top-B texture-energy probes at B = 1 %, 5 %, 10 %.

python3 make_illustration_figure.py --root experiments/dessert2026 --cover-root ALASKA_v2_TIFF_512_GrayScale_10K
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

from embedding import to_u8_array
from localization_metrics import budget_pixels
from read_changes import build_adaptive_probabilities, iter_record_paths, load_change_record
from run_probing_experiment import config_id


def top_b_mask(scores, b, seed=0):
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(scores.size), -scores))
    mask = np.zeros(scores.size, dtype=bool)
    mask[order[:b]] = True
    return mask


def densest_crop(mask2d, size):
    density = uniform_filter(mask2d.astype(np.float64), size=size, mode="constant")
    r, c = np.unravel_index(np.argmax(density), density.shape)
    r0 = int(np.clip(r - size // 2, 0, mask2d.shape[0] - size))
    c0 = int(np.clip(c - size // 2, 0, mask2d.shape[1] - size))
    return slice(r0, r0 + size), slice(c0, c0 + size)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("experiments/dessert2026"))
    p.add_argument("--cover-root", type=Path, default=Path("ALASKA_v2_TIFF_512_GrayScale_10K"))
    p.add_argument("--method", default="SUNIWARD")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)

    cdir = args.root / "steganograms" / config_id(args.method, args.alpha)
    rec = load_change_record(iter_record_paths(cdir)[args.index])
    cover = to_u8_array(args.cover_root / rec["cover_relative_path"])
    stego = to_u8_array(cdir / rec["stego_relative_path"])
    shape = tuple(rec["image_shape"])
    n = shape[0] * shape[1]
    carriers = np.zeros(n, dtype=bool)
    carriers[rec["changed_flat_indices"]] = True
    carriers2d = carriers.reshape(shape)
    scores = build_adaptive_probabilities("texture_energy", stego)
    rs, cs = densest_crop(carriers2d, args.crop)

    panels = [("cover", cover[rs, cs].astype(np.float64) / 255.0),
              ("carrier set C", np.where(carriers2d[rs, cs], 0.0, 1.0))]
    for B in args.budgets:
        sel = top_b_mask(scores, budget_pixels(n, B)).reshape(shape)
        img = np.ones(shape)
        img[sel & ~carriers2d] = 0.75
        img[sel & carriers2d] = 0.0
        hits = int((sel & carriers2d).sum())
        panels.append((f"B = {100 * B:g} %, hits = {hits}", img[rs, cs]))

    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.7))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    out = args.root / "comparison" / "figures" / "fig1_illustration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out} (record {rec['record_path'].name}, |C| = {int(carriers.sum())}, crop rows {rs.start}:{rs.stop}, cols {cs.start}:{cs.stop})")


if __name__ == "__main__":
    main()
