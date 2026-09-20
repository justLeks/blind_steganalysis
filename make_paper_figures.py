"""Paper figures for the DESSERT-2026 revision (grayscale, 300 dpi).

Fig. 2  recall vs budget (log x): texture_energy, best other map, uniform, oracle; one panel per embedder.
Fig. 3  lift@1% vs payload (log-log): all seven stego maps + oracle; one panel per embedder.
Fig. 4  cover-vs-stego paired delta-AP with 95% CI per map at one payload.

python3 make_paper_figures.py --root experiments/dessert2026
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import localization_metrics as M
from make_comparison_figures import STEGO_MAPS, method_label, pretty_name, read_rows, style_for

METHOD_ORDER = ["HUGO", "MIPOD", "SUNIWARD"]


def alpha_tag(alpha: float) -> str:
    return ("%g" % alpha).replace(".", "p")


def lookup(summary_rows):
    return {(r["method"], float(r["alpha"]), r["localizer"]): r for r in summary_rows}


def budgets_from(summary_rows):
    labels = sorted({c[len("mean_recall_at_"):] for c in summary_rows[0] if c.startswith("mean_recall_at_")},
                    key=lambda s: float(s.replace("p", ".")))
    return labels, np.array([float(s.replace("p", ".")) for s in labels])


def methods_present(lk):
    return [m for m in METHOD_ORDER if any(k[0] == m for k in lk)]


def best_other_map(lk, method, alpha):
    cands = [f"{m}_stego" for m in STEGO_MAPS if m != "texture_energy" and (method, alpha, f"{m}_stego") in lk]
    return max(cands, key=lambda loc: float(lk[(method, alpha, loc)]["mean_average_precision"]))


def fig2_recall_vs_budget(lk, alpha, out, dpi):
    labels, pcts = budgets_from(list(lk.values()))
    methods = methods_present(lk)
    fig, axes = plt.subplots(1, len(methods), figsize=(7.0, 2.4), sharey=True, squeeze=False)
    for ax, method in zip(axes[0], methods):
        series = ["texture_energy_stego", best_other_map(lk, method, alpha), "uniform", "oracle"]
        for loc in series:
            r = lk[(method, alpha, loc)]
            y = np.array([float(r[f"mean_recall_at_{l}"]) for l in labels])
            lo = np.array([float(r[f"recall_at_{l}_ci_lo"]) for l in labels])
            hi = np.array([float(r[f"recall_at_{l}_ci_hi"]) for l in labels])
            ax.plot(pcts, y, label=pretty_name(loc), **style_for(loc))
            ax.fill_between(pcts, lo, hi, color="0.5", alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(pcts)
        ax.set_xticklabels([l.replace("p", ".") for l in labels], fontsize=6)
        ax.minorticks_off()
        ax.set_title(method_label(method), fontsize=8)
        ax.set_xlabel("budget B, %", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, color="0.85", linewidth=0.5)
    axes[0][0].set_ylabel("recall R(B)", fontsize=7)
    axes[0][0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def fig3_lift_vs_alpha(lk, budget_label, out, dpi):
    alphas = sorted({a for (_, a, _) in lk})
    methods = methods_present(lk)
    fig, axes = plt.subplots(1, len(methods), figsize=(7.0, 2.4), sharey=True, squeeze=False)
    for ax, method in zip(axes[0], methods):
        for loc in [f"{m}_stego" for m in STEGO_MAPS] + ["oracle"]:
            if (method, alphas[0], loc) not in lk:
                continue
            y = [float(lk[(method, a, loc)][f"mean_lift_at_{budget_label}"]) for a in alphas]
            ax.plot(alphas, y, label=pretty_name(loc), **style_for(loc))
        ax.axhline(1.0, color="0.3", linewidth=0.6, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(method_label(method), fontsize=8)
        ax.set_xlabel("payload α, bpp", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, color="0.85", linewidth=0.5)
    axes[0][0].set_ylabel(f"lift L({budget_label.replace('p', '.')} %)", fontsize=7)
    axes[0][-1].legend(fontsize=5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def cover_vs_stego_deltas(per_image_path, alpha, bootstrap=2000, seed=0):
    """(method, map) -> (mean delta-AP, ci_lo, ci_hi, n) for stego-side minus cover-side scoring."""
    ap = {}
    with Path(per_image_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if abs(float(row["alpha"]) - alpha) > 1e-12 or row["localizer"] in ("uniform", "oracle"):
                continue
            dist, _, source = row["localizer"].rpartition("_")
            ap.setdefault((row["method"], dist, source), {})[row["record_name"]] = float(row["average_precision"])
    out = {}
    for method in METHOD_ORDER:
        for m in STEGO_MAPS:
            if (method, m, "stego") not in ap:
                continue
            s, c = ap[(method, m, "stego")], ap[(method, m, "cover")]
            d = np.array([s[k] - c[k] for k in sorted(set(s) & set(c))])
            mean, lo, hi = M.bootstrap_ci(d, bootstrap, seed=seed)
            out[(method, m)] = (mean, lo, hi, int(np.sum(~np.isnan(d))))
    return out


def fig4_cover_vs_stego(deltas, out, dpi):
    methods = [m for m in METHOD_ORDER if any(k[0] == m for k in deltas)]
    fig, axes = plt.subplots(1, len(methods), figsize=(7.0, 2.2), sharey=True, squeeze=False)
    x = np.arange(len(STEGO_MAPS))
    for ax, method in zip(axes[0], methods):
        means = [deltas[(method, m)][0] for m in STEGO_MAPS]
        los = [deltas[(method, m)][0] - deltas[(method, m)][1] for m in STEGO_MAPS]
        his = [deltas[(method, m)][2] - deltas[(method, m)][0] for m in STEGO_MAPS]
        ax.bar(x, means, yerr=[los, his], color="0.6", edgecolor="0.2", linewidth=0.6, capsize=2)
        ax.axhline(0, color="0.2", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_name(f"{m}_stego") for m in STEGO_MAPS], rotation=60, ha="right", fontsize=5)
        ax.set_title(method_label(method), fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0][0].set_ylabel("ΔAP (stego − cover)", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("experiments/dessert2026"))
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--lift-budget", default="1", help="Budget column label for Fig. 3 (e.g. '1' or '0p5').")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    lk = lookup(read_rows(args.root / "benchmark" / "summary.csv"))
    figs = args.root / "comparison" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    tag = alpha_tag(args.alpha)
    fig2_recall_vs_budget(lk, args.alpha, figs / f"fig2_recall_vs_budget_alpha{tag}.png", args.dpi)
    fig3_lift_vs_alpha(lk, args.lift_budget, figs / f"fig3_lift_at_{args.lift_budget}_vs_alpha.png", args.dpi)
    deltas = cover_vs_stego_deltas(args.root / "benchmark" / "per_image.csv", args.alpha)
    fig4_cover_vs_stego(deltas, figs / f"fig4_cover_vs_stego_alpha{tag}.png", args.dpi)
    with (figs / f"fig4_cover_vs_stego_alpha{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "map", "mean_delta_ap", "ci_lo", "ci_hi", "n"])
        for (method, m), (mean, lo, hi, n) in sorted(deltas.items()):
            w.writerow([method, m, f"{mean:.6f}", f"{lo:.6f}", f"{hi:.6f}", n])
    print(f"Wrote figures to {figs}")


if __name__ == "__main__":
    main()
