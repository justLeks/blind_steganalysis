"""Paired per-image significance tests: the proposed map against every baseline.

For each (method, alpha) and metric, d_i = metric_target(i) - metric_baseline(i) over the images
both localizers scored; two-sided Wilcoxon signed-rank test, Holm-adjusted across the baselines
of one (method, alpha, metric) family; effect size = median d with a bootstrap CI of mean d.

python3 run_paired_tests.py --benchmark experiments/dessert2026/benchmark --out experiments/dessert2026/paired_tests
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

import localization_metrics as M

DEFAULT_TARGET = "texture_energy_stego"
DEFAULT_BASELINES = ["uniform", "gradient_magnitude_stego", "laplacian_residual_stego", "local_variance_stego",
                     "local_entropy_stego", "wavelet_energy_stego", "srm_residual_stego"]
DEFAULT_METRICS = ["average_precision", "recall_at_1", "recall_at_5", "recall_at_10", "naurc_10"]


def holm_adjust(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def wilcoxon_p(diff) -> tuple[float, int]:
    """Two-sided Wilcoxon signed-rank p-value over the non-NaN differences; p = 1 if all zero."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[~np.isnan(d)]
    if d.size == 0 or np.all(d == 0.0):
        return 1.0, int(d.size)
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue), int(d.size)


def load_per_image(path: Path, localizers: set[str], metrics: list[str]) -> dict:
    """(method, alpha, localizer) -> {record_name: {metric: value}} for the requested localizers."""
    table: dict = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["localizer"] not in localizers:
                continue
            key = (row["method"], float(row["alpha"]), row["localizer"])
            table.setdefault(key, {})[row["record_name"]] = {m: float(row[m]) for m in metrics}
    return table


def paired_rows(table: dict, target: str, baselines: list[str], metrics: list[str],
                bootstrap: int, seed: int) -> list[dict]:
    out: list[dict] = []
    configs = sorted({(m, a) for (m, a, loc) in table if loc == target})
    for method, alpha in configs:
        tgt = table[(method, alpha, target)]
        for metric in metrics:
            family: list[dict] = []
            for baseline in baselines:
                ref = table.get((method, alpha, baseline))
                if ref is None:
                    continue
                names = sorted(set(tgt) & set(ref))
                d = np.array([tgt[n][metric] - ref[n][metric] for n in names], dtype=np.float64)
                p, n_eff = wilcoxon_p(d)
                mean, lo, hi = M.bootstrap_ci(d, bootstrap, seed=seed)
                family.append({
                    "method": method, "alpha": float(alpha), "metric": metric, "target": target,
                    "baseline": baseline, "n": n_eff,
                    "median_diff": float(np.nanmedian(d)) if d.size else float("nan"),
                    "mean_diff": mean, "mean_diff_ci_lo": lo, "mean_diff_ci_hi": hi, "p_raw": p,
                })
            for row, p_adj in zip(family, holm_adjust([r["p_raw"] for r in family])):
                row["p_holm"] = float(p_adj)
            out.extend(family)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", type=Path, required=True, help="Directory holding per_image.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    table = load_per_image(args.benchmark / "per_image.csv", set(args.baselines) | {args.target}, args.metrics)
    rows = paired_rows(table, args.target, args.baselines, args.metrics, args.bootstrap, args.seed)
    if not rows:
        raise ValueError(f"No paired rows produced from {args.benchmark / 'per_image.csv'}")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "paired_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "metadata.json").write_text(json.dumps({
        "benchmark": str(args.benchmark.resolve()), "target": args.target, "baselines": args.baselines,
        "metrics": args.metrics, "test": "wilcoxon signed-rank, two-sided, zero_method=wilcox",
        "correction": "holm within (method, alpha, metric)", "bootstrap": args.bootstrap, "seed": args.seed,
        "n_rows": len(rows)}, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.out / 'paired_tests.csv'}")


if __name__ == "__main__":
    main()
