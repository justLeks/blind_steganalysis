"""Paper tables for the DESSERT-2026 revision, emitted as CSV (for the docx builder) and Markdown.

python3 make_paper_tables.py --root experiments/dessert2026
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from make_comparison_figures import STEGO_MAPS, pretty_name, read_rows

METHOD_ORDER = ["HUGO", "MIPOD", "SUNIWARD"]
METHOD_TEXT = {"HUGO": "HUGO", "MIPOD": "MiPOD", "SUNIWARD": "S-UNIWARD"}
LOCALIZER_ORDER = ["uniform"] + [f"{m}_stego" for m in STEGO_MAPS] + ["oracle"]


def f3(x): return f"{float(x):.3f}"
def f1(x): return f"{float(x):.1f}"
def f2(x): return f"{float(x):.2f}"
def fp(x): return "<0.001" if float(x) < 0.001 else f"{float(x):.3f}"
def ci(lo, hi): return f"[{f3(lo)}, {f3(hi)}]"
def tag(alpha): return ("%g" % alpha).replace(".", "p")


def write(rows, header, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    out_csv.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def methods_present(lk):
    return [m for m in METHOD_ORDER if any(k[0] == m for k in lk)]


def table1(lk, alphas, out):
    methods = methods_present(lk)
    rows = []
    for a in alphas:
        row = [f"{a:g}"]
        row += [f"{float(lk[(m, a, 'uniform')]['mean_prevalence']) * 512 * 512:.0f}" for m in methods]
        row += [f"{100 * float(lk[(m, a, 'uniform')]['mean_prevalence']):.3f}" for m in methods]
        rows.append(row)
    header = ["α, bpp"] + [f"|C|, {METHOD_TEXT[m]}" for m in methods] + [f"π, %, {METHOD_TEXT[m]}" for m in methods]
    write(rows, header, out)


def table2(lk, probing_rows, alpha, labels, out):
    methods = methods_present(lk)
    rows = []
    for lab in labels:
        B = float(lab.replace("p", "."))
        row = [f"{B:g}"]
        for m in methods:
            r = lk[(m, alpha, "texture_energy_stego")]
            row += [f3(r[f"mean_recall_at_{lab}"]), ci(r[f"recall_at_{lab}_ci_lo"], r[f"recall_at_{lab}_ci_hi"]),
                    f3(r[f"mean_precision_at_{lab}"]), f1(r[f"mean_lift_at_{lab}"])]
        rows.append(row)
    # secondary variant: gamma = 1 weighted sampling with K repeated draws, recall at every budget
    samp = {(r["method"], float(r["alpha"]), round(float(r["probe_budget_fraction"]), 6)): r for r in probing_rows}
    for lab in labels:
        B = float(lab.replace("p", ".")) / 100.0
        row = [f"{100 * B:g} (γ=1 sampling)"]
        for m in methods:
            r = samp.get((m, alpha, round(B, 6)))
            if r is None:
                row += ["", "", "", ""]
            else:
                rec = float(r["mean_carrier_recall"])
                sd = r.get("mean_sd_carrier_recall", "")
                row += [f3(rec), f"sd {f3(sd)}" if sd != "" else "", f3(r["mean_precision_hit_rate"]), f1(rec / B)]
        rows.append(row)
    header = ["B, %"] + [f"{c}, {METHOD_TEXT[m]}" for m in methods for c in ("R", "95% CI", "Q", "L")]
    write(rows, header, out)


def table3(lk, alpha, out):
    methods = methods_present(lk)
    rows = []
    for loc in LOCALIZER_ORDER:
        row = [pretty_name(loc)]
        for m in methods:
            r = lk[(m, alpha, loc)]
            u = lk[(m, alpha, "uniform")]
            o = lk[(m, alpha, "oracle")]
            ap, ap_u, ap_o = (float(x["mean_average_precision"]) for x in (r, u, o))
            eta = (ap - ap_u) / (ap_o - ap_u) if ap_o > ap_u else float("nan")
            row += [f3(ap), ci(r["ap_ci_lo"], r["ap_ci_hi"]), f2(eta), f3(r["mean_naurc_10"])]
        rows.append(row)
    header = ["localizer"] + [f"{c}, {METHOD_TEXT[m]}" for m in methods for c in ("AP", "95% CI", "η", "nAURC(0.10)")]
    write(rows, header, out)


def table4(paired, alpha, methods, out):
    by = {(r["method"], float(r["alpha"]), r["metric"], r["baseline"]): r for r in paired}
    rows = []
    for base in LOCALIZER_ORDER[:-1]:
        if base == "texture_energy_stego":
            continue
        row = [pretty_name(base)]
        for m in methods:
            a = by[(m, alpha, "average_precision", base)]
            r1 = by[(m, alpha, "recall_at_1", base)]
            row += [f"{float(a['median_diff']):+.3f}", fp(a["p_holm"]), f"{float(r1['median_diff']):+.3f}", fp(r1["p_holm"])]
        rows.append(row)
    header = ["baseline"] + [f"{c}, {METHOD_TEXT[m]}" for m in methods for c in ("median ΔAP", "p", "median ΔR(1%)", "p")]
    write(rows, header, out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("experiments/dessert2026"))
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--table3-alphas", type=float, nargs="+", default=[0.005, 0.01, 0.05])
    args = p.parse_args(argv)
    summary = read_rows(args.root / "benchmark" / "summary.csv")
    lk = {(r["method"], float(r["alpha"]), r["localizer"]): r for r in summary}
    alphas = sorted({float(r["alpha"]) for r in summary})
    labels = sorted({c[len("mean_recall_at_"):] for c in summary[0] if c.startswith("mean_recall_at_")},
                    key=lambda s: float(s.replace("p", ".")))
    paper = args.root / "paper"
    table1(lk, alphas, paper / "table1_carriers.csv")
    table2(lk, read_rows(args.root / "probing_texture_energy" / "probe_summary.csv"), args.alpha, labels,
           paper / f"table2_budget_curve_alpha{tag(args.alpha)}.csv")
    for a in args.table3_alphas:
        table3(lk, a, paper / f"table3_localizers_alpha{tag(a)}.csv")
    table4(read_rows(args.root / "paired_tests" / "paired_tests.csv"), args.alpha, methods_present(lk),
           paper / f"table4_paired_alpha{tag(args.alpha)}.csv")
    print(f"Wrote tables to {paper}")


if __name__ == "__main__":
    main()
