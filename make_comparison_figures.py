"""Cross-distribution comparison report: tidy CSVs + grayscale figures.

Consumes the distribution benchmark (``run_localization_benchmark.py``) and the per-distribution
probing runs (``run_probing_experiment.py``) and writes, under ``--output-root``:

- ``comparison_ranking.csv``  -- per (method, alpha, localizer): AP/ROC/recall/lift with CIs, plus
  the joined random-floor AP, oracle AP, and ``ap_efficiency = (AP - floor) / (oracle - floor)``.
- ``comparison_sampling.csv`` -- per (distribution, method, alpha, budget): the gamma=1 stochastic
  sampling view (mean carrier recall / precision / lift) from the probing runs.
- ``figures/*.png``           -- grayscale-only metric charts (line style + marker + gray level
  distinguish series; no color anywhere, readable in B/W print).
- ``panels/*.png``            -- per-image grayscale panels: cover, true carrier mask, oracle
  probability map, and every score map, each independently percentile-normalised.
- ``metadata.json``           -- resolved configuration and inputs.

Run:  python3 make_comparison_figures.py           # after the benchmark + probing runs exist
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import localization_metrics as M
from embedding import to_u8_array
from read_changes import build_score_map, load_change_record, reconstruct_change_mask
from run_probing_experiment import alpha_tag, config_id
from run_localization_benchmark import SCORE_PARAMS
from selection_channel import change_probability_map


DEFAULT_PROBING_ROOTS = [
    Path("experiments/probing_only"),
    Path("experiments/probing_uniform"),
    Path("experiments/probing_laplacian_residual"),
    Path("experiments/probing_local_variance"),
    Path("experiments/probing_wavelet_energy"),
    Path("experiments/probing_srm_residual"),
]

# One fixed style per approach, identical across every figure. Gray level x line
# style x marker keeps 7 series distinguishable in black-and-white print.
GRAY_STYLES: dict[str, dict] = {
    "uniform": dict(color="0.45", linestyle=":", marker="", linewidth=1.4),
    "oracle": dict(color="0.0", linestyle="-", marker="", linewidth=2.2),
    "texture_energy": dict(color="0.0", linestyle="--", marker="o"),
    "laplacian_residual": dict(color="0.3", linestyle="-.", marker="s"),
    "local_variance": dict(color="0.0", linestyle="-", marker="^"),
    "wavelet_energy": dict(color="0.35", linestyle="--", marker="D"),
    "srm_residual": dict(color="0.55", linestyle="-", marker="v"),
}
PRETTY = {
    "uniform": "uniform (floor)",
    "oracle": "oracle (ceiling)",
    "texture_energy": "texture energy",
    "laplacian_residual": "Laplacian residual",
    "local_variance": "local variance",
    "wavelet_energy": "wavelet energy",
    "srm_residual": "SRM residual",
}
STEGO_MAPS = ["texture_energy", "laplacian_residual", "local_variance", "wavelet_energy", "srm_residual"]
MARKER_SIZE = 4.0


def style_for(localizer: str) -> dict:
    """Style dict for a localizer name (``<dist>_<stego|cover>``, ``uniform`` or ``oracle``)."""
    base = localizer
    open_marker = False
    for suffix in ("_stego", "_cover"):
        if localizer.endswith(suffix):
            base = localizer[: -len(suffix)]
            open_marker = suffix == "_cover"
    style = dict(GRAY_STYLES[base])
    style.setdefault("linewidth", 1.4)
    style["markersize"] = MARKER_SIZE
    if open_marker:
        style["markerfacecolor"] = "white"
    return style


def pretty_name(localizer: str) -> str:
    for suffix, tag in (("_stego", ""), ("_cover", " (cover)")):
        if localizer.endswith(suffix):
            return PRETTY[localizer[: -len(suffix)]] + tag
    return PRETTY[localizer]


def method_label(method: str) -> str:
    return {"HUGO": "HUGO", "MIPOD": "MiPOD"}.get(method.upper(), method)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def budget_pcts(summary_rows: list[dict]) -> list[int]:
    pcts = sorted(
        int(name.removeprefix("mean_recall_at_"))
        for name in summary_rows[0]
        if name.startswith("mean_recall_at_")
    )
    if not pcts:
        raise ValueError("No mean_recall_at_<pct> columns found in the benchmark summary.")
    return pcts


def load_probing(roots: list[Path]) -> list[dict]:
    """Rows from every probe_summary.csv, tagged with the run's distribution/source from metadata."""
    rows: list[dict] = []
    for root in roots:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        distribution = metadata["probe_distribution"]
        image_source = metadata.get("probe_image_source", "")
        for row in read_rows(root / "probe_summary.csv"):
            rows.append(
                {
                    "distribution": distribution,
                    "image_source": image_source if distribution != "uniform" else "",
                    "method": row["method"],
                    "alpha": float(row["alpha"]),
                    "probe_budget_fraction": float(row["probe_budget_fraction"]),
                    "mean_carrier_recall": float(row["mean_carrier_recall"]),
                    "mean_precision_hit_rate": float(row["mean_precision_hit_rate"]),
                    "lift": float(row["mean_carrier_recall"]) / max(float(row["probe_budget_fraction"]), 1e-12),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Comparison CSVs
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_comparison_ranking(summary_rows: list[dict], pcts: list[int]) -> list[dict]:
    anchors: dict[tuple[str, float], dict[str, float]] = {}
    for row in summary_rows:
        key = (row["method"], float(row["alpha"]))
        if row["localizer"] in ("uniform", "oracle"):
            anchors.setdefault(key, {})[row["localizer"]] = float(row["mean_average_precision"])

    out: list[dict] = []
    for row in summary_rows:
        key = (row["method"], float(row["alpha"]))
        floor_ap = anchors[key]["uniform"]
        oracle_ap = anchors[key]["oracle"]
        span = oracle_ap - floor_ap
        mean_ap = float(row["mean_average_precision"])
        rec: dict = {
            "method": row["method"],
            "alpha": float(row["alpha"]),
            "localizer": row["localizer"],
            "n_images": int(row["n_images"]),
            "mean_average_precision": mean_ap,
            "ap_ci_lo": float(row["ap_ci_lo"]),
            "ap_ci_hi": float(row["ap_ci_hi"]),
            "mean_roc_auc": float(row["mean_roc_auc"]),
            "floor_ap": floor_ap,
            "oracle_ap": oracle_ap,
            "ap_efficiency": (mean_ap - floor_ap) / span if span > 0 else float("nan"),
        }
        for pct in pcts:
            for col in (f"mean_recall_at_{pct}", f"recall_at_{pct}_ci_lo", f"recall_at_{pct}_ci_hi", f"mean_lift_at_{pct}"):
                rec[col] = float(row[col])
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Metric figures
# ---------------------------------------------------------------------------


def summary_lookup(summary_rows: list[dict]) -> dict[tuple[str, float, str], dict]:
    return {(r["method"], float(r["alpha"]), r["localizer"]): r for r in summary_rows}


def series_over_budgets(row: dict, pcts: list[int], prefix: str) -> np.ndarray:
    return np.array([float(row[f"{prefix}_{pct}"]) for pct in pcts], dtype=np.float64)


def apply_budget_axis(ax, pcts: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(pcts)
    ax.set_xticklabels([str(p) for p in pcts])
    ax.minorticks_off()
    ax.set_xlabel("probing budget B (% of pixels)")


def plot_recall_vs_budget(lookup, pcts, method, alpha, out_path, dpi):
    fig, ax = plt.subplots()
    for localizer in ["uniform"] + [f"{d}_stego" for d in STEGO_MAPS] + ["oracle"]:
        row = lookup[(method, alpha, localizer)]
        recalls = series_over_budgets(row, pcts, "mean_recall_at")
        lo = np.array([float(row[f"recall_at_{p}_ci_lo"]) for p in pcts])
        hi = np.array([float(row[f"recall_at_{p}_ci_hi"]) for p in pcts])
        yerr = np.vstack([np.maximum(recalls - lo, 0.0), np.maximum(hi - recalls, 0.0)])
        ax.errorbar(pcts, recalls, yerr=yerr, capsize=2, label=pretty_name(localizer), **style_for(localizer))
    apply_budget_axis(ax, pcts)
    ax.set_ylabel("carrier recall")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"{method_label(method)}, α={alpha:g} — recall@budget (n per config, 95% CI)")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_lift_vs_budget(lookup, pcts, method, alpha, out_path, dpi):
    fig, ax = plt.subplots()
    for localizer in [f"{d}_stego" for d in STEGO_MAPS] + ["oracle"]:
        row = lookup[(method, alpha, localizer)]
        lifts = series_over_budgets(row, pcts, "mean_lift_at")
        ax.plot(pcts, lifts, label=pretty_name(localizer), **style_for(localizer))
    ax.axhline(1.0, color="0.6", linestyle=":", linewidth=1.2, label="random (lift = 1)")
    apply_budget_axis(ax, pcts)
    ax.set_yscale("log")
    ax.set_ylabel("lift = recall / (B/N)")
    ax.set_title(f"{method_label(method)}, α={alpha:g} — lift@budget")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_ap_vs_alpha(lookup, alphas, method, out_path, dpi):
    fig, ax = plt.subplots()
    for localizer in ["uniform"] + [f"{d}_stego" for d in STEGO_MAPS] + ["oracle"]:
        aps, lo, hi = [], [], []
        for alpha in alphas:
            row = lookup[(method, alpha, localizer)]
            aps.append(float(row["mean_average_precision"]))
            lo.append(float(row["ap_ci_lo"]))
            hi.append(float(row["ap_ci_hi"]))
        aps, lo, hi = np.array(aps), np.array(lo), np.array(hi)
        yerr = np.vstack([np.maximum(aps - lo, 0.0), np.maximum(hi - aps, 0.0)])
        ax.errorbar(alphas, aps, yerr=yerr, capsize=2, label=pretty_name(localizer), **style_for(localizer))
    ax.set_xlabel("payload α (bpp)")
    ax.set_ylabel("Average Precision")
    ax.set_title(f"{method_label(method)} — AP vs payload (95% CI)")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def efficiency_by_config(per_image_rows: list[dict], seed: int, bootstrap: int):
    """Per-image AP efficiency (AP - prevalence) / (oracle AP - prevalence), aggregated with CIs.

    Uses the analytic floor AP = prevalence per image (exact for a random ranker), and joins the
    oracle AP row of the same (config, record).
    """
    oracle_ap: dict[tuple[str, str], float] = {}
    prevalence: dict[tuple[str, str], float] = {}
    for row in per_image_rows:
        key = (row["config_id"], row["record_name"])
        if row["localizer"] == "oracle":
            oracle_ap[key] = float(row["average_precision"])
            prevalence[key] = float(row["prevalence"])

    values: dict[tuple[str, float, str], list[float]] = {}
    for row in per_image_rows:
        localizer = row["localizer"]
        if localizer in ("uniform", "oracle") or localizer.endswith("_cover"):
            continue
        key = (row["config_id"], row["record_name"])
        span = oracle_ap[key] - prevalence[key]
        if span <= 0:
            continue
        eff = (float(row["average_precision"]) - prevalence[key]) / span
        values.setdefault((row["method"], float(row["alpha"]), localizer), []).append(eff)

    out: dict[tuple[str, float, str], tuple[float, float, float]] = {}
    for key, vals in values.items():
        out[key] = M.bootstrap_ci(np.array(vals), n_boot=bootstrap, seed=seed)
    return out


def plot_ap_efficiency_vs_alpha(efficiency, alphas, method, out_path, dpi):
    fig, ax = plt.subplots()
    for dist in STEGO_MAPS:
        localizer = f"{dist}_stego"
        means, lo, hi = [], [], []
        for alpha in alphas:
            mean, ci_lo, ci_hi = efficiency[(method, alpha, localizer)]
            means.append(mean)
            lo.append(ci_lo)
            hi.append(ci_hi)
        means, lo, hi = np.array(means), np.array(lo), np.array(hi)
        yerr = np.vstack([np.maximum(means - lo, 0.0), np.maximum(hi - means, 0.0)])
        ax.errorbar(alphas, means, yerr=yerr, capsize=2, label=pretty_name(localizer), **style_for(localizer))
    ax.axhline(1.0, color="0.0", linestyle="-", linewidth=1.0)
    ax.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_xlabel("payload α (bpp)")
    ax.set_ylabel("AP efficiency  (AP − floor) / (oracle − floor)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{method_label(method)} — fraction of the oracle AP captured (95% CI)")
    ax.legend(loc="best", fontsize=7.5)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_cover_vs_stego(lookup, alphas, method, out_path, dpi):
    fig, axes = plt.subplots(1, len(STEGO_MAPS), figsize=(2.3 * len(STEGO_MAPS), 2.7), sharey=True)
    for ax, dist in zip(axes, STEGO_MAPS):
        for source, label_tag in (("stego", "stego"), ("cover", "cover")):
            localizer = f"{dist}_{source}"
            aps = [float(lookup[(method, alpha, localizer)]["mean_average_precision"]) for alpha in alphas]
            ax.plot(alphas, aps, label=label_tag, **style_for(localizer))
        oracle_aps = [float(lookup[(method, alpha, "oracle")]["mean_average_precision"]) for alpha in alphas]
        ax.plot(alphas, oracle_aps, color="0.75", linestyle=":", linewidth=1.2, label="oracle")
        ax.set_title(pretty_name(dist), fontsize=8.5)
        ax.set_xlabel("α", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Average Precision", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"{method_label(method)} — cover-vs-stego ablation (matching curves = no embedding-specific signal)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_sampling_recall(probing_rows, method, alpha, out_path, dpi):
    fig, ax = plt.subplots()
    order = ["uniform"] + STEGO_MAPS
    for dist in order:
        rows = sorted(
            (
                r
                for r in probing_rows
                if r["distribution"] == dist and r["method"] == method and abs(r["alpha"] - alpha) < 1e-9
            ),
            key=lambda r: r["probe_budget_fraction"],
        )
        if not rows:
            continue
        pcts = [100.0 * r["probe_budget_fraction"] for r in rows]
        recalls = [r["mean_carrier_recall"] for r in rows]
        ax.plot(pcts, recalls, label=pretty_name(dist), **style_for(dist))
    ref_pcts = [1, 5, 10, 20, 30, 40, 50]
    ax.plot(ref_pcts, [p / 100.0 for p in ref_pcts], color="0.75", linestyle=":", linewidth=1.2, label="E[recall] = B/N")
    apply_budget_axis(ax, ref_pcts)
    ax.set_ylabel("mean carrier recall")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"{method_label(method)}, α={alpha:g} — γ=1 stochastic sampling (probing runs)")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Score-map panels
# ---------------------------------------------------------------------------


def normalise_for_display(values: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Percentile clip [p1, p99] then linear map to [0, 1] (independently per panel)."""
    values = values.astype(np.float64, copy=False)
    lo, hi = np.percentile(values, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def pick_panel_stems(per_image_rows: list[dict], method: str, alpha: float) -> list[str]:
    """Deterministic pick: stems at the 10th/50th/90th prevalence percentiles of a config."""
    cfg = config_id(method, alpha)
    rows = sorted(
        (r for r in per_image_rows if r["config_id"] == cfg and r["localizer"] == "oracle"),
        key=lambda r: (float(r["prevalence"]), r["record_name"]),
    )
    if not rows:
        raise ValueError(f"No oracle rows for config {cfg}; cannot pick panel images.")
    picks = [rows[int(round(q * (len(rows) - 1)))] for q in (0.10, 0.50, 0.90)]
    return [p["record_name"].replace("_changes.npz", "") for p in picks]


def render_panel(steganogram_root, cover_root, method, alpha, stem, out_path, dpi):
    cdir = steganogram_root / config_id(method, alpha)
    record = load_change_record(cdir / f"{stem}_changes.npz")
    cover = to_u8_array(cover_root / record["cover_relative_path"])
    stego = to_u8_array(cdir / record["stego_relative_path"])
    mask = reconstruct_change_mask(record["image_shape"], record["changed_flat_indices"])

    panels = [
        ("cover", cover / 255.0),
        ("true carrier mask", mask.astype(np.float64)),
        ("oracle p(change)", normalise_for_display(change_probability_map(cover, method, alpha))),
        ("texture energy", normalise_for_display(build_score_map("texture_energy", stego))),
        ("local variance", normalise_for_display(build_score_map("local_variance", stego, SCORE_PARAMS.get("local_variance")))),
        ("wavelet energy", normalise_for_display(build_score_map("wavelet_energy", stego))),
        ("SRM residual", normalise_for_display(build_score_map("srm_residual", stego))),
        ("Laplacian residual", normalise_for_display(build_score_map("laplacian_residual", stego))),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.6))
    for ax, (title, values) in zip(axes.ravel(), panels):
        ax.imshow(values, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()
    prevalence = 100.0 * record["used_pixel_fraction"]
    fig.suptitle(
        f"{method_label(method)}, α={alpha:g}, {stem} — score maps on the stego image "
        f"(carriers: {prevalence:.2f}% of pixels; maps clipped to [p1, p99])",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark-root", type=Path, default=Path("experiments/distribution_benchmark"))
    p.add_argument("--probing-roots", type=Path, nargs="+", default=DEFAULT_PROBING_ROOTS)
    p.add_argument("--cover-root", type=Path, default=Path("ALASKA_v2_TIFF_512_GrayScale_50"))
    p.add_argument("--steganogram-root", type=Path, default=Path("experiments/probing_only/steganograms"))
    p.add_argument("--output-root", type=Path, default=Path("experiments/distribution_comparison"))
    p.add_argument("--figure-alphas", type=float, nargs="+", default=[0.05, 0.2])
    p.add_argument("--panel-alpha", type=float, default=0.2)
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for efficiency CIs.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--dpi", type=int, default=300)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "figure.figsize": (5.0, 3.4),
        }
    )

    summary_rows = read_rows(args.benchmark_root / "summary.csv")
    per_image_rows = read_rows(args.benchmark_root / "per_image.csv")
    benchmark_meta = json.loads((args.benchmark_root / "metadata.json").read_text(encoding="utf-8"))
    pcts = budget_pcts(summary_rows)
    methods = sorted({r["method"] for r in summary_rows})
    alphas = sorted({float(r["alpha"]) for r in summary_rows})
    lookup = summary_lookup(summary_rows)

    figures_dir = args.output_root / "figures"
    panels_dir = args.output_root / "panels"
    figures_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    # --- CSVs ---
    ranking_rows = build_comparison_ranking(summary_rows, pcts)
    write_csv(args.output_root / "comparison_ranking.csv", ranking_rows)
    probing_rows = load_probing(args.probing_roots)
    write_csv(args.output_root / "comparison_sampling.csv", probing_rows)
    print(f"[csv] comparison_ranking.csv ({len(ranking_rows)} rows), comparison_sampling.csv ({len(probing_rows)} rows)")

    # --- metric figures ---
    written: list[str] = []
    for method in methods:
        mtag = method.lower()
        for alpha in args.figure_alphas:
            atag = alpha_tag(alpha)
            path = figures_dir / f"fig_recall_vs_budget_{mtag}_alpha{atag}.png"
            plot_recall_vs_budget(lookup, pcts, method, alpha, path, args.dpi)
            written.append(path.name)
            path = figures_dir / f"fig_lift_vs_budget_{mtag}_alpha{atag}.png"
            plot_lift_vs_budget(lookup, pcts, method, alpha, path, args.dpi)
            written.append(path.name)
        path = figures_dir / f"fig_ap_vs_alpha_{mtag}.png"
        plot_ap_vs_alpha(lookup, alphas, method, path, args.dpi)
        written.append(path.name)
        path = figures_dir / f"fig_cover_vs_stego_{mtag}.png"
        plot_cover_vs_stego(lookup, alphas, method, path, args.dpi)
        written.append(path.name)
        path = figures_dir / f"fig_sampling_recall_vs_budget_{mtag}_alpha{alpha_tag(args.panel_alpha)}.png"
        plot_sampling_recall(probing_rows, method, args.panel_alpha, path, args.dpi)
        written.append(path.name)

    efficiency = efficiency_by_config(per_image_rows, seed=args.seed, bootstrap=args.bootstrap)
    for method in methods:
        path = figures_dir / f"fig_ap_efficiency_vs_alpha_{method.lower()}.png"
        plot_ap_efficiency_vs_alpha(efficiency, alphas, method, path, args.dpi)
        written.append(path.name)
    print(f"[figures] {len(written)} PNGs -> {figures_dir}")

    # --- score-map panels ---
    panel_files: list[str] = []
    stems = pick_panel_stems(per_image_rows, methods[0], args.panel_alpha)
    for method in methods:
        for stem in stems:
            path = panels_dir / f"panel_{method.lower()}_alpha{alpha_tag(args.panel_alpha)}_{stem}.png"
            render_panel(args.steganogram_root, args.cover_root, method, args.panel_alpha, stem, path, args.dpi)
            panel_files.append(path.name)
    print(f"[panels] {len(panel_files)} PNGs -> {panels_dir}")

    (args.output_root / "metadata.json").write_text(
        json.dumps(
            {
                "benchmark_root": str(args.benchmark_root.resolve()),
                "benchmark_metadata": benchmark_meta,
                "probing_roots": [str(p.resolve()) for p in args.probing_roots],
                "cover_root": str(args.cover_root.resolve()),
                "steganogram_root": str(args.steganogram_root.resolve()),
                "figure_alphas": args.figure_alphas,
                "panel_alpha": args.panel_alpha,
                "panel_stems": stems,
                "bootstrap_resamples": args.bootstrap,
                "seed": args.seed,
                "dpi": args.dpi,
                "matplotlib_version": matplotlib.__version__,
                "figures": written,
                "panels": panel_files,
                "note": "All figures are grayscale-only (gray level x line style x marker); "
                "panels are independently percentile-normalised to [p1, p99].",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote comparison outputs to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
