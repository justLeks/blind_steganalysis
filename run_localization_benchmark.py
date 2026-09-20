"""Distribution benchmark: bracket every probing score map between floor and ceiling.

Localizers compared, per (method, alpha), over the recorded carrier masks:

- ``uniform``            -- random floor. AP = prevalence, recall@B = B/N, lift = 1 (analytic).
- ``<dist>_<stego|cover>`` -- one localizer per score map in ``read_changes.SCORE_MAP_BUILDERS``
  (texture_energy, gradient_magnitude, laplacian_residual, local_variance, local_entropy,
  wavelet_energy, srm_residual), scored on the
  stego image (the blind setting) and on the cover image (the cover-vs-stego ablation: if the two
  match, the map carries no embedding-specific signal).
- ``oracle``             -- ceiling: true conseal selection channel (cover-derived change probability).

Legacy names ``texture_stego`` / ``texture_cover`` are accepted as aliases and normalised to
``texture_energy_stego`` / ``texture_energy_cover``.

Metrics per image: Average Precision (primary), ROC-AUC (secondary), and deterministic top-B
recall/lift at each budget fraction. Aggregated per (method, alpha, localizer) with percentile
bootstrap CIs across images.

Reproducibility: the recall/lift tie-break RNG is seeded per record (SHA-256 of config id + record
name mixed with ``--seed``), so per-image rows are independent of the localizer list, ``--limit``,
and ``--workers`` — any worker count reproduces bit-identical CSVs. (Before 2026-07-16 one RNG was
shared sequentially across rows; the frozen runs of record in ``experiments/localization_benchmark``
and ``experiments/distribution_benchmark`` carry that older tie-break stream — their recall columns
are not regenerable bit-exact, AP/ROC-AUC are RNG-free and unaffected.)

Reuses already-embedded steganograms (no re-embedding). Config is taken from CLI args and persisted to
``metadata.json`` next to the outputs -- the reproducible pattern that replaces config-as-globals.

Run:  python3 run_localization_benchmark.py            # full sweep, all images (~1 h: oracle dominates)
      python3 run_localization_benchmark.py --limit 20 # quick pass
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from embedding import to_u8_array
from read_changes import (
    IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS,
    build_adaptive_probabilities,
    iter_record_paths,
    load_change_record,
    resolve_score_map_params,
)
from run_probing_experiment import config_id
from selection_channel import oracle_scores
import localization_metrics as M


DEFAULTS = dict(
    steganogram_root=Path("experiments/probing_only/steganograms"),
    cover_root=Path("ALASKA_v2_TIFF_512_GrayScale_50"),
    output_root=Path("experiments/distribution_benchmark"),
    methods=["HUGO", "MIPOD"],
    alphas=[0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    budgets=[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    localizers=[
        "uniform",
        "texture_energy_stego",
        "texture_energy_cover",
        "gradient_magnitude_stego",
        "gradient_magnitude_cover",
        "laplacian_residual_stego",
        "laplacian_residual_cover",
        "local_variance_stego",
        "local_variance_cover",
        "local_entropy_stego",
        "local_entropy_cover",
        "wavelet_energy_stego",
        "wavelet_energy_cover",
        "srm_residual_stego",
        "srm_residual_cover",
        "oracle",
    ],
)
# Overrides on top of the registry defaults; resolve_score_map_params fills the rest.
SCORE_PARAMS: dict[str, dict] = {"local_variance": {"window": 9}}
LOCALIZER_ALIASES = {
    "texture_stego": "texture_energy_stego",
    "texture_cover": "texture_energy_cover",
}


def parse_localizer(name: str) -> tuple[str, str]:
    """Split ``<dist>_<stego|cover>`` into (distribution, source); reject anything else.

    ``uniform`` and ``oracle`` are not parseable map localizers and raise, as does any
    unknown distribution or source -- callers validate the full list up front with this.
    """
    dist, sep, source = name.rpartition("_")
    if sep and source in ("stego", "cover") and dist in IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS:
        return dist, source
    raise ValueError(
        f"Unknown localizer '{name}'. Expected 'uniform', 'oracle', or '<distribution>_<stego|cover>' "
        f"with distribution in {sorted(IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS)}."
    )


def per_image_scores(localizer, cover, stego, method, alpha):
    """Per-pixel score vector for a localizer. ``uniform`` is handled analytically by the caller.

    Adaptive localizers score by the sampling-probability map (the monotone
    (energy + floor)**gamma normalisation) -- the same code path the probing runners use, which
    keeps the texture_energy rows bit-identical to the Milestone-1 run. AP/recall only depend on
    the score ordering, so this is equivalent to ranking by the raw energy map.
    """
    if localizer == "oracle":
        return oracle_scores(cover, method, alpha)
    dist, source = parse_localizer(localizer)
    image = stego if source == "stego" else cover
    return build_adaptive_probabilities(dist, image, SCORE_PARAMS.get(dist))


def derive_record_seed(base_seed: int, cfg: str, record_name: str) -> int:
    """Deterministic per-record tie-break seed: independent of localizer list, limit, and workers."""
    digest = hashlib.sha256(f"{cfg}|{record_name}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return (int(base_seed) + offset) % (2**32)


def _evaluate_record(task: tuple) -> list[dict]:
    """All localizer rows for one change record. Module-level (picklable) for multiprocessing.

    Deterministic: the tie-break RNG is derived from (seed, config id, record name) only, so the
    output is identical for any worker count or record order.
    """
    rp, cfg, method, alpha, localizers, budget_list, budget_pct, cover_root, cdir, base_seed = task
    budgets = np.asarray(budget_list, dtype=np.float64)
    rng = np.random.default_rng(derive_record_seed(base_seed, cfg, rp.name))

    r = load_change_record(rp)
    height, width = r["image_shape"]
    total = height * width
    labels = np.zeros(total, dtype=bool)
    labels[r["changed_flat_indices"]] = True
    prevalence = float(labels.mean())

    cover = to_u8_array(cover_root / r["cover_relative_path"])
    stego = to_u8_array(cdir / r["stego_relative_path"])

    rows: list[dict] = []
    for localizer in localizers:
        if localizer == "uniform":
            # Analytic expectation of a random ranker (exact, no Monte-Carlo noise).
            ap = prevalence
            roc = 0.5
            recalls = budgets.copy()
        else:
            scores = per_image_scores(localizer, cover, stego, method, alpha)
            ap = M.average_precision(scores, labels)
            roc = M.roc_auc(scores, labels)
            recalls = M.recall_at_budgets(scores, labels, budgets, rng)
        lifts = M.lift_at_budgets(recalls, budgets)

        row = {
            "config_id": cfg,
            "method": method.upper(),
            "alpha": float(alpha),
            "record_name": rp.name,
            "localizer": localizer,
            "carrier_pixels": int(labels.sum()),
            "prevalence": prevalence,
            "average_precision": ap,
            "roc_auc": roc,
        }
        for pct, rec, lift in zip(budget_pct, recalls, lifts):
            row[f"recall_at_{pct}"] = float(rec)
            row[f"lift_at_{pct}"] = float(lift)
        rows.append(row)
    return rows


def evaluate(args) -> None:
    args.localizers = [str(LOCALIZER_ALIASES.get(name, name)) for name in args.localizers]
    for name in args.localizers:
        if name not in ("uniform", "oracle"):
            parse_localizer(name)  # fail fast on typos before the long sweep
    if args.workers < 1:
        raise ValueError(f"workers must be >= 1, got {args.workers}.")

    budget_pct = [f"{int(round(100 * b))}" for b in args.budgets]

    tasks: list[tuple] = []
    for method in args.methods:
        for alpha in args.alphas:
            cfg = config_id(method, alpha)
            cdir = (args.steganogram_root / cfg)
            if not cdir.exists():
                print(f"[skip] {cfg}: no steganograms at {cdir}")
                continue
            records = iter_record_paths(cdir)
            if args.limit:
                records = records[: args.limit]
            print(f"[run] {cfg}: {len(records)} images")
            tasks.extend(
                (rp, cfg, method, alpha, args.localizers, args.budgets, budget_pct,
                 args.cover_root, cdir, args.seed)
                for rp in records
            )

    if args.workers == 1:
        per_record_rows = [_evaluate_record(task) for task in tasks]
    else:
        # "spawn" gives identical worker behaviour on macOS/Linux; each worker pays a one-time
        # conseal import + JIT warmup. imap preserves input order, so the CSVs are byte-identical
        # to the sequential path (per-record tie-break seeds make rows order-independent anyway).
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=args.workers) as pool:
            per_record_rows = list(pool.imap(_evaluate_record, tasks, chunksize=4))

    raw_rows: list[dict] = [row for rows in per_record_rows for row in rows]
    summary_rows = summarize(raw_rows, budget_pct, args.bootstrap, args.seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "per_image.csv", raw_rows)
    write_csv(args.output_root / "summary.csv", summary_rows)
    (args.output_root / "metadata.json").write_text(
        json.dumps(
            {
                "steganogram_root": str(args.steganogram_root.resolve()),
                "cover_root": str(args.cover_root.resolve()),
                "methods": args.methods,
                "alphas": args.alphas,
                "budgets": args.budgets,
                "localizers": args.localizers,
                "score_map_params": {
                    dist: resolve_score_map_params(dist, SCORE_PARAMS.get(dist))
                    for dist in sorted(
                        {
                            parse_localizer(name)[0]
                            for name in args.localizers
                            if name not in ("uniform", "oracle")
                        }
                    )
                },
                "images_per_config": args.limit or "all",
                "bootstrap_resamples": args.bootstrap,
                "seed": args.seed,
                "workers": args.workers,
                "tie_break_rng": "per-record (sha256 of config_id|record_name mixed with seed)",
                "n_raw_rows": len(raw_rows),
                "n_summary_rows": len(summary_rows),
                "note": "uniform localizer reported as analytic expectation (AP=prevalence, recall=fraction).",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {len(raw_rows)} per-image rows and {len(summary_rows)} summary rows to {args.output_root.resolve()}")


def summarize(raw_rows, budget_pct, bootstrap, seed):
    groups: dict[tuple, list[dict]] = {}
    for row in raw_rows:
        groups.setdefault((row["method"], row["alpha"], row["localizer"]), []).append(row)

    out: list[dict] = []
    for (method, alpha, localizer), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        def col(name):
            return np.array([row[name] for row in rows], dtype=np.float64)

        ap_mean, ap_lo, ap_hi = M.bootstrap_ci(col("average_precision"), bootstrap, seed=seed)
        rec = {
            "config_id": config_id(method, alpha),
            "method": method,
            "alpha": float(alpha),
            "localizer": localizer,
            "n_images": len(rows),
            "mean_prevalence": float(np.nanmean(col("prevalence"))),
            "mean_average_precision": ap_mean,
            "ap_ci_lo": ap_lo,
            "ap_ci_hi": ap_hi,
            "mean_roc_auc": float(np.nanmean(col("roc_auc"))),
        }
        for pct in budget_pct:
            r_mean, r_lo, r_hi = M.bootstrap_ci(col(f"recall_at_{pct}"), bootstrap, seed=seed)
            rec[f"mean_recall_at_{pct}"] = r_mean
            rec[f"recall_at_{pct}_ci_lo"] = r_lo
            rec[f"recall_at_{pct}_ci_hi"] = r_hi
            rec[f"mean_lift_at_{pct}"] = float(np.nanmean(col(f"lift_at_{pct}")))
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steganogram-root", type=Path, default=DEFAULTS["steganogram_root"])
    p.add_argument("--cover-root", type=Path, default=DEFAULTS["cover_root"])
    p.add_argument("--output-root", type=Path, default=DEFAULTS["output_root"])
    p.add_argument("--methods", nargs="+", default=DEFAULTS["methods"])
    p.add_argument("--alphas", nargs="+", type=float, default=DEFAULTS["alphas"])
    p.add_argument("--budgets", nargs="+", type=float, default=DEFAULTS["budgets"])
    p.add_argument("--localizers", nargs="+", default=DEFAULTS["localizers"])
    p.add_argument("--limit", type=int, default=0, help="Max images per config (0 = all).")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for CIs.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default 1 = sequential). Tie-break seeds are per-record, "
        "so results are identical for any worker count.",
    )
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
