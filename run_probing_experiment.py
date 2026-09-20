from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import secrets
from pathlib import Path

import numpy as np

from embedding import embed_and_log, read_cover_list, to_u8_array
from read_changes import (
    IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS,
    SUPPORTED_PROBE_DISTRIBUTIONS,
    build_sampling_probabilities,
    filter_record_paths_by_cover_list,
    iter_record_paths,
    load_change_record,
    resolve_probe_image_path,
    resolve_score_map_params,
)


# These module-level constants are the DEFAULTS. Override any of them on the command line
# (see build_parser). Committed defaults reproduce the published experiment in
# experiments/probing_only (texture_energy, fixed seed 12345).
COVER_ROOT = Path("ALASKA_v2_TIFF_512_GrayScale_50")
# Optional cover-list file (one file name per line) restricting the run to a subset of COVER_ROOT.
COVER_LIST: Path | None = None
# Parallel worker processes for embedding and probing (spawn context; results are order-independent).
WORKERS = 1
# Repeated weighted draws per image for the stochastic (gamma-weighted) sampling view. 1 = the
# published single-draw schema; K > 1 stores per-image mean/sd/min/max over the K draws.
REPEATS = 1
EXPERIMENT_ROOT = Path("experiments/probing_only")
STEGANOGRAM_ROOT = Path("experiments/probing_only/steganograms")

METHODS = ["HUGO", "MIPOD"]
ALPHAS = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
EMBED_SEED = 12345

PROBE_BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
PROBE_DISTRIBUTION = "texture_energy"
PROBE_DISTRIBUTION_PARAMS = {"gamma": 1.0, "floor": 1e-6}
PROBE_IMAGE_SOURCE = "stego"
# Fixed for reproducibility (matches experiments/probing_only). Pass --probe-seed -1 (or set this to
# None) for a fresh random seed each run, saved to metadata.json.
PROBE_SEED: int | None = 12345

# Max records processed per config (0 = all). Set via --limit.
LIMIT = 0

RAW_CSV = EXPERIMENT_ROOT / "probe_raw.csv"
SUMMARY_CSV = EXPERIMENT_ROOT / "probe_summary.csv"
METADATA_JSON = EXPERIMENT_ROOT / "metadata.json"


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def config_id(method: str, alpha: float) -> str:
    return f"{method.lower()}_alpha{alpha_tag(alpha)}"


def resolve_run_probe_seed() -> tuple[int, str]:
    if PROBE_SEED is None:
        return secrets.randbits(32), "generated_random"
    return int(PROBE_SEED), "configured_fixed"


def derive_seed(base_seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return (int(base_seed) + offset) % (2**32)


def resolve_cover_list() -> set[str] | None:
    return read_cover_list(COVER_LIST) if COVER_LIST else None


def cover_image_count() -> int:
    cover_list = resolve_cover_list()
    if cover_list is not None:
        return len(cover_list)
    return sum(1 for path in COVER_ROOT.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"})


def manifest_is_complete(output_dir: Path, method: str, alpha: float, expected_count: int) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if len(manifest) != expected_count:
        return False
    return all(
        item.get("method") == method.upper() and abs(float(item.get("alpha", -1)) - alpha) < 1e-12
        for item in manifest
    )


def ensure_steganograms(method: str, alpha: float, expected_count: int) -> Path:
    output_dir = (STEGANOGRAM_ROOT / config_id(method, alpha)).expanduser().resolve()
    if manifest_is_complete(output_dir, method, alpha, expected_count):
        print(f"[embed:skip] {config_id(method, alpha)}")
        return output_dir

    print(f"[embed] {config_id(method, alpha)}")
    embed_and_log(
        src_dir=COVER_ROOT,
        dst_dir=output_dir,
        method=method,
        alpha=alpha,
        seed=EMBED_SEED,
        workers=WORKERS,
        cover_list=resolve_cover_list(),
    )
    return output_dir


def budget_pixels(total_pixels: int, fraction: float) -> int:
    return min(total_pixels, max(1, int(round(total_pixels * float(fraction)))))


def sample_unique_probe_order(total_pixels: int, max_budget: int, seed: int, probabilities: np.ndarray | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if probabilities is None:
        return rng.permutation(total_pixels)[:max_budget].astype(np.int64, copy=False)

    weights = probabilities.astype(np.float64, copy=False)
    if weights.size != total_pixels:
        raise ValueError("Probability vector length does not match the image pixel count.")
    if np.any(weights < 0):
        raise ValueError("Sampling probabilities must be non-negative.")
    if float(weights.sum()) <= 0:
        raise ValueError("Sampling probabilities must have positive total weight.")

    # Exponential-race weighted sampling without replacement. Smaller keys are sampled first.
    uniform = np.clip(rng.random(total_pixels), np.finfo(np.float64).tiny, 1.0)
    keys = -np.log(uniform) / weights
    keys[weights <= 0] = np.inf
    selected = np.argpartition(keys, max_budget - 1)[:max_budget]
    return selected[np.argsort(keys[selected])].astype(np.int64, copy=False)


def probe_record(
    record: dict, input_base: Path, max_budget: int, run_probe_seed: int, *,
    distribution: str, distribution_params: dict, image_source: str, cover_root: Path,
    budget_fractions: list[float], repeats: int = 1,
) -> list[dict]:
    """Probe one change record; explicit configuration so spawned workers never read module globals.

    ``repeats`` == 1 reproduces the published row schema and values bit-exactly. For ``repeats`` > 1
    the per-budget hit counts are averaged over K independent draws (draw 0 is the K == 1 draw) and
    the row gains ``repeats``, ``sd_/min_/max_carrier_recall`` and ``sd_guessed_carrier_pixels``.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")
    image_shape = tuple(record["image_shape"])
    total_pixels = image_shape[0] * image_shape[1]
    changed_lookup = np.zeros(total_pixels, dtype=bool)
    changed_lookup[record["changed_flat_indices"]] = True

    sampling_image = None
    if distribution in IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS:
        sampling_image = to_u8_array(resolve_probe_image_path(
            record=record, input_base=input_base, image_source=image_source, cover_root=cover_root))
    probabilities = build_sampling_probabilities(
        image_shape=image_shape, distribution=distribution,
        distribution_params=distribution_params, sampling_image=sampling_image)

    base_parts = (record["method"], record["alpha"], record["record_path"].name, distribution)
    seeds = [derive_seed(run_probe_seed, *base_parts)] + [
        derive_seed(run_probe_seed, *base_parts, f"repeat{k}") for k in range(1, repeats)]
    budgets = [budget_pixels(total_pixels, f) for f in budget_fractions]
    hits = np.zeros((repeats, len(budgets)), dtype=np.int64)
    unique_first = np.zeros(len(budgets), dtype=np.int64)
    for k, seed in enumerate(seeds):
        order = sample_unique_probe_order(total_pixels, max_budget, seed, probabilities)
        for j, b in enumerate(budgets):
            positions = order[:b]
            hits[k, j] = int(np.count_nonzero(changed_lookup[positions]))
            if k == 0:
                unique_first[j] = int(np.unique(positions).size)

    carrier_pixels = int(record["used_pixels"])
    rows: list[dict] = []
    previous_budget = 0
    previous_hits = np.zeros(repeats, dtype=np.float64)
    for j, (fraction, b) in enumerate(zip(budget_fractions, budgets)):
        h = hits[:, j].astype(np.float64)
        inc_b = b - previous_budget
        inc_h = h - previous_hits
        recall = h / carrier_pixels if carrier_pixels else np.zeros_like(h)
        row = {
            "config_id": config_id(record["method"], record["alpha"]),
            "record_name": record["record_path"].name,
            "record_path": str(record["record_path"]),
            "cover_relative_path": record["cover_relative_path"],
            "stego_relative_path": record["stego_relative_path"],
            "method": record["method"],
            "alpha": float(record["alpha"]),
            "record_seed": int(record["seed"]),
            "probe_seed": int(seeds[0]),
            "probe_distribution": distribution,
            "probe_image_source": image_source if distribution in IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS else "",
            "image_height": int(image_shape[0]),
            "image_width": int(image_shape[1]),
            "total_pixels": int(total_pixels),
            "carrier_pixels": carrier_pixels,
            "used_pixels": carrier_pixels,
            "used_pixel_fraction": float(record["used_pixel_fraction"]),
            "used_pixel_percentage": float(record["used_pixel_percentage"]),
            "probe_budget_fraction": float(fraction),
            "probe_budget_percentage": float(100.0 * fraction),
            "probe_budget_pixels": int(b),
            "unique_probed_pixels": int(unique_first[j]),
            "guessed_carrier_pixels": int(h[0]) if repeats == 1 else float(h.mean()),
            "precision_hit_rate": float(h.mean() / b) if b else 0.0,
            "carrier_recall": float(recall.mean()),
            "incremental_probe_pixels": int(inc_b),
            "incremental_guessed_pixels": int(inc_h[0]) if repeats == 1 else float(inc_h.mean()),
            "incremental_precision_hit_rate": float(inc_h.mean() / inc_b) if inc_b else 0.0,
        }
        if repeats > 1:
            row.update({
                "repeats": repeats,
                "sd_carrier_recall": float(recall.std(ddof=1)),
                "min_carrier_recall": float(recall.min()),
                "max_carrier_recall": float(recall.max()),
                "sd_guessed_carrier_pixels": float(h.std(ddof=1)),
            })
        rows.append(row)
        previous_budget, previous_hits = b, h
    return rows


def _probe_task(task: tuple) -> list[dict]:
    """(record_path, input_base, max_budget, run_probe_seed, cfg) -> rows. Picklable for spawn workers."""
    record_path, input_base, max_budget, run_probe_seed, cfg = task
    record = load_change_record(record_path)
    return probe_record(record, input_base, max_budget, run_probe_seed, **cfg)


def run_tasks(tasks: list[tuple], workers: int) -> list[list[dict]]:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")
    if workers == 1:
        return [_probe_task(t) for t in tasks]
    # "spawn" gives identical worker behaviour on macOS/Linux; imap preserves input order and the
    # per-record seeds make every row independent of worker count.
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        return list(pool.imap(_probe_task, tasks, chunksize=4))

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for row in rows:
        key = (str(row["method"]), float(row["alpha"]), float(row["probe_budget_fraction"]))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (method, alpha, fraction), group_rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        def mean(key: str) -> float:
            return float(np.mean([float(row[key]) for row in group_rows]))

        def total(key: str) -> int:
            return int(sum(int(row[key]) for row in group_rows))

        summary_row = {
                "config_id": config_id(method, alpha),
                "method": method,
                "alpha": float(alpha),
                "probe_budget_fraction": float(fraction),
                "probe_budget_percentage": float(100.0 * fraction),
                "records": int(len(group_rows)),
                "mean_total_pixels": mean("total_pixels"),
                "mean_carrier_pixels": mean("carrier_pixels"),
                "mean_used_pixels": mean("used_pixels"),
                "mean_used_pixel_fraction": mean("used_pixel_fraction"),
                "mean_used_pixel_percentage": mean("used_pixel_percentage"),
                "mean_probe_budget_pixels": mean("probe_budget_pixels"),
                "total_probe_budget_pixels": total("probe_budget_pixels"),
                "total_guessed_carrier_pixels": total("guessed_carrier_pixels"),
                "mean_guessed_carrier_pixels": mean("guessed_carrier_pixels"),
                "mean_precision_hit_rate": mean("precision_hit_rate"),
                "mean_carrier_recall": mean("carrier_recall"),
                "mean_incremental_probe_pixels": mean("incremental_probe_pixels"),
                "mean_incremental_guessed_pixels": mean("incremental_guessed_pixels"),
                "mean_incremental_precision_hit_rate": mean("incremental_precision_hit_rate"),
        }
        if "sd_carrier_recall" in group_rows[0]:
            summary_row["repeats"] = int(group_rows[0]["repeats"])
            summary_row["mean_sd_carrier_recall"] = mean("sd_carrier_recall")
        summary_rows.append(summary_row)
    return summary_rows


def run_experiment(run_probe_seed: int) -> tuple[list[dict], list[dict]]:
    expected_count = cover_image_count()
    if expected_count <= 0:
        raise FileNotFoundError(f"No cover TIFF images found in {COVER_ROOT.resolve()}")
    cover_list = resolve_cover_list()
    cfg = dict(distribution=PROBE_DISTRIBUTION, distribution_params=PROBE_DISTRIBUTION_PARAMS,
               image_source=PROBE_IMAGE_SOURCE, cover_root=COVER_ROOT,
               budget_fractions=PROBE_BUDGET_FRACTIONS, repeats=REPEATS)
    tasks: list[tuple] = []
    for method in METHODS:
        for alpha in ALPHAS:
            config_dir = ensure_steganograms(method, alpha, expected_count)
            record_paths = filter_record_paths_by_cover_list(iter_record_paths(config_dir), cover_list)
            if LIMIT:
                record_paths = record_paths[:LIMIT]
            for record_path in record_paths:
                shape = load_change_record(record_path)["image_shape"]
                max_budget = budget_pixels(shape[0] * shape[1], max(PROBE_BUDGET_FRACTIONS))
                tasks.append((record_path, config_dir, max_budget, run_probe_seed, cfg))
    raw_rows = [row for rows in run_tasks(tasks, WORKERS) for row in rows]
    return raw_rows, summarize(raw_rows)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Budgeted carrier-pixel probing sweep. Defaults reproduce experiments/probing_only."
    )
    p.add_argument("--cover-root", type=Path, default=COVER_ROOT)
    p.add_argument("--cover-list", type=Path, default=COVER_LIST,
                   help="Restrict the run to the cover file names listed in this file (one per line).")
    p.add_argument("--workers", type=int, default=WORKERS, help="Parallel worker processes (default 1).")
    p.add_argument("--repeats", type=int, default=REPEATS,
                   help="Weighted draws per image (default 1 = published schema; K > 1 stores per-image aggregates).")
    p.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    p.add_argument("--steganogram-root", type=Path, default=STEGANOGRAM_ROOT)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    p.add_argument("--budgets", nargs="+", type=float, default=PROBE_BUDGET_FRACTIONS)
    p.add_argument("--distribution", default=PROBE_DISTRIBUTION, choices=sorted(SUPPORTED_PROBE_DISTRIBUTIONS))
    p.add_argument("--image-source", default=PROBE_IMAGE_SOURCE, choices=["stego", "cover"])
    p.add_argument(
        "--probe-seed", type=int, default=PROBE_SEED,
        help="Fixed probe seed (default 12345). Pass -1 for a fresh random seed each run.",
    )
    p.add_argument("--limit", type=int, default=LIMIT, help="Max records per config (0 = all).")
    return p


def apply_args(args: argparse.Namespace) -> None:
    """Override the module-level config from parsed CLI args (keeps the importable defaults intact)."""
    global COVER_ROOT, COVER_LIST, WORKERS, REPEATS, EXPERIMENT_ROOT, STEGANOGRAM_ROOT, METHODS, ALPHAS
    global PROBE_BUDGET_FRACTIONS, PROBE_DISTRIBUTION, PROBE_IMAGE_SOURCE, PROBE_SEED, LIMIT
    global RAW_CSV, SUMMARY_CSV, METADATA_JSON
    COVER_ROOT = args.cover_root
    COVER_LIST = args.cover_list
    WORKERS = args.workers
    REPEATS = args.repeats
    EXPERIMENT_ROOT = args.experiment_root
    STEGANOGRAM_ROOT = args.steganogram_root
    METHODS = args.methods
    ALPHAS = args.alphas
    PROBE_BUDGET_FRACTIONS = args.budgets
    PROBE_DISTRIBUTION = args.distribution
    PROBE_IMAGE_SOURCE = args.image_source
    PROBE_SEED = None if (args.probe_seed is not None and args.probe_seed < 0) else args.probe_seed
    LIMIT = args.limit
    RAW_CSV = EXPERIMENT_ROOT / "probe_raw.csv"
    SUMMARY_CSV = EXPERIMENT_ROOT / "probe_summary.csv"
    METADATA_JSON = EXPERIMENT_ROOT / "metadata.json"


def main(argv: list[str] | None = None) -> None:
    apply_args(build_parser().parse_args(argv))
    run_probe_seed, probe_seed_mode = resolve_run_probe_seed()
    print(f"[probe_seed] {run_probe_seed} ({probe_seed_mode})")

    raw_rows, summary_rows = run_experiment(run_probe_seed)
    write_csv(RAW_CSV, raw_rows)
    write_csv(SUMMARY_CSV, summary_rows)
    METADATA_JSON.write_text(
        json.dumps(
            {
                "cover_root": str(COVER_ROOT.resolve()),
                "cover_list": str(COVER_LIST.resolve()) if COVER_LIST else None,
                "workers": WORKERS,
                "repeats": REPEATS,
                "experiment_root": str(EXPERIMENT_ROOT.resolve()),
                "steganogram_root": str(STEGANOGRAM_ROOT.resolve()),
                "methods": METHODS,
                "alphas": ALPHAS,
                "embed_seed": EMBED_SEED,
                "configured_probe_seed": PROBE_SEED,
                "run_probe_seed": run_probe_seed,
                "probe_seed_mode": probe_seed_mode,
                "probe_budget_fractions": PROBE_BUDGET_FRACTIONS,
                "probe_distribution": PROBE_DISTRIBUTION,
                "probe_distribution_params": PROBE_DISTRIBUTION_PARAMS,
                "resolved_distribution_params": (
                    resolve_score_map_params(PROBE_DISTRIBUTION, PROBE_DISTRIBUTION_PARAMS)
                    if PROBE_DISTRIBUTION in IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS
                    else None
                ),
                "probe_image_source": PROBE_IMAGE_SOURCE,
                "raw_csv": str(RAW_CSV.resolve()),
                "summary_csv": str(SUMMARY_CSV.resolve()),
                "raw_rows": len(raw_rows),
                "summary_rows": len(summary_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(raw_rows)} raw rows to {RAW_CSV.resolve()}")
    print(f"Wrote {len(summary_rows)} summary rows to {SUMMARY_CSV.resolve()}")


if __name__ == "__main__":
    main()
