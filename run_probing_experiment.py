from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np

from embedding import embed_and_log, to_u8_array
from read_changes import (
    IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS,
    SUPPORTED_PROBE_DISTRIBUTIONS,
    build_sampling_probabilities,
    iter_record_paths,
    load_change_record,
    resolve_probe_image_path,
)


# These module-level constants are the DEFAULTS. Override any of them on the command line
# (see build_parser); the reporting scripts import these names, so they remain the source of truth
# for the default result location. Committed defaults reproduce the published experiment in
# experiments/probing_only (texture_energy, fixed seed 12345).
COVER_ROOT = Path("ALASKA_v2_TIFF_512_GrayScale_50")
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


def cover_image_count() -> int:
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


def probe_record(record: dict, input_base: Path, max_budget: int, run_probe_seed: int) -> list[dict]:
    image_shape = tuple(record["image_shape"])
    total_pixels = image_shape[0] * image_shape[1]
    changed_lookup = np.zeros(total_pixels, dtype=bool)
    changed_lookup[record["changed_flat_indices"]] = True

    sampling_image = None
    probabilities = None
    if PROBE_DISTRIBUTION in IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS:
        sampling_image_path = resolve_probe_image_path(
            record=record,
            input_base=input_base,
            image_source=PROBE_IMAGE_SOURCE,
            cover_root=COVER_ROOT,
        )
        sampling_image = to_u8_array(sampling_image_path)

    probabilities = build_sampling_probabilities(
        image_shape=image_shape,
        distribution=PROBE_DISTRIBUTION,
        distribution_params=PROBE_DISTRIBUTION_PARAMS,
        sampling_image=sampling_image,
    )
    probe_seed = derive_seed(
        run_probe_seed,
        record["method"],
        record["alpha"],
        record["record_path"].name,
        PROBE_DISTRIBUTION,
    )
    probe_order = sample_unique_probe_order(total_pixels, max_budget, probe_seed, probabilities)

    rows: list[dict] = []
    previous_budget = 0
    previous_guessed = 0
    carrier_pixels = int(record["used_pixels"])

    for fraction in PROBE_BUDGET_FRACTIONS:
        current_budget = budget_pixels(total_pixels, fraction)
        current_positions = probe_order[:current_budget]
        guessed_carrier_pixels = int(np.count_nonzero(changed_lookup[current_positions]))

        incremental_budget = current_budget - previous_budget
        incremental_guessed = guessed_carrier_pixels - previous_guessed

        rows.append(
            {
                "config_id": config_id(record["method"], record["alpha"]),
                "record_name": record["record_path"].name,
                "record_path": str(record["record_path"]),
                "cover_relative_path": record["cover_relative_path"],
                "stego_relative_path": record["stego_relative_path"],
                "method": record["method"],
                "alpha": float(record["alpha"]),
                "record_seed": int(record["seed"]),
                "probe_seed": int(probe_seed),
                "probe_distribution": PROBE_DISTRIBUTION,
                "probe_image_source": (
                    PROBE_IMAGE_SOURCE
                    if PROBE_DISTRIBUTION in IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS
                    else ""
                ),
                "image_height": int(image_shape[0]),
                "image_width": int(image_shape[1]),
                "total_pixels": int(total_pixels),
                "carrier_pixels": carrier_pixels,
                "used_pixels": carrier_pixels,
                "used_pixel_fraction": float(record["used_pixel_fraction"]),
                "used_pixel_percentage": float(record["used_pixel_percentage"]),
                "probe_budget_fraction": float(fraction),
                "probe_budget_percentage": float(100.0 * fraction),
                "probe_budget_pixels": int(current_budget),
                "unique_probed_pixels": int(np.unique(current_positions).size),
                "guessed_carrier_pixels": guessed_carrier_pixels,
                "precision_hit_rate": guessed_carrier_pixels / current_budget if current_budget else 0.0,
                "carrier_recall": guessed_carrier_pixels / carrier_pixels if carrier_pixels else 0.0,
                "incremental_probe_pixels": int(incremental_budget),
                "incremental_guessed_pixels": int(incremental_guessed),
                "incremental_precision_hit_rate": incremental_guessed / incremental_budget if incremental_budget else 0.0,
            }
        )

        previous_budget = current_budget
        previous_guessed = guessed_carrier_pixels

    return rows


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

        summary_rows.append(
            {
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
        )
    return summary_rows


def run_experiment(run_probe_seed: int) -> tuple[list[dict], list[dict]]:
    expected_count = cover_image_count()
    if expected_count <= 0:
        raise FileNotFoundError(f"No cover TIFF images found in {COVER_ROOT.resolve()}")

    raw_rows: list[dict] = []
    for method in METHODS:
        for alpha in ALPHAS:
            config_dir = ensure_steganograms(method, alpha, expected_count)
            record_paths = iter_record_paths(config_dir)
            if LIMIT:
                record_paths = record_paths[:LIMIT]
            for record_path in record_paths:
                record = load_change_record(record_path)
                total_pixels = record["image_shape"][0] * record["image_shape"][1]
                max_budget = budget_pixels(total_pixels, max(PROBE_BUDGET_FRACTIONS))
                raw_rows.extend(probe_record(record, config_dir, max_budget, run_probe_seed))

    summary_rows = summarize(raw_rows)
    return raw_rows, summary_rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Budgeted carrier-pixel probing sweep. Defaults reproduce experiments/probing_only."
    )
    p.add_argument("--cover-root", type=Path, default=COVER_ROOT)
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
    global COVER_ROOT, EXPERIMENT_ROOT, STEGANOGRAM_ROOT, METHODS, ALPHAS
    global PROBE_BUDGET_FRACTIONS, PROBE_DISTRIBUTION, PROBE_IMAGE_SOURCE, PROBE_SEED, LIMIT
    global RAW_CSV, SUMMARY_CSV, METADATA_JSON
    COVER_ROOT = args.cover_root
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
