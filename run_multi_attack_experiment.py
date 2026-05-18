from __future__ import annotations

import csv
import json
from pathlib import Path

from embedding import embed_and_log
import message_damage_experiment as damage


# Edit these settings and run: python3 run_multi_attack_experiment.py
COVER_ROOT = Path("ALASKA_v2_TIFF_512_GrayScale_50")
EXPERIMENT_ROOT = Path("experiments/multi_method_alpha")
METHODS = ["HUGO", "MIPOD"]
ALPHAS = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
EMBED_SEED = 12345

COMBINED_DETAIL_CSV = EXPERIMENT_ROOT / "combined_message_damage_detail.csv"
COMBINED_SUMMARY_CSV = EXPERIMENT_ROOT / "combined_message_damage_summary.csv"
COMBINED_METADATA_JSON = EXPERIMENT_ROOT / "metadata.json"


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def config_id(method: str, alpha: float) -> str:
    return f"{method.lower()}_alpha{alpha_tag(alpha)}"


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def manifest_is_complete(output_dir: Path, method: str, alpha: float) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if len(manifest) != 100:
        return False
    return all(
        item.get("method") == method.upper() and abs(float(item.get("alpha", -1)) - alpha) < 1e-12
        for item in manifest
    )


def config_attack_is_complete(output_dir: Path) -> bool:
    detail_path = output_dir / "message_damage_detail.csv"
    summary_path = output_dir / "message_damage_summary.csv"
    detail_rows = read_csv_rows(detail_path)
    summary_rows = read_csv_rows(summary_path)
    return len(detail_rows) == 100 * len(damage.ATTACK_BUDGETS) and len(summary_rows) == len(damage.ATTACK_BUDGETS)


def add_config_columns(rows: list[dict], method: str, alpha: float) -> list[dict]:
    cid = config_id(method, alpha)
    updated: list[dict] = []
    for row in rows:
        prefixed = {
            "config_id": cid,
            "method": method.upper(),
            "alpha": float(alpha),
        }
        for key, value in row.items():
            if key not in prefixed:
                prefixed[key] = value
        updated.append(prefixed)
    return updated


def run_config(method: str, alpha: float) -> tuple[list[dict], list[dict]]:
    cid = config_id(method, alpha)
    output_dir = (EXPERIMENT_ROOT / cid).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_is_complete(output_dir, method, alpha):
        print(f"[embed] {cid}")
        embed_and_log(
            src_dir=COVER_ROOT,
            dst_dir=output_dir,
            method=method,
            alpha=alpha,
            seed=EMBED_SEED,
        )
    else:
        print(f"[embed:skip] {cid}")

    if not config_attack_is_complete(output_dir):
        print(f"[attack] {cid}")
        damage.INPUT_PATH = output_dir
        damage.COVER_ROOT = COVER_ROOT
        damage.DETAIL_CSV_PATH = output_dir / "message_damage_detail.csv"
        damage.SUMMARY_CSV_PATH = output_dir / "message_damage_summary.csv"
        detail_rows, summary_rows = damage.run_experiment()
        damage.write_csv(damage.DETAIL_CSV_PATH, detail_rows)
        damage.write_csv(damage.SUMMARY_CSV_PATH, summary_rows)
    else:
        print(f"[attack:skip] {cid}")

    detail_rows = add_config_columns(read_csv_rows(output_dir / "message_damage_detail.csv"), method, alpha)
    summary_rows = add_config_columns(read_csv_rows(output_dir / "message_damage_summary.csv"), method, alpha)
    return detail_rows, summary_rows


def main() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    all_detail_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for method in METHODS:
        for alpha in ALPHAS:
            detail_rows, summary_rows = run_config(method, alpha)
            all_detail_rows.extend(detail_rows)
            all_summary_rows.extend(summary_rows)
            write_csv(COMBINED_DETAIL_CSV, all_detail_rows)
            write_csv(COMBINED_SUMMARY_CSV, all_summary_rows)

    metadata = {
        "cover_root": str(COVER_ROOT.resolve()),
        "experiment_root": str(EXPERIMENT_ROOT.resolve()),
        "methods": METHODS,
        "alphas": ALPHAS,
        "embed_seed": EMBED_SEED,
        "attack_budgets": damage.ATTACK_BUDGETS,
        "probe_distribution": damage.PROBE_DISTRIBUTION,
        "probe_distribution_params": damage.PROBE_DISTRIBUTION_PARAMS,
        "attack_mode": damage.ATTACK_MODE,
        "detail_rows": len(all_detail_rows),
        "summary_rows": len(all_summary_rows),
        "combined_detail_csv": str(COMBINED_DETAIL_CSV.resolve()),
        "combined_summary_csv": str(COMBINED_SUMMARY_CSV.resolve()),
    }
    COMBINED_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_detail_rows)} detail rows to {COMBINED_DETAIL_CSV.resolve()}")
    print(f"Wrote {len(all_summary_rows)} summary rows to {COMBINED_SUMMARY_CSV.resolve()}")


if __name__ == "__main__":
    main()
