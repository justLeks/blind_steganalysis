from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from embedding import (
    change_indices,
    derive_image_seed,
    save_change_record,
    simulate_embedding,
    to_u8_array,
    write_manifest,
)
from message_damage_experiment import apply_attack, compute_psnr, derive_seed
from read_changes import SUPPORTED_PROBE_DISTRIBUTIONS, sample_pixel_indices


# Edit these settings and run: python3 visual_attack_test.py
COVER_ROOT = Path("ALASKA_v2_TIFF_512_GrayScale_50")
COVER_IMAGE = COVER_ROOT / "00001.tif"
OUTPUT_ROOT = Path("experiments/visual_attack_test")
TEST_NAME = "single_image_visual_attack"

METHOD = "HUGO"
ALPHA = 0.3
EMBED_BASE_SEED = 12345

PROBE_DISTRIBUTION = "texture_energy"
PROBE_DISTRIBUTION_PARAMS = {
    "gamma": 1.0,
    "floor": 1e-6,
}
PROBE_IMAGE_SOURCE = "stego"
PROBE_SEED = 41

ATTACK_MODE = "plus_minus_one"
ATTACK_SEED = 314159
ATTACK_BUDGETS = [1000, 5000, 10000, 20000]

DIFF_GAIN = 64


def resolve_relative_cover_path(cover_path: Path, cover_root: Path) -> Path:
    cover_path = cover_path.expanduser().resolve()
    cover_root = cover_root.expanduser().resolve()
    try:
        return cover_path.relative_to(cover_root)
    except ValueError:
        return Path(cover_path.name)


def save_tiff(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8, copy=False)).save(path, format="TIFF")


def build_mask_image(image_shape: tuple[int, int], flat_indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
    mask[flat_indices.astype(np.int64, copy=False)] = 255
    return mask.reshape(image_shape)


def build_amplified_abs_diff(reference: np.ndarray, target: np.ndarray, gain: int) -> np.ndarray:
    diff = np.abs(target.astype(np.int16) - reference.astype(np.int16))
    amplified = np.clip(diff * int(gain), 0, 255)
    return amplified.astype(np.uint8)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_attack_order_csv(
    path: Path,
    attack_order: np.ndarray,
    image_shape: tuple[int, int],
    changed_lookup: np.ndarray,
) -> None:
    width = int(image_shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "flat_index", "row", "col", "was_embedding_change"],
        )
        writer.writeheader()
        for rank, flat_index in enumerate(attack_order, start=1):
            writer.writerow(
                {
                    "rank": int(rank),
                    "flat_index": int(flat_index),
                    "row": int(flat_index // width),
                    "col": int(flat_index % width),
                    "was_embedding_change": int(changed_lookup[int(flat_index)]),
                }
            )


def build_output_dir(relative_cover_path: Path) -> Path:
    alpha_tag = str(ALPHA).replace(".", "p")
    dirname = (
        f"{TEST_NAME}_{relative_cover_path.stem}_{METHOD.lower()}_alpha{alpha_tag}_"
        f"{PROBE_DISTRIBUTION}_{ATTACK_MODE}"
    )
    return OUTPUT_ROOT / dirname


def select_sampling_image(cover_image: np.ndarray, stego_image: np.ndarray) -> np.ndarray | None:
    if PROBE_DISTRIBUTION not in SUPPORTED_PROBE_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported probe distribution '{PROBE_DISTRIBUTION}'. "
            f"Supported: {sorted(SUPPORTED_PROBE_DISTRIBUTIONS)}"
        )
    if PROBE_DISTRIBUTION != "texture_energy":
        return None
    if PROBE_IMAGE_SOURCE == "cover":
        return cover_image
    if PROBE_IMAGE_SOURCE == "stego":
        return stego_image
    raise ValueError("PROBE_IMAGE_SOURCE must be 'cover' or 'stego'.")


def run_visual_attack_test() -> dict:
    cover_path = COVER_IMAGE.expanduser().resolve()
    cover_root = COVER_ROOT.expanduser().resolve()
    relative_cover_path = resolve_relative_cover_path(cover_path, cover_root)
    output_dir = build_output_dir(relative_cover_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cover_image = to_u8_array(cover_path)
    embed_seed = derive_image_seed(EMBED_BASE_SEED, relative_cover_path)
    stego_image = simulate_embedding(
        cover_image=cover_image,
        method=METHOD,
        alpha=ALPHA,
        seed=embed_seed,
    )
    changed_flat_indices = change_indices(stego_image=stego_image, cover_image=cover_image)
    changed_lookup = np.zeros(cover_image.size, dtype=bool)
    changed_lookup[changed_flat_indices] = True

    cover_tiff_path = output_dir / f"{relative_cover_path.stem}_cover.tiff"
    stego_tiff_path = output_dir / f"{relative_cover_path.stem}_stego.tiff"
    change_mask_tiff_path = output_dir / f"{relative_cover_path.stem}_embedding_change_mask.tiff"
    stego_diff_tiff_path = output_dir / f"{relative_cover_path.stem}_stego_vs_cover_absdiff_x{DIFF_GAIN}.tiff"
    change_record_path = output_dir / f"{relative_cover_path.stem}_changes.npz"
    manifest_path = output_dir / "manifest.json"
    attack_order_csv_path = output_dir / "attack_order.csv"
    metadata_json_path = output_dir / "metadata.json"
    attack_summary_csv_path = output_dir / "attack_summary.csv"

    save_tiff(cover_image, cover_tiff_path)
    save_tiff(stego_image, stego_tiff_path)
    save_tiff(build_mask_image(cover_image.shape, changed_flat_indices), change_mask_tiff_path)
    save_tiff(build_amplified_abs_diff(cover_image, stego_image, DIFF_GAIN), stego_diff_tiff_path)

    save_change_record(
        record_path=change_record_path,
        relative_cover_path=relative_cover_path,
        relative_stego_path=Path(stego_tiff_path.name),
        image_shape=cover_image.shape,
        changed_flat_indices=changed_flat_indices,
        method=METHOD,
        alpha=ALPHA,
        seed=embed_seed,
    )
    manifest_record = {
        "cover_image": str(relative_cover_path),
        "stego_image": stego_tiff_path.name,
        "change_record": change_record_path.name,
        "method": METHOD.upper(),
        "alpha": float(ALPHA),
        "seed": int(embed_seed),
        "height": int(cover_image.shape[0]),
        "width": int(cover_image.shape[1]),
        "used_pixels": int(changed_flat_indices.size),
        "used_pixel_fraction": float(changed_flat_indices.size / cover_image.size),
        "used_pixel_percentage": float(100.0 * changed_flat_indices.size / cover_image.size),
        "num_changed": int(changed_flat_indices.size),
        "change_rate": float(changed_flat_indices.size / cover_image.size),
    }
    write_manifest(manifest_path, [manifest_record])

    budgets = sorted(set(int(budget) for budget in ATTACK_BUDGETS))
    if not budgets:
        raise ValueError("ATTACK_BUDGETS must not be empty.")
    if budgets[0] <= 0:
        raise ValueError("ATTACK_BUDGETS must contain positive integers.")

    sampling_image = select_sampling_image(cover_image=cover_image, stego_image=stego_image)
    attack_order_seed = derive_seed(
        PROBE_SEED,
        relative_cover_path.as_posix(),
        PROBE_DISTRIBUTION,
        "attack_order",
    )
    attack_order = sample_pixel_indices(
        image_shape=cover_image.shape,
        sample_count=max(budgets),
        seed=attack_order_seed,
        distribution=PROBE_DISTRIBUTION,
        distribution_params=PROBE_DISTRIBUTION_PARAMS,
        sampling_image=sampling_image,
    )
    write_attack_order_csv(
        path=attack_order_csv_path,
        attack_order=attack_order,
        image_shape=cover_image.shape,
        changed_lookup=changed_lookup,
    )

    attack_rows: list[dict] = []
    attack_metadata: list[dict] = []
    for attack_budget in budgets:
        attack_positions = attack_order[:attack_budget]
        perturbation_seed = derive_seed(
            ATTACK_SEED,
            relative_cover_path.as_posix(),
            ATTACK_MODE,
            str(attack_budget),
        )
        attacked_image = apply_attack(
            image=stego_image,
            attack_positions=attack_positions,
            attack_seed=perturbation_seed,
            attack_mode=ATTACK_MODE,
        )

        attacked_tiff_path = output_dir / f"{relative_cover_path.stem}_attacked_{attack_budget:06d}.tiff"
        attack_mask_tiff_path = output_dir / f"{relative_cover_path.stem}_attack_mask_{attack_budget:06d}.tiff"
        attack_diff_tiff_path = output_dir / (
            f"{relative_cover_path.stem}_attack_vs_stego_absdiff_x{DIFF_GAIN}_{attack_budget:06d}.tiff"
        )

        save_tiff(attacked_image, attacked_tiff_path)
        save_tiff(build_mask_image(cover_image.shape, attack_positions), attack_mask_tiff_path)
        save_tiff(build_amplified_abs_diff(stego_image, attacked_image, DIFF_GAIN), attack_diff_tiff_path)

        attacked_embedding_changes = int(np.count_nonzero(changed_lookup[attack_positions]))
        nonzero_delta = int(np.count_nonzero(attacked_image != stego_image))
        mean_abs_delta = float(
            np.mean(np.abs(attacked_image.astype(np.int16) - stego_image.astype(np.int16)))
        )
        max_abs_delta = int(
            np.max(np.abs(attacked_image.astype(np.int16) - stego_image.astype(np.int16)))
        )

        row = {
            "attack_budget": int(attack_budget),
            "attack_order_seed": int(attack_order_seed),
            "attack_perturbation_seed": int(perturbation_seed),
            "attack_mode": ATTACK_MODE,
            "probe_distribution": PROBE_DISTRIBUTION,
            "probe_image_source": PROBE_IMAGE_SOURCE if PROBE_DISTRIBUTION == "texture_energy" else "",
            "attacked_pixels": int(attack_positions.size),
            "attacked_embedding_changes": attacked_embedding_changes,
            "attack_hit_rate_vs_embedding_changes": (
                attacked_embedding_changes / attack_positions.size if attack_positions.size else 0.0
            ),
            "embedding_changed_pixels_total": int(changed_flat_indices.size),
            "embedding_change_recall_at_budget": (
                attacked_embedding_changes / changed_flat_indices.size if changed_flat_indices.size else 0.0
            ),
            "nonzero_delta_pixels_vs_stego": nonzero_delta,
            "mean_abs_delta_vs_stego": mean_abs_delta,
            "max_abs_delta_vs_stego": max_abs_delta,
            "psnr_db_vs_stego": float(compute_psnr(stego_image, attacked_image)),
            "psnr_db_vs_cover": float(compute_psnr(cover_image, attacked_image)),
            "attacked_tiff": attacked_tiff_path.name,
            "attack_mask_tiff": attack_mask_tiff_path.name,
            "attack_absdiff_tiff": attack_diff_tiff_path.name,
        }
        attack_rows.append(row)
        attack_metadata.append(dict(row))

    with attack_summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attack_rows[0].keys()))
        writer.writeheader()
        for row in attack_rows:
            writer.writerow(row)

    metadata = {
        "script": "visual_attack_test.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "config": {
            "cover_root": str(cover_root),
            "cover_image": str(cover_path),
            "method": METHOD.upper(),
            "alpha": float(ALPHA),
            "embed_base_seed": int(EMBED_BASE_SEED),
            "probe_distribution": PROBE_DISTRIBUTION,
            "probe_distribution_params": dict(PROBE_DISTRIBUTION_PARAMS),
            "probe_image_source": PROBE_IMAGE_SOURCE,
            "probe_seed": int(PROBE_SEED),
            "attack_mode": ATTACK_MODE,
            "attack_seed": int(ATTACK_SEED),
            "attack_budgets": budgets,
            "diff_gain": int(DIFF_GAIN),
        },
        "embedding": {
            "cover_relative_path": str(relative_cover_path),
            "embedding_seed": int(embed_seed),
            "image_height": int(cover_image.shape[0]),
            "image_width": int(cover_image.shape[1]),
            "num_changed": int(changed_flat_indices.size),
            "change_rate": float(changed_flat_indices.size / cover_image.size),
            "psnr_db_cover_to_stego": float(compute_psnr(cover_image, stego_image)),
            "cover_tiff": cover_tiff_path.name,
            "stego_tiff": stego_tiff_path.name,
            "embedding_change_mask_tiff": change_mask_tiff_path.name,
            "stego_vs_cover_absdiff_tiff": stego_diff_tiff_path.name,
            "change_record": change_record_path.name,
            "manifest": manifest_path.name,
        },
        "attack_order": {
            "attack_order_seed": int(attack_order_seed),
            "attack_order_csv": attack_order_csv_path.name,
            "max_budget": int(max(budgets)),
        },
        "attacks": attack_metadata,
    }
    write_json(metadata_json_path, metadata)

    return metadata


if __name__ == "__main__":
    metadata = run_visual_attack_test()
    print(f"Wrote visual attack test to {metadata['output_dir']}")
    print(f"Embedding change rate: {metadata['embedding']['change_rate']:.6f}")
    for attack in metadata["attacks"]:
        print(
            "budget={attack_budget} attacked_changed={attacked_embedding_changes} "
            "psnr_vs_stego={psnr_db_vs_stego:.2f}dB".format(**attack)
        )
