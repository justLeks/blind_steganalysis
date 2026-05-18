from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embedding import to_u8_array
from message_damage_experiment import apply_attack, compute_psnr, derive_seed
from read_changes import load_change_record, sample_pixel_indices


# Edit these settings and run: python3 generate_test_images.py
EXPERIMENT_ROOT = Path("experiments/multi_method_alpha")
COVER_ROOT = Path("ALASKA_v2_TIFF_512_GrayScale_50")
OUTPUT_ROOT = Path("experiments/test_images")

EXAMPLES = [
    {"method": "HUGO", "alpha": 0.3, "image_stem": "00001", "attack_budget": 20000},
    {"method": "MIPOD", "alpha": 0.3, "image_stem": "00001", "attack_budget": 20000},
]

PROBE_DISTRIBUTION = "texture_energy"
PROBE_DISTRIBUTION_PARAMS = {"gamma": 1.0, "floor": 1e-6}
PROBE_SEED = 41
ATTACK_MODE = "plus_minus_one"
ATTACK_SEED = 314159
DIFF_GAIN = 64


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def config_id(method: str, alpha: float) -> str:
    return f"{method.lower()}_alpha{alpha_tag(alpha)}"


def save_tiff(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8, copy=False)).save(path, format="TIFF")


def build_mask(image_shape: tuple[int, int], flat_indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
    mask[flat_indices.astype(np.int64, copy=False)] = 255
    return mask.reshape(image_shape)


def build_absdiff(reference: np.ndarray, target: np.ndarray, gain: int = DIFF_GAIN) -> np.ndarray:
    diff = np.abs(target.astype(np.int16) - reference.astype(np.int16))
    return np.clip(diff * int(gain), 0, 255).astype(np.uint8)


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_name = "Arial Bold.ttf" if bold else "Arial.ttf"
    font_path = Path("/System/Library/Fonts/Supplemental") / font_name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str) -> None:
    font = load_font(18, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2), text, font=font, fill=(24, 24, 24))


def save_panel(images: list[tuple[str, np.ndarray]], path: Path) -> None:
    cell_w, cell_h = 320, 300
    label_h = 46
    padding = 28
    cols = 3
    rows = 2
    panel = Image.new("RGB", (padding * 2 + cols * cell_w, padding * 2 + rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(panel)

    for idx, (label, array) in enumerate(images):
        row, col = divmod(idx, cols)
        x = padding + col * cell_w
        y = padding + row * (cell_h + label_h)
        img = Image.fromarray(array.astype(np.uint8, copy=False)).convert("L")
        img.thumbnail((cell_w - 20, cell_h - 20), Image.Resampling.LANCZOS)
        panel.paste(Image.merge("RGB", (img, img, img)), (x + (cell_w - img.width) // 2, y + 8))
        draw.rectangle((x + 6, y + 6, x + cell_w - 6, y + cell_h - 6), outline=(110, 110, 110), width=1)
        draw_label(draw, (x + 4, y + cell_h, x + cell_w - 4, y + cell_h + label_h), label)

    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def generate_example(example: dict) -> dict:
    method = str(example["method"]).upper()
    alpha = float(example["alpha"])
    image_stem = str(example["image_stem"])
    attack_budget = int(example["attack_budget"])
    cid = config_id(method, alpha)

    config_dir = (EXPERIMENT_ROOT / cid).expanduser().resolve()
    record_path = config_dir / f"{image_stem}_changes.npz"
    stego_path = config_dir / f"{image_stem}_stego.png"
    cover_path = (COVER_ROOT / f"{image_stem}.tif").expanduser().resolve()
    output_dir = (OUTPUT_ROOT / f"{cid}_{image_stem}_budget{attack_budget}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not record_path.exists():
        raise FileNotFoundError(f"Missing change record: {record_path}")
    if not stego_path.exists():
        raise FileNotFoundError(f"Missing stego image: {stego_path}")
    if not cover_path.exists():
        raise FileNotFoundError(f"Missing cover image: {cover_path}")

    record = load_change_record(record_path)
    cover = to_u8_array(cover_path)
    stego = to_u8_array(stego_path)
    attack_order = sample_pixel_indices(
        image_shape=record["image_shape"],
        sample_count=attack_budget,
        seed=derive_seed(PROBE_SEED, record["record_path"].name, PROBE_DISTRIBUTION),
        distribution=PROBE_DISTRIBUTION,
        distribution_params=PROBE_DISTRIBUTION_PARAMS,
        sampling_image=stego,
    )
    attacked = apply_attack(
        image=stego,
        attack_positions=attack_order,
        attack_seed=derive_seed(ATTACK_SEED, record["record_path"].name, str(attack_budget)),
        attack_mode=ATTACK_MODE,
    )

    changed_mask = build_mask(record["image_shape"], record["changed_flat_indices"])
    attack_mask = build_mask(record["image_shape"], attack_order)
    stego_diff = build_absdiff(cover, stego)
    attack_diff = build_absdiff(stego, attacked)

    outputs = {
        "cover": output_dir / "cover.tiff",
        "stego": output_dir / "stego.tiff",
        "attacked_stego": output_dir / "attacked_stego.tiff",
        "embedding_change_mask": output_dir / "embedding_change_mask.tiff",
        "attack_mask": output_dir / "attack_mask.tiff",
        "stego_vs_cover_absdiff_x64": output_dir / "stego_vs_cover_absdiff_x64.tiff",
        "attack_vs_stego_absdiff_x64": output_dir / "attack_vs_stego_absdiff_x64.tiff",
        "panel": output_dir / "panel.png",
        "metadata": output_dir / "metadata.json",
    }

    save_tiff(cover, outputs["cover"])
    save_tiff(stego, outputs["stego"])
    save_tiff(attacked, outputs["attacked_stego"])
    save_tiff(changed_mask, outputs["embedding_change_mask"])
    save_tiff(attack_mask, outputs["attack_mask"])
    save_tiff(stego_diff, outputs["stego_vs_cover_absdiff_x64"])
    save_tiff(attack_diff, outputs["attack_vs_stego_absdiff_x64"])
    save_panel(
        [
            ("Cover", cover),
            ("Stego", stego),
            ("Attacked stego", attacked),
            ("Embedding mask", changed_mask),
            ("Attack mask", attack_mask),
            ("Attack diff x64", attack_diff),
        ],
        outputs["panel"],
    )

    changed_lookup = np.zeros(cover.size, dtype=bool)
    changed_lookup[record["changed_flat_indices"]] = True
    attacked_changed = int(np.count_nonzero(changed_lookup[attack_order]))
    metadata = {
        "config_id": cid,
        "method": method,
        "alpha": alpha,
        "image_stem": image_stem,
        "attack_budget": attack_budget,
        "probe_distribution": PROBE_DISTRIBUTION,
        "probe_distribution_params": PROBE_DISTRIBUTION_PARAMS,
        "attack_mode": ATTACK_MODE,
        "changed_pixels": int(record["changed_flat_indices"].size),
        "change_rate": float(record["changed_flat_indices"].size / cover.size),
        "attacked_changed_pixels": attacked_changed,
        "attack_hit_rate_vs_embedding_changes": float(attacked_changed / attack_budget),
        "embedding_change_recall_at_budget": float(attacked_changed / record["changed_flat_indices"].size),
        "psnr_cover_to_stego": float(compute_psnr(cover, stego)),
        "psnr_stego_to_attacked": float(compute_psnr(stego, attacked)),
        "source_record": str(record_path),
        "source_stego": str(stego_path),
        "source_cover": str(cover_path),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = [generate_example(example) for example in EXAMPLES]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_ROOT / "index.json"
    index_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(metadata)} test image sets to {OUTPUT_ROOT.resolve()}")
    for item in metadata:
        print(f"{item['config_id']} {item['image_stem']}: {item['outputs']['panel']}")


if __name__ == "__main__":
    main()
