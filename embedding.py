import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_METHODS = {"HUGO", "MIPOD"}


def to_u8_array(image_path: Path) -> np.ndarray:
    image = Image.open(image_path)
    if image.mode not in {"L", "I;16", "I"}:
        raise ValueError(
            f"Expected a grayscale image for ALASKA input, got mode '{image.mode}' for {image_path}."
        )
    array = np.array(image)
    if array.dtype == np.uint16:
        return (array / 257).astype(np.uint8)
    return array.astype(np.uint8, copy=False)


def iter_image_files(src_dir: Path) -> Iterable[Path]:
    for path in sorted(src_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def derive_image_seed(base_seed: int, relative_path: Path) -> int:
    digest = hashlib.sha256(relative_path.as_posix().encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return (int(base_seed) + offset) % (2**32)


def simulate_embedding(
    cover_image: np.ndarray,
    method: str,
    alpha: float,
    seed: int,
) -> np.ndarray:
    import conseal as cl

    normalized_method = method.upper()
    if normalized_method == "HUGO":
        return cl.hugo.simulate_single_channel(x0=cover_image, alpha=alpha, seed=seed)
    if normalized_method == "MIPOD":
        return cl.mipod.simulate_single_channel(x0=cover_image, alpha=alpha, seed=seed)
    raise ValueError(f"Unsupported method '{method}'. Supported methods: {sorted(SUPPORTED_METHODS)}")


def change_indices(stego_image: np.ndarray, cover_image: np.ndarray) -> np.ndarray:
    if stego_image.shape != cover_image.shape:
        raise ValueError("Cover and stego image shapes do not match.")
    return np.flatnonzero(stego_image != cover_image)


def indices_to_coords(flat_indices: np.ndarray, width: int) -> np.ndarray:
    if flat_indices.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    rows = flat_indices // width
    cols = flat_indices % width
    return np.column_stack((rows, cols))


def save_change_record(
    record_path: Path,
    relative_cover_path: Path,
    relative_stego_path: Path,
    image_shape: tuple[int, int],
    changed_flat_indices: np.ndarray,
    method: str,
    alpha: float,
    seed: int,
) -> None:
    coords = indices_to_coords(changed_flat_indices, image_shape[1])
    total_pixels = image_shape[0] * image_shape[1]
    used_pixel_fraction = changed_flat_indices.size / total_pixels if total_pixels else 0.0
    np.savez_compressed(
        record_path,
        changed_flat_indices=changed_flat_indices.astype(np.int64, copy=False),
        changed_coords=coords.astype(np.int64, copy=False),
        image_shape=np.array(image_shape, dtype=np.int64),
        cover_relative_path=np.array(str(relative_cover_path)),
        stego_relative_path=np.array(str(relative_stego_path)),
        method=np.array(method.upper()),
        alpha=np.array(float(alpha)),
        seed=np.array(int(seed), dtype=np.int64),
        num_changed=np.array(int(changed_flat_indices.size), dtype=np.int64),
        used_pixel_fraction=np.array(float(used_pixel_fraction)),
        used_pixel_percentage=np.array(float(100.0 * used_pixel_fraction)),
    )


def write_manifest(manifest_path: Path, records: list[dict]) -> None:
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def embed_and_log(
    src_dir: str | Path,
    dst_dir: str | Path,
    method: str = "HUGO",
    alpha: float = 0.4,
    seed: int = 12345,
) -> list[dict]:
    src_dir = Path(src_dir).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src_dir}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")

    image_paths = list(iter_image_files(src_dir))
    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files were found in {src_dir}. Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    normalized_method = method.upper()
    if normalized_method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods: {sorted(SUPPORTED_METHODS)}"
        )

    for image_path in image_paths:
        relative_path = image_path.relative_to(src_dir)
        image_seed = derive_image_seed(seed, relative_path)
        cover_image = to_u8_array(image_path)
        stego_image = simulate_embedding(
            cover_image=cover_image,
            method=normalized_method,
            alpha=alpha,
            seed=image_seed,
        )
        changed_flat_indices = change_indices(stego_image=stego_image, cover_image=cover_image)

        out_dir = dst_dir / relative_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        stego_filename = f"{image_path.stem}_stego.png"
        record_filename = f"{image_path.stem}_changes.npz"

        stego_path = out_dir / stego_filename
        record_path = out_dir / record_filename

        Image.fromarray(stego_image).save(stego_path)
        save_change_record(
            record_path=record_path,
            relative_cover_path=relative_path,
            relative_stego_path=relative_path.parent / stego_filename,
            image_shape=cover_image.shape,
            changed_flat_indices=changed_flat_indices,
            method=normalized_method,
            alpha=alpha,
            seed=image_seed,
        )

        record = {
            "cover_image": str(relative_path),
            "stego_image": str(relative_path.parent / stego_filename),
            "change_record": str(relative_path.parent / record_filename),
            "method": normalized_method,
            "alpha": float(alpha),
            "seed": int(image_seed),
            "height": int(cover_image.shape[0]),
            "width": int(cover_image.shape[1]),
            "used_pixels": int(changed_flat_indices.size),
            "used_pixel_fraction": float(changed_flat_indices.size / cover_image.size),
            "used_pixel_percentage": float(100.0 * changed_flat_indices.size / cover_image.size),
            "num_changed": int(changed_flat_indices.size),
            "change_rate": float(changed_flat_indices.size / cover_image.size),
        }
        records.append(record)

    write_manifest(dst_dir / "manifest.json", records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create stego images and log changed pixels for blind steganalysis experiments."
    )
    parser.add_argument("src_dir", help="Directory with cover images.")
    parser.add_argument("dst_dir", help="Directory where stego images and metadata will be saved.")
    parser.add_argument("--method", default="HUGO", choices=sorted(SUPPORTED_METHODS))
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=12345)
    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()
    manifest = embed_and_log(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        method=args.method,
        alpha=args.alpha,
        seed=args.seed,
    )
    print(f"Processed {len(manifest)} images into {Path(args.dst_dir).expanduser().resolve()}")
