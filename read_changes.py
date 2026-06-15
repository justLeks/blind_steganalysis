import argparse
import csv
import json
from pathlib import Path

import numpy as np

from embedding import to_u8_array


SUPPORTED_PROBE_DISTRIBUTIONS = {
    "uniform",
    "center_gaussian",
    "center_laplace",
    "edge_gaussian",
    "laplacian_residual",
    "texture_energy",
}
IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS = {"texture_energy", "laplacian_residual"}
SUPPORTED_PROBE_IMAGE_SOURCES = {"cover", "stego"}


def iter_record_paths(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix != ".npz":
            raise ValueError(f"Expected a .npz change record, got: {input_path}")
        return [input_path]

    manifest_path = input_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record_paths = [input_path / item["change_record"] for item in manifest]
    else:
        record_paths = sorted(input_path.rglob("*_changes.npz"))

    if not record_paths:
        raise FileNotFoundError(f"No *_changes.npz records found under {input_path}")
    return [path.resolve() for path in record_paths]


def load_change_record(record_path: str | Path) -> dict:
    record_path = Path(record_path).expanduser().resolve()
    with np.load(record_path, allow_pickle=False) as data:
        image_shape = tuple(int(x) for x in data["image_shape"])
        changed_flat_indices = data["changed_flat_indices"].astype(np.int64, copy=False)
        changed_coords = data["changed_coords"].astype(np.int64, copy=False)
        total_pixels = image_shape[0] * image_shape[1]
        used_pixel_fraction = (
            float(data["used_pixel_fraction"].item())
            if "used_pixel_fraction" in data.files
            else (len(changed_flat_indices) / total_pixels if total_pixels else 0.0)
        )
        used_pixel_percentage = (
            float(data["used_pixel_percentage"].item())
            if "used_pixel_percentage" in data.files
            else 100.0 * used_pixel_fraction
        )
        return {
            "record_path": record_path,
            "image_shape": image_shape,
            "changed_flat_indices": changed_flat_indices,
            "changed_coords": changed_coords,
            "cover_relative_path": str(data["cover_relative_path"].item()),
            "stego_relative_path": str(data["stego_relative_path"].item()),
            "method": str(data["method"].item()),
            "alpha": float(data["alpha"].item()),
            "seed": int(data["seed"].item()),
            "num_changed": int(data["num_changed"].item()),
            "used_pixels": int(len(changed_flat_indices)),
            "used_pixel_fraction": used_pixel_fraction,
            "used_pixel_percentage": used_pixel_percentage,
        }


def reconstruct_change_mask(image_shape: tuple[int, int], changed_flat_indices: np.ndarray) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros(height * width, dtype=bool)
    mask[changed_flat_indices] = True
    return mask.reshape(image_shape)


def compare_cover_and_stego(
    cover_path: str | Path,
    stego_path: str | Path,
    change_mask: np.ndarray,
) -> dict:
    cover = to_u8_array(Path(cover_path)).astype(np.int16)
    stego = to_u8_array(Path(stego_path)).astype(np.int16)
    if cover.shape != stego.shape:
        raise ValueError("Cover and stego image shapes do not match.")

    delta = stego - cover
    return {
        "delta_min": int(delta.min()),
        "delta_max": int(delta.max()),
        "plus_one_changes": int(np.count_nonzero(delta[change_mask] == 1)),
        "minus_one_changes": int(np.count_nonzero(delta[change_mask] == -1)),
        "nonzero_outside_mask": int(np.count_nonzero(delta[~change_mask] != 0)),
    }


def _resolve_positive_scale(
    params: dict | None,
    key: str,
    fallback_scale: float,
) -> float:
    value = float((params or {}).get(key, fallback_scale))
    if value <= 0:
        raise ValueError(f"Expected '{key}' to be positive, got {value}.")
    return value


def _resolve_nonnegative_value(
    params: dict | None,
    key: str,
    fallback_value: float,
) -> float:
    value = float((params or {}).get(key, fallback_value))
    if value < 0:
        raise ValueError(f"Expected '{key}' to be non-negative, got {value}.")
    return value


def build_texture_energy_probabilities(
    sampling_image: np.ndarray,
    distribution_params: dict | None = None,
) -> np.ndarray:
    params = distribution_params or {}
    gamma = _resolve_positive_scale(params, "gamma", 1.0)
    floor = _resolve_nonnegative_value(params, "floor", 1e-6)

    image = sampling_image.astype(np.float32, copy=False)

    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)
    gradient_x[:, 1:] = np.abs(image[:, 1:] - image[:, :-1])
    gradient_y[1:, :] = np.abs(image[1:, :] - image[:-1, :])

    laplacian = np.zeros_like(image, dtype=np.float32)
    laplacian[1:-1, 1:-1] = np.abs(
        4.0 * image[1:-1, 1:-1]
        - image[:-2, 1:-1]
        - image[2:, 1:-1]
        - image[1:-1, :-2]
        - image[1:-1, 2:]
    )

    energy = gradient_x + gradient_y + laplacian
    weights = (energy + floor) ** gamma
    flat_weights = weights.reshape(-1).astype(np.float64, copy=False)
    weight_sum = float(flat_weights.sum())
    if weight_sum <= 0:
        raise ValueError("Texture-energy probing produced non-positive total weight.")
    return flat_weights / weight_sum


def build_laplacian_residual_probabilities(
    sampling_image: np.ndarray,
    distribution_params: dict | None = None,
) -> np.ndarray:
    params = distribution_params or {}
    gamma = _resolve_positive_scale(params, "gamma", 1.0)
    floor = _resolve_nonnegative_value(params, "floor", 1e-6)

    image = sampling_image.astype(np.float32, copy=False)
    laplacian = np.zeros_like(image, dtype=np.float32)
    laplacian[1:-1, 1:-1] = np.abs(
        4.0 * image[1:-1, 1:-1]
        - image[:-2, 1:-1]
        - image[2:, 1:-1]
        - image[1:-1, :-2]
        - image[1:-1, 2:]
    )

    weights = (laplacian + floor) ** gamma
    flat_weights = weights.reshape(-1).astype(np.float64, copy=False)
    weight_sum = float(flat_weights.sum())
    if weight_sum <= 0:
        raise ValueError("Laplacian-residual probing produced non-positive total weight.")
    return flat_weights / weight_sum


def build_sampling_probabilities(
    image_shape: tuple[int, int],
    distribution: str,
    distribution_params: dict | None = None,
    sampling_image: np.ndarray | None = None,
) -> np.ndarray | None:
    height, width = image_shape
    if distribution == "uniform":
        return None

    params = distribution_params or {}
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    center_row = 0.5 * (height - 1)
    center_col = 0.5 * (width - 1)

    if distribution == "texture_energy":
        if sampling_image is None:
            raise ValueError("Texture-energy probing requires a sampling image.")
        if tuple(sampling_image.shape) != tuple(image_shape):
            raise ValueError(
                "Sampling image shape does not match the recorded image shape: "
                f"{sampling_image.shape} vs {image_shape}"
            )
        return build_texture_energy_probabilities(
            sampling_image=sampling_image,
            distribution_params=params,
        )

    if distribution == "laplacian_residual":
        if sampling_image is None:
            raise ValueError("Laplacian-residual probing requires a sampling image.")
        if tuple(sampling_image.shape) != tuple(image_shape):
            raise ValueError(
                "Sampling image shape does not match the recorded image shape: "
                f"{sampling_image.shape} vs {image_shape}"
            )
        return build_laplacian_residual_probabilities(
            sampling_image=sampling_image,
            distribution_params=params,
        )

    if distribution == "center_gaussian":
        sigma_row = _resolve_positive_scale(params, "sigma_row", 0.2 * height)
        sigma_col = _resolve_positive_scale(params, "sigma_col", 0.2 * width)
        weights = np.exp(
            -0.5 * (((rows - center_row) / sigma_row) ** 2 + ((cols - center_col) / sigma_col) ** 2)
        )
    elif distribution == "center_laplace":
        scale_row = _resolve_positive_scale(params, "scale_row", 0.15 * height)
        scale_col = _resolve_positive_scale(params, "scale_col", 0.15 * width)
        weights = np.exp(
            -(np.abs(rows - center_row) / scale_row + np.abs(cols - center_col) / scale_col)
        )
    elif distribution == "edge_gaussian":
        sigma_edge = _resolve_positive_scale(params, "sigma_edge", 0.15 * min(height, width))
        row_distance = np.minimum(rows, height - 1 - rows)
        col_distance = np.minimum(cols, width - 1 - cols)
        edge_distance = np.minimum(row_distance, col_distance)
        weights = np.exp(-0.5 * (edge_distance / sigma_edge) ** 2)
    else:
        raise ValueError(
            f"Unsupported probe distribution '{distribution}'. Supported: {sorted(SUPPORTED_PROBE_DISTRIBUTIONS)}"
        )

    flat_weights = weights.reshape(-1)
    weight_sum = float(flat_weights.sum())
    if weight_sum <= 0:
        raise ValueError(f"Probe distribution '{distribution}' produced non-positive total weight.")
    return flat_weights / weight_sum


def validate_sample_count(image_shape: tuple[int, int], sample_count: int) -> int:
    total_pixels = image_shape[0] * image_shape[1]
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative.")
    if sample_count > total_pixels:
        raise ValueError(
            f"sample_count={sample_count} exceeds the number of pixels ({total_pixels})."
        )
    return total_pixels


def sample_pixel_labels(
    image_shape: tuple[int, int],
    changed_flat_indices: np.ndarray,
    sample_count: int,
    seed: int,
    distribution: str = "uniform",
    distribution_params: dict | None = None,
    sampling_image: np.ndarray | None = None,
) -> np.ndarray:
    height, width = image_shape
    total_pixels = validate_sample_count(image_shape, sample_count)

    probabilities = build_sampling_probabilities(
        image_shape=image_shape,
        distribution=distribution,
        distribution_params=distribution_params,
        sampling_image=sampling_image,
    )
    sampled_flat_indices = sample_pixel_indices(
        image_shape=image_shape,
        sample_count=sample_count,
        seed=seed,
        distribution=distribution,
        distribution_params=distribution_params,
        sampling_image=sampling_image,
        probabilities=probabilities,
    )
    changed_lookup = np.zeros(total_pixels, dtype=np.uint8)
    changed_lookup[changed_flat_indices] = 1

    rows = sampled_flat_indices // width
    cols = sampled_flat_indices % width
    labels = changed_lookup[sampled_flat_indices]
    return np.column_stack((rows, cols, labels))


def sample_pixel_indices(
    image_shape: tuple[int, int],
    sample_count: int,
    seed: int,
    distribution: str = "uniform",
    distribution_params: dict | None = None,
    sampling_image: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    total_pixels = validate_sample_count(image_shape, sample_count)

    rng = np.random.default_rng(seed)
    if probabilities is None:
        probabilities = build_sampling_probabilities(
            image_shape=image_shape,
            distribution=distribution,
            distribution_params=distribution_params,
            sampling_image=sampling_image,
        )
    sampled_flat_indices = rng.choice(
        total_pixels,
        size=sample_count,
        replace=False,
        p=probabilities,
    )
    return sampled_flat_indices.astype(np.int64, copy=False)


def audit_samples(
    record: dict,
    sample_count: int,
    seed: int | None,
    distribution: str = "uniform",
    distribution_params: dict | None = None,
    sampling_image: np.ndarray | None = None,
) -> np.ndarray:
    audit_seed = record["seed"] if seed is None else seed
    return sample_pixel_labels(
        image_shape=record["image_shape"],
        changed_flat_indices=record["changed_flat_indices"],
        sample_count=sample_count,
        seed=audit_seed,
        distribution=distribution,
        distribution_params=distribution_params,
        sampling_image=sampling_image,
    )


def resolve_probe_seed(record: dict, seed: int | None) -> int:
    return record["seed"] if seed is None else seed


def summarize_record(record: dict) -> dict:
    return {
        "record_path": str(record["record_path"]),
        "cover_relative_path": record["cover_relative_path"],
        "stego_relative_path": record["stego_relative_path"],
        "method": record["method"],
        "alpha": record["alpha"],
        "seed": record["seed"],
        "height": record["image_shape"][0],
        "width": record["image_shape"][1],
        "used_pixels": record["used_pixels"],
        "used_pixel_fraction": record["used_pixel_fraction"],
        "used_pixel_percentage": record["used_pixel_percentage"],
        "num_changed": record["num_changed"],
        "change_rate": record["used_pixel_fraction"],
        "first_changed_coords": record["changed_coords"][:10].tolist(),
    }


def summarize_dataset(records: list[dict]) -> dict:
    total_pixels = sum(record["image_shape"][0] * record["image_shape"][1] for record in records)
    total_changed = sum(record["num_changed"] for record in records)
    return {
        "num_records": len(records),
        "total_changed": total_changed,
        "overall_change_rate": (total_changed / total_pixels) if total_pixels else 0.0,
        "records": [summarize_record(record) for record in records],
    }


def write_probe_csv(save_path: Path, samples: np.ndarray) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        save_path,
        samples.astype(np.int64, copy=False),
        fmt="%d",
        delimiter=",",
        header="row,col,was_embedded",
        comments="",
    )


def write_probe_summary_csv(save_path: Path, summaries: list[dict]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_name",
        "record_path",
        "cover_relative_path",
        "stego_relative_path",
        "method",
        "alpha",
        "record_seed",
        "probe_seed",
        "distribution",
        "image_source",
        "probed_pixels",
        "guessed_pixels",
        "guessed_fraction",
        "guessed_percentage",
        "used_pixels",
        "used_pixel_fraction",
        "used_pixel_percentage",
        "recall_at_budget",
    ]
    with save_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


def default_probe_path(record: dict) -> Path:
    return record["record_path"].with_name(
        record["record_path"].name.replace("_changes.npz", "_sampled_pixels.csv")
    )


def summarize_probe(
    record: dict,
    samples: np.ndarray,
    used_seed: int,
    distribution: str,
    image_source: str | None = None,
) -> dict:
    probed_pixels = int(samples.shape[0])
    guessed_count = int(samples[:, 2].sum()) if probed_pixels else 0
    used_pixels = int(record["used_pixels"])
    summary = {
        "record_name": record["record_path"].name,
        "record_path": str(record["record_path"]),
        "cover_relative_path": record["cover_relative_path"],
        "stego_relative_path": record["stego_relative_path"],
        "method": record["method"],
        "alpha": record["alpha"],
        "record_seed": int(record["seed"]),
        "used_seed": int(used_seed),
        "distribution": distribution,
        "probed_pixels": probed_pixels,
        "guessed_pixels": guessed_count,
        "guessed_fraction": guessed_count / probed_pixels if probed_pixels else 0.0,
        "guessed_percentage": 100.0 * guessed_count / probed_pixels if probed_pixels else 0.0,
        "used_pixels": used_pixels,
        "used_pixel_fraction": record["used_pixel_fraction"],
        "used_pixel_percentage": record["used_pixel_percentage"],
        "recall_at_budget": guessed_count / used_pixels if used_pixels else 0.0,
    }
    if image_source is not None:
        summary["image_source"] = image_source
    else:
        summary["image_source"] = ""
    summary["probe_seed"] = summary.pop("used_seed")
    return summary


def resolve_probe_image_path(
    record: dict,
    input_base: str | Path,
    image_source: str,
    cover_root: str | Path | None = None,
) -> Path:
    input_base = Path(input_base).expanduser().resolve()
    if image_source == "stego":
        return (input_base / record["stego_relative_path"]).resolve()
    if image_source == "cover":
        if cover_root is None:
            raise ValueError("cover_root must be provided when probe_image_source='cover'.")
        return (Path(cover_root).expanduser().resolve() / record["cover_relative_path"]).resolve()
    raise ValueError(
        f"Unsupported probe image source '{image_source}'. Supported: {sorted(SUPPORTED_PROBE_IMAGE_SOURCES)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe saved embedding outputs by sampling pixels from change records."
    )
    parser.add_argument(
        "input_path",
        help="Path to a *_changes.npz record or a directory containing change records.",
    )
    parser.add_argument("--cover-image", help="Optional path to the original cover image.")
    parser.add_argument("--stego-image", help="Optional path to the corresponding stego image.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Number of random pixels to probe per record.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Overrides the seed used for random pixel auditing.",
    )
    parser.add_argument(
        "--save-samples",
        help="Optional path for a merged CSV when probing multiple records, or an explicit CSV path for one record.",
    )
    parser.add_argument(
        "--save-summary",
        help="Optional CSV path for per-record probing results.",
    )
    parser.add_argument(
        "--probe-distribution",
        default="uniform",
        choices=sorted(SUPPORTED_PROBE_DISTRIBUTIONS),
        help="Sampling distribution used for probing.",
    )
    parser.add_argument(
        "--probe-image-source",
        default="stego",
        choices=sorted(SUPPORTED_PROBE_IMAGE_SOURCES),
        help="Image source used by image-adaptive probing distributions.",
    )
    parser.add_argument(
        "--cover-root",
        help="Root directory for original cover images when probe-image-source is 'cover'.",
    )
    return parser


def run_probe_cli(args: argparse.Namespace) -> None:
    run_probe(
        input_path=args.input_path,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
        probe_distribution=args.probe_distribution,
        save_samples=args.save_samples,
        save_summary=args.save_summary,
        probe_image_source=args.probe_image_source,
        cover_root=args.cover_root,
        cover_image=args.cover_image,
        stego_image=args.stego_image,
    )


def run_probe(
    input_path: str | Path,
    sample_count: int = 100,
    sample_seed: int | None = None,
    probe_distribution: str = "uniform",
    probe_distribution_params: dict | None = None,
    save_samples: str | Path | None = None,
    save_summary: str | Path | None = None,
    probe_image_source: str = "stego",
    cover_root: str | Path | None = None,
    cover_image: str | Path | None = None,
    stego_image: str | Path | None = None,
) -> None:
    if probe_distribution not in SUPPORTED_PROBE_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported probe distribution '{probe_distribution}'. Supported: {sorted(SUPPORTED_PROBE_DISTRIBUTIONS)}"
        )
    if probe_image_source not in SUPPORTED_PROBE_IMAGE_SOURCES:
        raise ValueError(
            f"Unsupported probe image source '{probe_image_source}'. Supported: {sorted(SUPPORTED_PROBE_IMAGE_SOURCES)}"
        )

    input_path = Path(input_path).expanduser().resolve()
    input_base = input_path if input_path.is_dir() else input_path.parent
    record_paths = iter_record_paths(input_path)
    records = [load_change_record(record_path) for record_path in record_paths]

    if len(records) == 1 and cover_image and stego_image:
        change_mask = reconstruct_change_mask(
            image_shape=records[0]["image_shape"],
            changed_flat_indices=records[0]["changed_flat_indices"],
        )
        comparison = compare_cover_and_stego(
            cover_path=cover_image,
            stego_path=stego_image,
            change_mask=change_mask,
        )
        print(json.dumps(comparison, indent=2))

    all_samples: list[tuple[str, int, int, int]] = []
    all_summaries: list[dict] = []
    for record in records:
        used_seed = resolve_probe_seed(record, sample_seed)
        sampling_image = None
        if probe_distribution in IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS:
            sampling_image_path = resolve_probe_image_path(
                record=record,
                input_base=input_base,
                image_source=probe_image_source,
                cover_root=cover_root,
            )
            sampling_image = to_u8_array(sampling_image_path)
        samples = audit_samples(
            record,
            sample_count=sample_count,
            seed=used_seed,
            distribution=probe_distribution,
            distribution_params=probe_distribution_params,
            sampling_image=sampling_image,
        )
        summary = summarize_probe(
            record,
            samples=samples,
            used_seed=used_seed,
            distribution=probe_distribution,
            image_source=(
                probe_image_source
                if probe_distribution in IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS
                else None
            ),
        )
        all_summaries.append(summary)
        print(
            json.dumps(
                summary,
                indent=2,
            )
        )
        if save_samples:
            for row, col, label in samples:
                all_samples.append(
                    (
                        record["record_path"].name,
                        int(row),
                        int(col),
                        int(label),
                    )
                )

    if save_summary:
        save_path = Path(save_summary).expanduser().resolve()
        write_probe_summary_csv(save_path, all_summaries)

    if save_samples:
        save_path = Path(save_samples).expanduser().resolve()
        if len(records) == 1:
            single_samples = np.array(
                [[row, col, label] for _, row, col, label in all_samples],
                dtype=np.int64,
            )
            write_probe_csv(save_path, single_samples)
        else:
            with save_path.open("w", encoding="utf-8") as handle:
                handle.write("record,row,col,was_embedded\n")
                for record_name, row, col, label in all_samples:
                    handle.write(f"{record_name},{row},{col},{label}\n")


if __name__ == "__main__":
    run_probe_cli(build_parser().parse_args())
