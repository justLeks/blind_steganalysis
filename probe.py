from pathlib import Path

from read_changes import run_probe


# Path to a single *_changes.npz file or to a directory like "output".
INPUT_PATH = Path("output")
SAMPLE_COUNT = 10000
SAMPLE_SEED = 41

# Supported values:
# uniform, center_gaussian, center_laplace, edge_gaussian, texture_energy
PROBE_DISTRIBUTION = "texture_energy"

# For texture_energy:
# - gamma > 1.0 makes probing more concentrated on textured regions.
# - floor keeps flat regions probeable.
PROBE_DISTRIBUTION_PARAMS = {
    "gamma": 1.0,
    "floor": 1e-6,
}

# Used only by image-adaptive distributions such as texture_energy.
# "stego" works out of the box with generated output/.
# "cover" requires COVER_ROOT to point to the original cover-image directory.
PROBE_IMAGE_SOURCE = "stego"
COVER_ROOT = Path("ALASKA_v2_TIFF_512_GrayScale_50")

SAVE_SAMPLES = None
SAVE_SUMMARY = Path("output/probe_results.csv")
COMPARE_COVER_IMAGE = None
COMPARE_STEGO_IMAGE = None


if __name__ == "__main__":
    run_probe(
        input_path=INPUT_PATH,
        sample_count=SAMPLE_COUNT,
        sample_seed=SAMPLE_SEED,
        probe_distribution=PROBE_DISTRIBUTION,
        probe_distribution_params=PROBE_DISTRIBUTION_PARAMS,
        save_samples=SAVE_SAMPLES,
        save_summary=SAVE_SUMMARY,
        probe_image_source=PROBE_IMAGE_SOURCE,
        cover_root=COVER_ROOT,
        cover_image=COMPARE_COVER_IMAGE,
        stego_image=COMPARE_STEGO_IMAGE,
    )
