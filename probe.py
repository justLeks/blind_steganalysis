from pathlib import Path

from read_changes import run_probe


INPUT_PATH = Path("output")
SAMPLE_COUNT = 10000
SAMPLE_SEED = 41
# Supported values: uniform, center_gaussian, center_laplace, edge_gaussian
PROBE_DISTRIBUTION = "edge_gaussian"
PROBE_DISTRIBUTION_PARAMS = {}
SAVE_SAMPLES = None
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
        cover_image=COMPARE_COVER_IMAGE,
        stego_image=COMPARE_STEGO_IMAGE,
    )
