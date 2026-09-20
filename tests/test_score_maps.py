"""Score-map registry tests.

Part 1 — frozen-legacy guards: verbatim copies of the pre-registry
``build_texture_energy_probabilities`` / ``build_laplacian_residual_probabilities``
bodies, asserted bit-exact (``np.array_equal``, not allclose) against the public
functions. These tests were written against the pre-refactor code and must stay
green through any refactor of ``read_changes``.

Run: python3 -m unittest discover -s tests -v
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from read_changes import (
    DB8_HPDF,
    DB8_LPDF,
    IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS,
    SCORE_MAP_BUILDERS,
    SUPPORTED_PROBE_DISTRIBUTIONS,
    build_adaptive_probabilities,
    build_laplacian_residual_probabilities,
    build_sampling_probabilities,
    build_score_map,
    build_texture_energy_probabilities,
    resolve_score_map_params,
)


COVER_DIR = pathlib.Path(__file__).resolve().parent.parent / "ALASKA_v2_TIFF_512_GrayScale_50"

PARAM_GRID = [
    {"gamma": 1.0, "floor": 1e-6},
    {"gamma": 2.0, "floor": 1e-6},
    {"gamma": 1.0, "floor": 0.5},
    {"gamma": 2.0, "floor": 0.5},
]


def _test_images():
    images = []
    for seed, shape in [(0, (24, 24)), (1, (24, 24)), (2, (33, 17))]:
        rng = np.random.default_rng(seed)
        images.append(rng.integers(0, 256, size=shape, dtype=np.uint8))
    images.append(np.full((24, 24), 7, dtype=np.uint8))  # constant image
    return images


# --- frozen verbatim copies of the pre-registry implementations -----------------


def _legacy_positive(params, key, fallback):
    value = float((params or {}).get(key, fallback))
    if value <= 0:
        raise ValueError(f"Expected '{key}' to be positive, got {value}.")
    return value


def _legacy_nonnegative(params, key, fallback):
    value = float((params or {}).get(key, fallback))
    if value < 0:
        raise ValueError(f"Expected '{key}' to be non-negative, got {value}.")
    return value


def _legacy_texture_energy_probabilities(sampling_image, distribution_params=None):
    params = distribution_params or {}
    gamma = _legacy_positive(params, "gamma", 1.0)
    floor = _legacy_nonnegative(params, "floor", 1e-6)

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


def _legacy_laplacian_residual_probabilities(sampling_image, distribution_params=None):
    params = distribution_params or {}
    gamma = _legacy_positive(params, "gamma", 1.0)
    floor = _legacy_nonnegative(params, "floor", 1e-6)

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


class TestLegacyBitExact(unittest.TestCase):
    """The registry refactor must not change a single bit of the published maps."""

    def test_texture_energy_bit_exact(self):
        for img in _test_images():
            for params in PARAM_GRID:
                expected = _legacy_texture_energy_probabilities(img, params)
                via_public = build_texture_energy_probabilities(img, params)
                via_dispatch = build_sampling_probabilities(
                    img.shape, "texture_energy", params, sampling_image=img
                )
                assert via_dispatch is not None  # adaptive distributions never return None
                self.assertTrue(np.array_equal(expected, via_public))
                self.assertTrue(np.array_equal(expected, via_dispatch))

    def test_laplacian_residual_bit_exact(self):
        for img in _test_images():
            for params in PARAM_GRID:
                expected = _legacy_laplacian_residual_probabilities(img, params)
                via_public = build_laplacian_residual_probabilities(img, params)
                via_dispatch = build_sampling_probabilities(
                    img.shape, "laplacian_residual", params, sampling_image=img
                )
                assert via_dispatch is not None  # adaptive distributions never return None
                self.assertTrue(np.array_equal(expected, via_public))
                self.assertTrue(np.array_equal(expected, via_dispatch))

    @unittest.skipUnless(COVER_DIR.exists(), "ALASKA working subset not present")
    def test_bit_exact_on_real_cover(self):
        from embedding import to_u8_array

        cover = to_u8_array(sorted(COVER_DIR.glob("*.tif"))[0])
        params = {"gamma": 1.0, "floor": 1e-6}
        self.assertTrue(
            np.array_equal(
                _legacy_texture_energy_probabilities(cover, params),
                build_texture_energy_probabilities(cover, params),
            )
        )
        self.assertTrue(
            np.array_equal(
                _legacy_laplacian_residual_probabilities(cover, params),
                build_laplacian_residual_probabilities(cover, params),
            )
        )


class TestRegistry(unittest.TestCase):
    def test_registry_complete(self):
        expected = {"texture_energy", "gradient_magnitude", "laplacian_residual", "local_variance",
                    "local_entropy", "wavelet_energy", "srm_residual"}
        self.assertEqual(set(SCORE_MAP_BUILDERS), expected)
        self.assertEqual(IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS, frozenset(SCORE_MAP_BUILDERS))
        self.assertTrue(IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS <= SUPPORTED_PROBE_DISTRIBUTIONS)

    def test_resolve_params_defaults_and_overrides(self):
        params = resolve_score_map_params("local_variance", None)
        self.assertEqual(params, {"gamma": 1.0, "floor": 1e-6, "window": 9})
        params = resolve_score_map_params("local_variance", {"window": 5, "gamma": 2.0})
        self.assertEqual(params, {"gamma": 2.0, "floor": 1e-6, "window": 5})
        self.assertEqual(resolve_score_map_params("wavelet_energy", None), {"gamma": 1.0, "floor": 1e-6})
        with self.assertRaises(ValueError):
            resolve_score_map_params("uniform", None)  # not image-adaptive

    def test_local_variance_window_validation(self):
        img = _test_images()[0]
        for bad_window in (0, -3, 2, 4):
            with self.assertRaises(ValueError):
                build_score_map("local_variance", img, {"window": bad_window})
        small = build_score_map("local_variance", img, {"window": 3})
        large = build_score_map("local_variance", img, {"window": 9})
        self.assertFalse(np.array_equal(small, large))


class TestScoreMapProperties(unittest.TestCase):
    def test_probabilities_properties_all_adaptive(self):
        img = _test_images()[0]
        for dist in sorted(IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS):
            probs = build_adaptive_probabilities(dist, img)
            self.assertEqual(probs.shape, (img.size,))
            self.assertEqual(probs.dtype, np.float64)
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=10)
            self.assertTrue(np.all(probs >= 0))
            self.assertTrue(np.array_equal(probs, build_adaptive_probabilities(dist, img)))  # deterministic

    def test_raw_maps_nonnegative_and_2d(self):
        img = _test_images()[0]
        for dist in sorted(IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS):
            energy = build_score_map(dist, img)
            self.assertEqual(energy.shape, img.shape)
            self.assertTrue(np.all(energy >= 0), dist)

    def test_constant_image_uniform_fallback(self):
        img = np.full((24, 24), 42, dtype=np.uint8)
        for dist in sorted(IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS):
            energy = build_score_map(dist, img)
            np.testing.assert_allclose(energy, 0.0, atol=1e-9, err_msg=dist)
            probs = build_adaptive_probabilities(dist, img)  # floor rescues to uniform
            np.testing.assert_allclose(probs, 1.0 / img.size, rtol=1e-6, err_msg=dist)
            if dist == "wavelet_energy":
                # Irrational Daubechies taps leave ~1e-15 float residue on a
                # constant image, so the floor=0 zero-weight guard cannot fire.
                continue
            with self.assertRaises(ValueError, msg=dist):
                build_adaptive_probabilities(dist, img, {"floor": 0.0})

    def test_texture_beats_flat_all_maps(self):
        # Left half flat, right half noisy -> the noisy half must carry more mass.
        shape = (32, 32)
        rng = np.random.default_rng(1)
        img = np.zeros(shape, dtype=np.uint8)
        img[:, shape[1] // 2:] = rng.integers(0, 256, size=(shape[0], shape[1] // 2), dtype=np.uint8)
        for dist in sorted(IMAGE_ADAPTIVE_PROBE_DISTRIBUTIONS):
            grid = build_adaptive_probabilities(dist, img).reshape(shape)
            self.assertGreater(
                grid[:, shape[1] // 2:].sum(), grid[:, : shape[1] // 2].sum(), dist
            )


class TestParseLocalizer(unittest.TestCase):
    def test_parse_localizer(self):
        from run_localization_benchmark import LOCALIZER_ALIASES, parse_localizer

        self.assertEqual(parse_localizer("texture_energy_stego"), ("texture_energy", "stego"))
        self.assertEqual(parse_localizer("local_variance_cover"), ("local_variance", "cover"))
        self.assertEqual(parse_localizer("srm_residual_stego"), ("srm_residual", "stego"))
        self.assertEqual(LOCALIZER_ALIASES["texture_stego"], "texture_energy_stego")
        self.assertEqual(LOCALIZER_ALIASES["texture_cover"], "texture_energy_cover")
        for bad in ("uniform", "oracle", "sobel_stego", "texture_energy_banana", "texture_energy", ""):
            with self.assertRaises(ValueError, msg=bad):
                parse_localizer(bad)


class TestWaveletFilters(unittest.TestCase):
    def test_highpass_kills_dc(self):
        self.assertAlmostEqual(float(DB8_HPDF.sum()), 0.0, places=9)

    def test_qmf_relation(self):
        signs = (-1.0) ** np.arange(DB8_HPDF.size)
        np.testing.assert_allclose(DB8_LPDF, signs * DB8_HPDF[::-1])
        self.assertAlmostEqual(float(DB8_LPDF.sum()), float(np.sqrt(2.0)), places=9)

    def test_energy_positive_on_noise(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
        self.assertGreater(float(build_score_map("wavelet_energy", img).sum()), 0.0)


class TestSrmEdgeLocalization(unittest.TestCase):
    def test_step_edge_peaks_at_edge(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        edge_col = 16
        img[:, edge_col:] = 200
        energy = build_score_map("srm_residual", img)
        peak_col = int(np.argmax(energy.sum(axis=0)))
        self.assertLessEqual(abs(peak_col - edge_col), 2)


if __name__ == "__main__":
    unittest.main()
