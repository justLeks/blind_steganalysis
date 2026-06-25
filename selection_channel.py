"""Oracle selection-channel localizer.

The *oracle* ranks pixels by their TRUE per-pixel change probability under the embedding algorithm --
the best any selection-channel-aware method could do. It is the upper bracket (ceiling) for the
estimator ladder; the random baseline is the lower bracket (floor).

This reproduces the exact probability map that ``conseal`` used to draw the recorded carriers
(``conseal.simulate._ternary.probability`` for HUGO, ``conseal.mipod.probability`` for MiPOD), so the
ceiling is exact rather than a cost proxy. Because the recorded carriers are a Bernoulli draw with these
probabilities, the identity ``E[#changes] = sum(p_+1 + p_-1)`` provides a built-in correctness check
(see :func:`expected_changes`).
"""

from __future__ import annotations

import warnings

import numpy as np

from embedding import prepare_conseal_import


SUPPORTED_METHODS = {"HUGO", "MIPOD"}


def change_probability_map(cover_u8: np.ndarray, method: str, alpha: float) -> np.ndarray:
    """Total per-pixel change probability ``p_+1 + p_-1`` (2-D, same shape as the cover).

    The cover must be the uint8 array that was embedded (``embedding.to_u8_array`` of the cover TIFF),
    so the costs/probabilities match the carrier-generating run exactly.
    """
    prepare_conseal_import()
    import conseal as cl
    from conseal.simulate import _ternary

    normalized = method.upper()
    if normalized not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method '{method}'. Supported: {sorted(SUPPORTED_METHODS)}")

    cover = np.asarray(cover_u8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MiPOD warns on flat-variance clipping; expected and harmless.
        if normalized == "HUGO":
            rhos = cl.hugo.compute_cost_adjusted(cover)  # (rho_p1, rho_m1) with simulator defaults
            ps, _ = _ternary.probability(rhos=rhos, alpha=alpha, n=cover.size)  # type: ignore[arg-type, misc]
        else:  # MIPOD
            ps, _ = cl.mipod.probability(cover, alpha)

    p_p1, p_m1 = ps  # type: ignore[misc]  # conseal stubs mis-type the (p_+1, p_-1) pair
    return (np.asarray(p_p1, dtype=np.float64) + np.asarray(p_m1, dtype=np.float64))


def oracle_scores(cover_u8: np.ndarray, method: str, alpha: float) -> np.ndarray:
    """Flat (1-D) oracle score vector; higher = more likely a carrier."""
    return change_probability_map(cover_u8, method, alpha).reshape(-1)


def expected_changes(cover_u8: np.ndarray, method: str, alpha: float) -> float:
    """E[#changed pixels] = sum of the per-pixel total change probability. Used to validate the map."""
    return float(change_probability_map(cover_u8, method, alpha).sum())
