"""Budget-free and budgeted localization metrics for carrier-pixel probing.

A *localizer* produces a per-pixel score (higher = more likely a carrier). We evaluate that score
against the binary carrier mask. Two complementary views:

- **Budget-free ranking quality** -- Average Precision (area under the precision-recall curve). Carriers
  are a 0.4-10% minority, so PR-AUC/AP is the honest primary metric (ROC-AUC flatters under imbalance).
- **Budgeted operating points** -- recall@budget (deterministic top-B by score) and
  lift@budget = recall / (B/N). Random sampling has E[recall] = B/N exactly (hypergeometric), so lift
  is the apples-to-apples "how many times better than chance" number; foreground small budgets.

Score-as-ranker framing: we take the top-B highest-scoring pixels (deterministic), which is the natural,
parameter-free way to evaluate a probability/saliency map. This is the upper envelope of the gamma-weighted
stochastic sampling the original method uses, and removes gamma from the evaluation.

numpy-only (no sklearn dependency).
"""

from __future__ import annotations

import numpy as np


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation), matching
    sklearn.metrics.average_precision_score. Ties are collapsed into one threshold group.
    """
    labels = np.asarray(labels).astype(bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(scores, kind="stable")[::-1]
    y = labels[order]
    s = scores[order]

    tp = np.cumsum(y)
    fp = np.cumsum(~y)

    # Keep only the last index of each tie group (entries with equal score share one operating point).
    keep = np.r_[np.diff(s) != 0, True]
    tp_d = tp[keep]
    fp_d = fp[keep]

    precision = tp_d / np.maximum(tp_d + fp_d, 1)
    recall = tp_d / n_pos
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U / rank-sum identity. Reported as a secondary metric only."""
    labels = np.asarray(labels).astype(bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks 1..N with ties assigned the average rank (needed for a correct AUC under ties)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = values.size
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)  # average of ranks (i+1)..j
        i = j
    return ranks


def budget_pixels(total_pixels: int, fraction: float) -> int:
    return min(total_pixels, max(1, int(round(total_pixels * float(fraction)))))


def recall_at_budgets(
    scores: np.ndarray,
    labels: np.ndarray,
    fractions: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Deterministic top-B recall for each budget fraction.

    Ties are broken randomly (via ``rng``) so that flat score maps (e.g. uniform) are scored fairly
    rather than by pixel index. Returns recall values aligned with ``fractions``.
    """
    labels = np.asarray(labels).astype(bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    total = scores.size
    n_pos = int(labels.sum())
    if rng is None:
        rng = np.random.default_rng()

    # Sort by score descending; random secondary key breaks ties without index bias.
    order = np.lexsort((rng.random(total), -scores))
    labels_sorted = labels[order]
    carriers_cumsum = np.cumsum(labels_sorted)

    recalls = np.empty(len(fractions), dtype=np.float64)
    for k, fraction in enumerate(fractions):
        b = budget_pixels(total, float(fraction))
        hits = int(carriers_cumsum[b - 1])
        recalls[k] = hits / n_pos if n_pos else 0.0
    return recalls


def lift_at_budgets(recalls: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """lift = recall / (B/N); B/N approximated by the budget fraction (exact baseline expectation)."""
    fractions = np.asarray(fractions, dtype=np.float64)
    return np.asarray(recalls, dtype=np.float64) / np.maximum(fractions, 1e-12)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean across samples (e.g. per-image metric values).

    Returns (mean, lo, hi). NaNs are dropped (images with no carriers cannot define AP).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, (100.0 - ci) / 2.0))
    hi = float(np.percentile(boot_means, 100.0 - (100.0 - ci) / 2.0))
    return float(values.mean()), lo, hi


def _self_test() -> None:
    # AP on a tiny worked example: scores .9 .8 .7 .6, labels 1 0 1 0
    # ranking: TP,FP,TP,FP -> precisions at recall steps: 1/1 and 2/3 -> AP = 0.5*1 + 0.5*(2/3)
    ap = average_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0]))
    assert abs(ap - (0.5 * 1.0 + 0.5 * (2.0 / 3.0))) < 1e-12, ap
    # Perfect ranking -> AP 1.0; reversed -> low.
    assert abs(average_precision(np.array([4, 3, 2, 1]), np.array([1, 1, 0, 0])) - 1.0) < 1e-12
    # ROC-AUC: perfect separation -> 1.0
    assert abs(roc_auc(np.array([4, 3, 2, 1]), np.array([1, 1, 0, 0])) - 1.0) < 1e-12
    # Random scores: recall@fraction ~ fraction in expectation; lift ~ 1.
    rng = np.random.default_rng(0)
    n = 100_000
    labels = rng.random(n) < 0.05
    scores = rng.random(n)
    recalls = recall_at_budgets(scores, labels, np.array([0.1, 0.5]), rng=rng)
    assert abs(recalls[0] - 0.1) < 0.02 and abs(recalls[1] - 0.5) < 0.02, recalls
    # AP of random scores ~ prevalence (0.05)
    assert abs(average_precision(scores, labels) - 0.05) < 0.01
    print("localization_metrics self-test OK")


if __name__ == "__main__":
    _self_test()
