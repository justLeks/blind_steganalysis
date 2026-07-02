# Findings — Probing Score-Map Comparison (Milestone 3a)

> Run date: 2026-07-01/02. Setup: n = 100 ALASKA v2 grayscale 512×512 covers; HUGO + MiPOD ×
> α ∈ {0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5} (14 configs, the canonical steganograms of
> `experiments/probing_only`, embed seed 12345); benchmark = 12 localizers (uniform floor, conseal
> oracle ceiling, and {texture_energy, laplacian_residual, local_variance, wavelet_energy,
> srm_residual} × {stego, cover}), probe/benchmark seed 12345, 2000-resample bootstrap CIs.
> Sources: `experiments/distribution_benchmark/` (ranking), `experiments/probing_<dist>/` (γ=1
> sampling), figures + comparison CSVs in this directory. Unless stated otherwise, "AP efficiency"
> is the ratio of config means `(mean AP − floor AP) / (oracle AP − floor AP)` from
> `comparison_ranking.csv`; the `fig_ap_efficiency_*` figures show the per-image mean of ratios with
> CIs instead (skewed right at low α, hence higher — both are reported deliberately).

## Result 1 — HUGO: nothing beats texture energy; the fixed SRM bank ties it

AP efficiency (stego side), HUGO:

| map | α=0.03 | α=0.1 | α=0.3 | α=0.5 |
|---|---|---|---|---|
| **texture_energy** | **0.59** | **0.68** | **0.78** | **0.84** |
| srm_residual | 0.59 | 0.67 | 0.77 | 0.82 |
| laplacian_residual | 0.45 | 0.53 | 0.64 | 0.70 |
| local_variance | 0.31 | 0.40 | 0.52 | 0.58 |
| wavelet_energy | 0.09 | 0.15 | 0.23 | 0.29 |

At α=0.1 the two leaders are statistically indistinguishable: texture AP = 0.122 [0.109, 0.137] vs
SRM AP = 0.122 [0.108, 0.135] (oracle 0.173, floor 0.016). Lift@1% reaches ~12× (oracle ~15×).
The published texture-energy map remains the right default for HUGO, and the (much heavier) fixed
SRM bank buys nothing over it.

## Result 2 — MiPOD: the gap is NOT closed; the local-variance hypothesis is refuted

The M1 question was whether a variance-based map (matching MiPOD's variance-driven cost model) closes
the 34–47 % efficiency gap. **It does not — local variance is the *worst* map on MiPOD:**

| map | α=0.03 | α=0.1 | α=0.3 | α=0.5 |
|---|---|---|---|---|
| laplacian_residual | **0.35** | **0.36** | 0.41 | 0.46 |
| texture_energy | 0.34 | 0.36 | 0.42 | 0.47 |
| srm_residual | 0.33 | 0.36 | **0.43** | **0.49** |
| wavelet_energy | 0.20 | 0.24 | 0.31 | 0.36 |
| local_variance | 0.12 | 0.15 | 0.21 | 0.26 |

At α=0.1: texture/Laplacian/SRM all at AP ≈ 0.031 (CIs overlap completely) vs oracle 0.0625 —
i.e. the three high-pass maps are tied at ~36 % efficiency and no hand-crafted map in this family
does better. A plain box-window variance is *too smooth*: it ranks whole textured regions, while
MiPOD's channel concentrates on locally unpredictable pixels within them. Closing the MiPOD gap
apparently needs the modeled residual-variance estimate itself (Wiener-residual + local fit) or a
learned predictor — the deferred M3 rungs, now with direct evidence motivating them.

## Result 3 — Ablation: only the high-pass maps carry any embedding-specific signal

Relative AP gain of stego-side over cover-side scoring (mean over α; `fig_cover_vs_stego_*`):

| map | HUGO | MiPOD |
|---|---|---|
| laplacian_residual | +10.6 % | +14.1 % |
| texture_energy | +7.7 % | +11.1 % |
| srm_residual | +6.9 % | +9.5 % |
| wavelet_energy | +0.3 % | +0.3 % |
| **local_variance** | **+0.04 %** | **+0.07 %** |

The ±1 embedding changes perturb short-support high-pass residuals directly (the 3×3 Laplacian most
of all), giving a real but small stego-specific bump — consistent with M1's ~5–10 % estimate. The
9×9 variance window and 16-tap wavelet support average the ±1 changes away entirely: those two maps
are **pure content predictors**. The scope statement stands: this is selection-channel
*localization*, not stego *detection*.

## Result 4 — The two evaluation frames disagree; trust the ranking frame

In the γ=1 stochastic-sampling frame (`comparison_sampling.csv`, recall@B=5 %, α=0.1) local variance
*looks* best on HUGO — 0.159 vs texture 0.139 — yet its top-B/AP ranking is far worse (Result 1).
γ=1 sampling conflates *ordering quality* with *probability-mass concentration*: squared-scale
variance energies concentrate mass on a few blocks, which helps proportional sampling but not the
ordering. This re-confirms M1's recommendation with a sharper example: **evaluate maps by
deterministic top-B / AP; treat γ as a tuned execution parameter, not part of the method's quality.**

## Result 5 — Controls behaved exactly as theory requires

The empirical uniform run deviates from E[recall] = B/N by at most 0.0039 across all 98
(method, α, B) cells (probing run), and the analytic floor reproduces AP = prevalence, lift = 1.0 in
the benchmark; the oracle ceiling dominates every map at every (method, α, B). The brackets are
trustworthy.

## What this changes for the draft

1. Report texture energy as the HUGO method of choice **with the SRM-bank tie as evidence the
   result is not an artifact of one filter choice**; both sit at 59–84 % of the oracle.
2. State the MiPOD limitation with the new evidence: no map in this hand-crafted family exceeds
   ~36 % efficiency at small payloads; a plain variance map specifically does not work. This is the
   strongest argument yet for the learned selection-channel estimator (M3).
3. Use the ablation table when defending "blind": localization is content-driven; only short-support
   high-pass maps see the embedding at all, and only by ~7–14 % relative.
4. Cite figures from `figures/` (grayscale-safe) and per-image panels from `panels/`.

## Caveats

- Border handling differs by construction: legacy maps keep zeroed borders (bit-exactness with the
  published run), new maps use reflect padding. A zeroed 8-px frame would already exceed the B=1 %
  budget, so this is documented rather than "fixed".
- `local_variance` was run at a single window (9); smaller/larger windows were not swept. Wavelet is
  the S-UNIWARD 16-tap Daubechies-8 bank; WOW's aggregation (Hölder-mean of directional residuals) was
  not replicated.
- Two efficiency estimators are shown (ratio of means in tables/CSV; per-image mean of ratios ± CI in
  figures); at α ≤ 0.05 the per-image ratio is heavily right-skewed — quote the tables for headline
  numbers.
- Benchmark recall/lift columns share one tie-break RNG stream (reproduce only under the identical
  localizer list/limit); AP and ROC-AUC are RNG-free. n = 100; 10K-scale confirmation remains open.

## Reproduce

```bash
for d in uniform laplacian_residual local_variance wavelet_energy srm_residual; do
  python3 run_probing_experiment.py --distribution $d \
    --experiment-root experiments/probing_$d \
    --steganogram-root experiments/probing_only/steganograms
done
python3 run_localization_benchmark.py       # → experiments/distribution_benchmark (~1 h)
python3 make_comparison_figures.py          # → this directory
```
