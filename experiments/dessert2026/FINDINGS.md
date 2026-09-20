# Findings — DESSERT-2026 revision campaign (n = 1000)

> Run: 2026-09-20 21:11 → 2026-09-21 02:53 EEST (5 h 42 min, 8 workers). Code: tag `dessert2026-rev1` on branch `dessert2026-revision`.
> Every number below is copied from `paper/*.md`, `benchmark/summary.csv`, `paired_tests/paired_tests.csv`,
> `probing_texture_energy/probe_summary.csv` or `comparison/figures/fig4_*.csv`; regenerate with
> `python3 make_paper_tables.py --root experiments/dessert2026` and `python -m tools.number_map`
> (paper repo) after any re-run.

## Setup

- **Covers:** 1000 of the 10 000 ALASKA v2 grayscale 512×512 TIFFs, drawn by
  `select_cover_subset.py --n 1000 --seed 12345` → `data/subsets/alaska10k_n1000_seed12345.txt`
  (16-bit → 8-bit by integer division by 257; no cropping).
- **Embedders:** HUGO, MiPOD, S-UNIWARD (conseal 2025.11 simulators, default parameters; per-image
  SHA-256 seeds from run seed 12345). Ground truth = cover–stego difference.
- **Payloads α (bpp):** 0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5.
- **Budgets B:** 0.5, 1, 2, 5, 10, 20, 30, 40, 50 %.
- **Localizers (16):** `uniform` (analytic floor), `oracle` (exact conseal change probabilities from the
  cover), and {texture_energy, gradient_magnitude, laplacian_residual, local_variance (9×9),
  local_entropy (9×9, 32 bins), wavelet_energy (db8, 1 level), srm_residual} × {stego, cover}.
  Ranking rule = deterministic top-b with random tie-break (per-record seed).
- **Sampling rule:** γ = 1, ε = 1e-6, K = 10 draws per image (`--repeats 10`), texture_energy and uniform.
- **Statistics:** 95 % percentile-bootstrap CIs (2000 resamples, seed 0); paired two-sided Wilcoxon
  signed-rank tests of texture_energy vs each baseline on AP, R(1 %), R(5 %), R(10 %), nAURC(0.10),
  Holm-corrected within (method, α, metric).
- **Software:** Python 3.11, numpy 2.4.2, scipy 1.17.1, conseal 2025.11, numba 0.64.0; 8 workers.

## Result 1 — lift over random at small budgets (Table II, Fig. 2)

Ranking rule, texture_energy, α = 0.01 bpp (n = 1000, 95 % CI):

| method | R(1 %) | L(1 %) | Q(1 %) | R(10 %) | L(10 %) | R(50 %) | L(50 %) | oracle R(1 %) |
|---|---|---|---|---|---|---|---|---|
| HUGO | 0.255 [0.248, 0.262] | 25.5 | 0.032 | 0.734 | 7.3 | 0.974 | 1.95 | 0.411 |
| MiPOD | 0.027 [0.026, 0.029] | 2.7 | 0.002 | 0.222 | 2.2 | 0.720 | 1.44 | 0.080 |
| S-UNIWARD | 0.088 [0.084, 0.092] | 8.8 | 0.009 | 0.388 | 3.9 | 0.824 | 1.65 | 0.187 |

nAURC(0.10): 0.516 / 0.117 / 0.238 vs uniform 0.050 and oracle 0.694 / 0.228 / 0.408. Lift decays toward 1
as B grows (recall saturates); the method earns its keep at B ≤ 10 %. This replaces the submitted
version's raw H(B) counts and its mis-transcribed "48 % / 51 % at B = 50 %" (the stored n = 100 run gave
0.877 / 0.662 at α = 0.03 with the sampling rule).

## Result 2 — baselines and the oracle gap (Table III, Fig. 3)

AP / η at α = 0.01 (stego side):

| map | HUGO | MiPOD | S-UNIWARD |
|---|---|---|---|
| texture_energy | 0.031 / 0.49 | 0.0019 / 0.27 | 0.0059 / 0.33 |
| srm_residual | 0.030 / 0.47 | 0.0020 / 0.29 | 0.0062 / 0.35 |
| laplacian_residual | 0.024 / 0.38 | 0.0020 / 0.28 | 0.0060 / 0.33 |
| gradient_magnitude | 0.027 / 0.43 | 0.0014 / 0.13 | 0.0033 / 0.16 |
| local_entropy | 0.016 / 0.25 | 0.0016 / 0.18 | 0.0044 / 0.23 |
| local_variance | 0.016 / 0.25 | 0.0013 / 0.12 | 0.0032 / 0.15 |
| wavelet_energy | 0.004 / 0.05 | 0.0017 / 0.21 | 0.0031 / 0.14 |
| oracle | 0.062 / 1 | 0.0049 / 1 | 0.0162 / 1 |

Same ordering at α = 0.005 (texture η 0.47 / 0.28 / 0.31) and 0.05 (0.59 / 0.31 / 0.44). η of texture
energy vs α: HUGO 0.46 → 0.81, S-UNIWARD 0.27 → 0.70, MiPOD 0.27–0.37 below 0.2 bpp (0.46 at 0.5).
Lift@1 % of texture energy vs α: HUGO 40.5 (0.001) → 3.8 (0.5); S-UNIWARD 13.3 → 2.4; MiPOD flat 2.8 → 2.1.
Oracle lift@1 % at 0.001: 66.8 / 8.1 / 32.3. MiPOD is the hard case for the whole family; local variance
(MiPOD's own model ingredient) is the worst map on MiPOD; wavelet energy (S-UNIWARD's bank) is a poor
predictor of S-UNIWARD (η 0.14).

## Result 3 — paired significance (Table IV)

Wilcoxon signed-rank, Holm-corrected, α = 0.01, n = 1000. Texture energy beats uniform, gradient magnitude,
local variance, local entropy and wavelet energy on AP and R(1 %) for all three embedders (p < 0.001;
median ΔR(1 %) on HUGO: +0.22, +0.013, +0.097, +0.085, +0.19). HUGO: texture energy also beats Laplacian
(ΔAP +0.0034, ΔR(1 %) +0.019) and SRM (ΔAP +0.0015, p = 3e-8; ΔR(1 %) +0.009, p = 8e-4). MiPOD and
S-UNIWARD: **not significant** — Laplacian on AP (p = 0.52, 0.56) and SRM on R(1 %) (p = 0.60, 0.058);
where significant among these three, |median Δ| < 0.005 recall / < 0.0002 AP. The three short-support
high-pass maps are one equivalence class; texture energy is its best member on HUGO.

## Result 4 — ranking rule vs sampling rule (γ = 1, K = 10)

α = 0.01, B = 1 %: sampling R = 0.038 / 0.016 / 0.024 (within-image sd 0.010 / 0.008 / 0.009; lift 3.8 /
1.6 / 2.4) vs ranking 0.255 / 0.027 / 0.088 → ranking is 6.6× / 1.7× / 3.7× higher. At B = 50 %: sampling
0.887 / 0.650 / 0.738 vs ranking 0.974 / 0.720 / 0.824. γ is a sharpness parameter of the selection, not
part of the map.

## Result 5 — cover-vs-stego ablation (Fig. 4)

ΔAP (stego − cover), α = 0.01, mean [95 % CI]: texture_energy +0.00147 [0.00098, 0.00214] HUGO (4.8 % of
its AP), +0.00022 [0.00012, 0.00037] MiPOD (12 %), +0.00026 [0.00015, 0.00041] S-UNIWARD (4.4 %);
laplacian_residual and srm_residual the same size; gradient_magnitude, local_variance, local_entropy,
wavelet_energy ≈ 0 (|Δ| ≤ 1e-5). Localization, not detection.

## Result 6 — the low-payload regime (α ≤ 0.005)

Mean |C| at α = 0.001 / 0.0025 / 0.005: HUGO 25 / 69 / 147, MiPOD 17 / 46 / 101, S-UNIWARD 20 / 54 / 116;
every image has |C| > 0 (effective n = 1000 everywhere). R(1 %) at 0.001: 0.405 [0.393, 0.417] / 0.028
[0.025, 0.031] / 0.133 [0.126, 0.140]; lift 40.5 / 2.8 / 13.3 (oracle 66.8 / 8.1 / 32.3). Lift rises as
payload falls for HUGO and S-UNIWARD, flat for MiPOD. Power-law fit R(B) = aB^p on the ranking curve at
α = 0.01: p = 0.38 / 0.86 / 0.59 (HUGO / MiPOD / S-UNIWARD), log-log R² 0.96 / 0.998 / 0.994.

## Reproduce

```bash
ROOT=experiments/dessert2026 bash experiments/dessert2026/RUN.md
python3 make_paper_figures.py --root experiments/dessert2026
python3 make_illustration_figure.py --root experiments/dessert2026 --cover-root ALASKA_v2_TIFF_512_GrayScale_10K
python3 make_paper_tables.py --root experiments/dessert2026
```
