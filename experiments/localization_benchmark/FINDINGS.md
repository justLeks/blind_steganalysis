# Milestone 1 — Localization Benchmark Findings

> Generated 2026-06-24. n = 100 ALASKA images, HUGO + MiPOD × 7 embedding levels (α = 0.03…0.50),
> budgets B ∈ {1,5,10,20,30,40,50}%, percentile bootstrap CIs (2000 resamples). Reuses the existing
> `experiments/probing_only/steganograms` (no re-embedding). Code: `run_localization_benchmark.py`,
> `localization_metrics.py`, `selection_channel.py`. Raw: `per_image.csv`, `summary.csv`.

Four localizers, all scored against the recorded carrier mask:
`uniform` (random **floor**) · `texture_stego` (current method) · `texture_cover` (ablation) ·
`oracle` (true conseal selection channel = **ceiling**).

The oracle is the simulator's *exact* selection channel (HUGO via `_ternary.probability`, MiPOD via
`mipod.probability` = `2·ternary_probs`). Validated empirically: per-image
`z = (C − Σp)/√Σp(1−p)` is standard-normal (HUGO mean z = −0.12, std 1.04; MiPOD −0.11, std 0.94 over
n=40), i.e. the recorded carriers are an exact Poisson-binomial draw from the oracle map.

---

## Result 1 — The brackets exist, and texture captures far more of HUGO than of MiPOD

Average Precision (primary metric; carriers are a 0.4–10% minority). **Texture efficiency** =
(AP_texture − AP_floor) / (AP_oracle − AP_floor) = fraction of the *achievable* selection-channel signal
that the content-only texture map captures.

| Method | α | AP floor | AP texture | AP oracle | **texture efficiency** |
|---|---|---|---|---|---|
| HUGO | 0.03 | 0.0043 | 0.0687 | 0.1132 | **59%** |
| HUGO | 0.10 | 0.0163 | 0.1223 | 0.1730 | **68%** |
| HUGO | 0.50 | 0.1065 | 0.2809 | 0.3146 | **84%** |
| MiPOD | 0.03 | 0.0031 | 0.0082 | 0.0183 | **34%** |
| MiPOD | 0.10 | 0.0129 | 0.0307 | 0.0625 | **36%** |
| MiPOD | 0.50 | 0.1005 | 0.1791 | 0.2674 | **47%** |

**Reading.** Texture-energy probing recovers **59–84%** of the optimal selection-channel AP for HUGO
(improving with payload) but only **34–47%** for MiPOD. This is the rigorous, quantified version of the
draft's vague "more sensitive to HUGO than MiPOD": for MiPOD the texture proxy leaves more than half of
the achievable signal on the table, because MiPOD's model-based costs are not well approximated by a
first-order gradient map. **The oracle gap is the headroom that the estimator ladder (SRM → CNN) targets.**

## Result 2 — For carrier recall, greedy ranking (γ→∞) dominates proportional sampling (γ=1)

The draft draws B pixels by weighted sampling with probability ∝ E^γ at **γ=1**. Deterministic top-B is
the **γ→∞** limit of the *same* probability map — i.e. the same method with the selection sharpness turned
up. For the carrier-recall objective, sharper is uniformly better. recall@1% as a function of γ (same
texture map, 40 images; γ=1 reproduces the published stochastic value):

| Method | α | γ=1 | γ=2 | γ=4 | γ=8 | **γ→∞ (top-B)** |
|---|---|---|---|---|---|---|
| HUGO | 0.03 | 0.038 | 0.080 | 0.145 | 0.186 | **0.201** |
| HUGO | 0.10 | 0.030 | 0.055 | 0.088 | 0.108 | **0.116** |
| MiPOD | 0.03 | 0.019 | 0.025 | 0.031 | 0.033 | **0.034** |

recall@budget, published γ=1 sampling vs the γ→∞ top-B of the same map (full n=100 benchmark):

| Method | α | B | γ=1 sampling | **γ→∞ top-B** | ratio |
|---|---|---|---|---|---|
| HUGO | 0.03 | 1% | 0.0364 | **0.2046** | **5.6×** |
| HUGO | 0.10 | 1% | 0.0294 | **0.1195** | 4.1× |
| HUGO | 0.03 | 10% | 0.3121 | **0.6853** | 2.2× |
| MiPOD | 0.03 | 1% | 0.0166 | **0.0339** | 2.0× |

**Reading.** The recall gain (up to **5.6×** at B=1%) comes entirely from the **γ knob**, not new
information — the response is monotone in γ and γ=8 already attains ~92% of the top-B value. **Actionable:**
for the recall objective, report the top-B (γ→∞) operating point, or treat γ as a tuned hyper-parameter
rather than fixing γ=1. (γ=1 sampling remains the right choice only if probe *diversity*, not recall, is
the goal.)

## Result 3 — Cover ≈ stego: this is localization, not detection

Paired per-image AP difference (texture on **stego** minus texture on **cover**), bootstrap 95% CI:

| Method | α | ΔAP (stego − cover) | 95% CI | as % of AP_oracle |
|---|---|---|---|---|
| HUGO | 0.03 | +0.0054 | [0.0015, 0.0126] | 4.8% |
| HUGO | 0.50 | +0.0213 | [0.0123, 0.0350] | 6.8% |
| MiPOD | 0.03 | +0.0018 | [0.0003, 0.0045] | 9.7% |
| MiPOD | 0.50 | +0.0167 | [0.0088, 0.0288] | 6.2% |

**Reading.** Probing the stego is **statistically** better than probing the cover (every CI excludes 0) —
the ±1 embedding changes do raise local texture energy at carrier sites, a real embedding fingerprint.
But the effect is **small: only ~5–10% of the oracle AP**; ~90%+ of the map is explained by image content
alone. **Scope statement for the paper:** texture-energy probing is a *content-based selection-channel
**localizer*** — it answers "given that this image carries a payload, where are the carriers?" It is **not
a stego detector** and must not be presented as one. (The small, significant stego>cover gap is a hook for
future detection-oriented work.)

## Result 4 — Report PR-AUC / AP, not ROC-AUC

ROC-AUC flatters under class imbalance: HUGO α=0.03 texture ROC-AUC = **0.898** looks excellent, while the
honest AP = **0.069** (vs 0.004 prevalence) shows the true difficulty. Lead with AP; ROC-AUC is secondary.

## Result 5 — Lift over random decays with both budget and payload

recall@1% lift = recall / (B/N): HUGO **20.5× (α=0.03) → 4.0× (α=0.50)**; MiPOD **3.4× → 2.2×**. Within a
fixed α, lift also falls as B grows (HUGO α=0.03: ~20× at B=1% → ~1.8× at B=50%). **The method earns its
keep at small budgets (B ≤ 10%) and low payloads** — exactly the operationally interesting regime.

---

## What this changes for the draft

1. Replace raw `H(B)` headline with **AP + lift@budget**, bracketed by floor and oracle; keep `H(B)=aB^p`
   as a descriptive secondary.
2. Re-state the HUGO/MiPOD contrast quantitatively: **texture efficiency 59–84% vs 34–47%**.
3. Treat **selection sharpness γ** as a tuned hyper-parameter (Result 2); report the top-B (γ→∞) operating
   point for the recall objective — a 2–5× gain at small budgets over the fixed γ=1.
4. Add the **localization-not-detection** scope paragraph (Result 3) pre-empting the obvious committee question.

## Scope & deferrals

- **n.** Results are n=100; the bootstrap CIs already separate every reported effect (floor vs texture vs
  oracle, and the cover-vs-stego delta). Scaling to the 10K set (Milestone 1 stretch) will tighten the
  numbers but is not expected to change any conclusion — a deliberate deferral, not a gap.
- **Random baseline.** The floor is reported as the exact analytic expectation (E[recall]=B/N, AP=prevalence),
  not a Monte-Carlo run; the still-dangling `experiments/probing_uniform` reference in
  `generate_distribution_comparison_excel.py` is a Milestone-2 reporting-cleanup item.
- **Ties.** `E` is integer-valued (heavy ties); recall@budget breaks ties at random and AP collapses tied
  thresholds (sklearn semantics). The oracle map is continuous (no ties). γ=1 here matches the published
  stochastic value, confirming the harness reproduces the original method.

## Reproduce

```bash
python3 run_localization_benchmark.py                 # full sweep → experiments/localization_benchmark/
python3 run_localization_benchmark.py --limit 10      # quick pass
python3 localization_metrics.py                        # metric self-tests
```
Config is taken from CLI args and saved to `metadata.json` beside the outputs (no config-as-globals).
