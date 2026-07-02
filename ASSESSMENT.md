# Repository Assessment — Blind Steganalysis (Texture-Oriented Carrier-Pixel Probing)

> Audit date: 2026-06-24. Scope: full repo review of the localization-oriented blind-steganalysis
> codebase, ahead of an extension phase. Companion: [`ROADMAP.md`](ROADMAP.md).

## 1. What this project does

**Goal.** *Blind* steganalysis in the **localization** sense: given a stego image and **no knowledge of
the embedding algorithm or the original cover**, estimate **where** the carrier (modified) pixels are,
under a fixed inspection **probing budget** `B`. This is positioned as an intermediate step between
passive detection ("is there hidden data?") and active-warden disruption ("destroy it with minimal
distortion").

**Method (current).** From the analyzed image, build a per-pixel **texture-energy** score
`E(i,j) = D_h + D_v + R4` (horizontal + vertical first differences + absolute 4-neighbour Laplacian
residual). Convert to a sampling distribution `p ∝ (E + ε)^γ` (ε = 1e-6, γ = 1). Sample `B` unique
pixels **without replacement**, weighted by `p`. Score
`H(B) = |probed ∩ true_carriers|` — correctly identified carrier pixels.

**Ground truth.** `conseal` simulates HUGO / MiPOD embedding and the exact changed pixels are recorded.
So this is a **controlled study**: the true carrier set `C` is known, enabling precision/recall.

**Headline result (as written in the draft theses).** `H(B) ≈ a·B^p`, with R² > 0.99; the exponent `p`
varies more with embedding level for HUGO than MiPOD.

## 2. Architecture

| File | Role | Notes |
|---|---|---|
| `embedding.py` | Simulate HUGO/MiPOD via `conseal`; record `*_changes.npz` + `manifest.json` | Per-image SHA-256 seeds → reproducible carriers; proper argparse CLI. **Best-engineered module.** |
| `read_changes.py` | Probing engine: distributions, weighted sampling, summaries, CLI | Distributions: `uniform`, `center_gaussian`, `center_laplace`, `edge_gaussian`, `texture_energy`, `laplacian_residual`. |
| `run_probing_experiment.py` | Sweep `methods × alphas × budget-fractions`; precision/recall | Efficient exponential-race (Efraimidis–Spirakis) weighted sample-without-replacement. |

> **2026-07-01:** `probe.py` (config-as-globals single-run entry point) and all paper-production tooling
> (`generate_*_excel.py`, `build_*_docx.py`, `make_probing_method_illustration.py`, `render_*`) were
> **removed** by project decision — reporting artifacts are recalculated from the tracked CSVs; figures
> now come from `make_comparison_figures.py` (matplotlib, grayscale). `read_changes.py` gained a
> score-map registry (`SCORE_MAP_BUILDERS`: texture_energy, laplacian_residual, local_variance,
> wavelet_energy, srm_residual) and `run_localization_benchmark.py` brackets all of them.

**Data.** ALASKA v2 grayscale 512×512 — a 10K set (≈2.5 GB) and a 50-image working subset. Experiment
outputs live under `experiments/` (gitignored). The stored result set is `experiments/probing_only`
(n = 100, `texture_energy`, fixed seed 12345).

## 3. Strengths (preserve these)

- **Deterministic & reproducible seeding** throughout (SHA-256 per-image seeds; logged run seeds).
- **Clean separation** embed → record → probe → summarize; the change-record `.npz` is a good contract.
- **Correct, efficient** weighted sampling without replacement (no `O(n log n)` re-sort per budget step).
- **Sensible evaluation skeleton**: budgeted localization with precision/recall and a compact power-law summary.

## 4. Findings — ranked

> **Measured (Milestone 1, n=100) — see [`experiments/localization_benchmark/FINDINGS.md`](experiments/localization_benchmark/FINDINGS.md).**
> The predicted findings below are now confirmed: clean **floor < texture < oracle** bracketing; texture
> captures **59–84%** of the achievable (oracle) AP for HUGO but only **34–47%** for MiPOD; cover ≈ stego
> (embedding-specific signal is significant but only ~5–10% of oracle AP → **localization, not detection**).
> Bonus result: deterministic **top-B ranking beats the published γ=1 stochastic sampling by up to 5.6×**
> at B=1% — a free improvement to the existing method.

### MUST-DO — scientific rigor (these gate the paper's claims)

**4.1 — The lift-over-random control was never run.** *(Resolved 2026-07-01: `experiments/probing_uniform`
now exists — the uniform baseline is run alongside every other distribution, and the dangling
`generate_distribution_comparison_excel.py` reference is gone with that script's removal.)*
Originally: the uniform-random baseline the draft's own reviewer explicitly asks for ("*how can we
compare … like naïve probing of random pixels?*") had no data behind it.

This matters because, for sampling-without-replacement, the random baseline is **analytically exact**:
`E[recall] = B/N` (hypergeometric). So the **lift is already provable** from the stored results:

| Config | B=1% | B=5% | B=10% | B=20% | B=50% |
|---|---|---|---|---|---|
| HUGO α=3% — recall | 0.036 | 0.170 | 0.312 | 0.531 | 0.877 |
| HUGO α=3% — **lift vs random** | **3.64×** | 3.41× | 3.12× | 2.66× | 1.75× |
| HUGO α=30% — **lift vs random** | **2.22×** | 2.17× | 2.09× | 1.95× | 1.54× |

**Interpretation.** The method genuinely **works** — 2–4× better than random at small budgets. The honest
caveats: (a) **lift decays toward 1×** as the budget grows (at B=50% you are barely beating a coin), and
(b) **raw `H(B)` counts hide the lift entirely** — `H` rises with budget even for random sampling. The
result lives in the **B ≤ 10%** regime; that is where probing earns its keep and where the writeup should
concentrate.

**4.2 — No oracle ceiling.**
`conseal` exposes the true selection channel — `conseal.mipod.probability(...)` returns per-pixel change
probabilities, and `conseal.hugo.compute_cost_adjusted(...)` returns costs. An **oracle localizer** (the
best any selection-channel-aware method could do) is therefore buildable, and for **rank-based metrics**
(AP, recall@budget) it needs **no payload-λ solving**: rank MiPOD pixels by `probability` descending, and
HUGO pixels by cost ascending (change-probability is monotone-decreasing in cost for additive-distortion
optimal embedding → identical ranking). The gap between texture-energy and this oracle quantifies how much
headroom remains for better content-only estimators.

**4.3 — The "blind" claim needs the cover-vs-stego ablation.**
`probe_image_source` already supports `cover | stego`, but the two have never been compared head-to-head.
Because adaptive embedding makes only ±1 changes, the texture map of the stego is almost identical to that
of the cover — so the expected finding is that **stego-based and cover-based probing perform the same**.
If so, the method extracts **no embedding-specific signal**; it predicts *where any adaptive algorithm
would embed, from image content alone*. **Scope correction that must be stated explicitly: this is
selection-channel _localization_, not stego _detection_** — it cannot answer "is this image stego?". That
is fine for the disruption use-case, but a committee will press this point, so own it.

**4.4 — Metrics conflate prevalence with method quality.**
`H(B)` (a raw count) and the fitted exponent `p` both scale with carrier **prevalence**, which itself
grows with `α`. Add **parameter-free / budget-free** ranking metrics computed directly from the per-pixel
`E` score treated as a detector of carrier-vs-non-carrier:
- **Lead with Average Precision / PR-AUC.** Carriers are only 0.4–10% of pixels; that imbalance makes
  ROC-AUC look flattering, so PR-AUC/AP is the honest primary.
- Report **recall@budget** and **lift@budget = recall / (B/N)** as the headline curve.
- Keep `H(B)=aB^p` as a **descriptive secondary**, not the primary evidence.

**4.5 — No uncertainty quantification.**
Only means are reported, over n = 100 images and a single embed seed. Add **bootstrap / percentile
confidence intervals** and per-image variance. The 10K set is available to scale n and tighten CIs.

### MUST-DO — reproducibility & engineering

**4.6 — The committed runner does not reproduce the published experiment (config drift).**
`run_probing_experiment.py` defaults to `PROBE_DISTRIBUTION = "laplacian_residual"` (line 31) and
`EXPERIMENT_ROOT = experiments/probing_laplacian` (line 23), but the stored, published result is
`texture_energy` in `experiments/probing_only` (per its `metadata.json`). Re-running the code **as
committed** silently produces a *different method's* result in a *different folder*. Root cause:
**configuration held in mutable module-level globals**, edited in place across `probe.py`, the runner, and
`make_probing_method_illustration.py`. → Move to argparse / a versioned config file; persist the resolved
config next to every result.

**4.7 — No `README`, no dependency manifest, no environment pin.**
There is no `requirements.txt` / `pyproject.toml` and no record of the working versions (`conseal 2025.11`,
`numpy 2.4.2`, `scipy 1.17.1`, Python 3.11). A future `conseal`/`numpy` bump can change carriers or break
imports with no warning. → Pin deps; add run instructions and a data-provenance note (which ALASKA subset,
how the 50/10K splits were drawn).

**4.8 — Zero tests.**
A probability/sampling codebase invites cheap, high-value property tests: probabilities sum to 1; sampling
is without replacement; recall is monotone non-decreasing in budget; uniform recall ≈ budget fraction
within tolerance; seed determinism (same seed → same carriers/probes); shape/validation guards.

**4.9 — Fragile global monkeypatch.**
`embedding.py` rebinds `numba.jit` at import to force `cache=False` (works around a `conseal` cache failure
in this environment). It is process-global and order-sensitive. → Isolate behind a documented helper and an
env flag; note the upstream cause.

**4.10 — Paper-production tooling is tangled with the research core.** *(Resolved 2026-07-01 by
removal: all Excel/DOCX/formula tooling deleted per project decision — everything paper-related is
recalculated; grayscale matplotlib figures come from `make_comparison_figures.py`.)*

### SHOULD / LATER (extension phase)

- **4.11 — Performance/scale.** Single-process; the probability map is recomputed per budget fraction;
  `np.unique(current_positions)` runs on already-unique positions. Vectorize and parallelize (e.g. joblib)
  before scaling to the 10K set × 2 methods × 7 α.
- **4.12 — Flat module layout.** Scripts import each other at top level. Promote to a `steganalysis/`
  package as the code grows.
- **4.13 — Narrow coverage.** Only HUGO/MiPOD, spatial domain. `conseal` also offers S-UNIWARD / HILL /
  WOW and JPEG-domain methods → breadth for generalization claims.
- **4.14 — Single hand-crafted feature.** *(Partially addressed 2026-07-01: the score-map registry now
  holds five hand-crafted maps — texture energy, Laplacian residual, local variance, wavelet detail
  energy, fixed-kernel SRM residual — compared in `experiments/distribution_comparison/`. The learned
  rungs of the ladder remain future work — see `ROADMAP.md`.)*

## 5. One-line verdict

A clean, reproducible **probing engine** with a promising small-budget result, currently **missing its
three load-bearing controls** (random floor, oracle ceiling, cover-vs-stego), reporting **prevalence-
sensitive metrics**, and carrying a **config-drift reproducibility bug**. All are addressable without
rewriting the core — see the hardening-first plan in [`ROADMAP.md`](ROADMAP.md).
