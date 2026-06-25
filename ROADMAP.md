# Roadmap — Hardening & Extending Blind Steganalysis

> Companion to [`ASSESSMENT.md`](ASSESSMENT.md). Priority order reflects the chosen direction:
> **harden the current texture-energy result first**, then climb to new methods. Disruption is a
> documented future milestone, not built yet.

## Organizing frame: a ladder of selection-channel estimators

Every localizer is scored under **one** metric set and bracketed by a floor and a ceiling:

```
random (floor)  →  texture-energy (current)  →  SRM residuals  →  learned CNN predictor  →  conseal oracle (ceiling)
```

- **Floor** = uniform random; `E[recall] = B/N` exactly (no run needed, but run once for variance/CIs).
- **Ceiling** = oracle ranking from `conseal`'s true selection channel.
- Progress = how far up the ladder a *blind, content-only* estimator can climb between floor and ceiling.

**Canonical metric set** (apply to every rung):
1. **Average Precision / PR-AUC** — primary, parameter-free, budget-free (carriers are a 0.4–10% minority).
2. **recall@budget** and **lift@budget = recall / (B/N)** — headline curve; foreground **B ≤ 10%**.
3. `H(B) = a·B^p` — descriptive secondary only.
4. **Bootstrap CIs** on all of the above.

---

## Milestone 1 — Set the brackets & fix metrics  *(must-do science)*

**Why:** the current claims are uninterpretable without floor, ceiling, and honest metrics (Assessment §4.1–4.5).

- [ ] **Uniform random baseline** → `experiments/probing_uniform`; overlay the analytic `recall = B/N`
      reference line. Fixes the dangling reference in `generate_distribution_comparison_excel.py`.
- [ ] **Oracle localizer** from `conseal`'s selection channel. Rank MiPOD by `mipod.probability` desc,
      HUGO by `hugo.compute_cost_adjusted` asc (monotone in change-probability — verify monotonicity once).
      Produce upper-bound curves and the texture-energy-vs-oracle gap.
- [ ] **Cover-vs-stego ablation** using the existing `probe_image_source` switch; write the
      *localization-not-detection* scope statement based on the result.
- [ ] **Metric upgrade**: add AP/PR-AUC, recall@budget, lift@budget as first-class outputs; the per-pixel
      `E` score is itself a detector, so AP needs no budget at all.
- [ ] **Uncertainty**: bootstrap/percentile CIs; scale n past 100 using the 10K set; report per-image variance.

**Primary skill:** `engineering-skills:senior-data-scientist`. **Supporting:** `senior-data-engineer` (scale-up).

## Milestone 2 — Reproducibility & hygiene  *(DONE 2026-06-25)*

**Why:** the committed code did not reproduce the published numbers (Assessment §4.6–4.10).

- [x] **Pin & document**: `requirements.txt` (numpy/pillow/conseal/numba/scipy pinned; reporting deps
      optional) + `README.md` (data layout & provenance, exact run commands, seeds, reproducibility).
- [x] **Kill config drift**: `run_probing_experiment.py` now has argparse over importable defaults, and the
      committed defaults reproduce `experiments/probing_only` (`texture_energy`, seed 12345). **Verified
      exact** — re-probing matches the stored `probe_raw.csv` 21/21 cells; resolved config persisted to
      `metadata.json`. `run_localization_benchmark.py` follows the same args-not-globals pattern.
- [x] **Property tests** (`tests/`, stdlib unittest, 16 tests): probabilities sum to 1; sampling without
      replacement; recall monotone in budget; uniform recall ≈ fraction; seed determinism; oracle
      `E[#changes]=Σp` identity; metric correctness.
- [x] **Isolate** the `numba.jit` workaround behind a documented `prepare_conseal_import()` + env flags
      (`NUMBA_DISABLE_JIT`, `CONSEAL_DISABLE_NUMBA_CACHE`).
- [x] **Decouple reporting** hardcoded path → `THESES_SOURCE_DOCX` env var with a clear error; no
      machine-specific path remains in `*.py`. *Deferred (do-no-harm to the live paper pipeline):* the
      physical move of the Excel/DOCX generators into `reporting/` and the matplotlib/notebook migration —
      these run the active conference submission, so reorganise only on request. `probe.py` and
      `make_probing_method_illustration.py` still use config-as-globals; migrate when next touched.

**Skills used:** `engineering-skills:tdd-guide` (tests) + `senior-data-engineer` (config/packaging).

## Milestone 3 — Estimator ladder  *(future; next-paper novelty, deferred)*

**Why:** texture-energy is a single hand-crafted feature; climbing the ladder is the novelty arc.

- [ ] **SRM-style residual localizer** — richer residual descriptors than the single 4-neighbour Laplacian.
- [ ] **Learned CNN selection-channel predictor** — train a small CNN to predict per-pixel change
      probability from the stego image (the true "blind selection-channel estimation"); compare to oracle.
- [ ] **Breadth** — add `conseal` methods (S-UNIWARD/HILL/WOW; JPEG-domain) and color, to test generalization.
- [ ] Promote the codebase to a `steganalysis/` package; parallelize the pipeline for the 10K set.

**Primary skill:** `engineering-skills:senior-computer-vision` (residuals, CNN, PyTorch) +
`engineering-skills:senior-ml-engineer` (training/tracking).

## Milestone 4 — Disruption / active-warden loop  *(future; documented design sketch only)*

**Why:** the draft motivates probing as a precursor to **targeted disruption with minimal distortion** —
the localize→disrupt→measure loop closes that narrative.

**Design sketch (not implemented this phase):**
1. **Localize** — probe the top-`B` ranked pixels (any rung of the ladder).
2. **Disrupt** — apply a localized perturbation only to those pixels (e.g. ±1 / small dither / local filter).
3. **Measure** — payload destruction (extraction failure / bit-error rate) **vs.** image distortion
   (PSNR/SSIM). The deliverable is a destruction-vs-distortion trade-off curve, with random and oracle
   localization as floor/ceiling.

**Starting point:** the deleted `message_damage_experiment.py`, `visual_attack_test.py`,
`run_multi_attack_experiment.py` are recoverable from git history (commit `6ce754b` removed them) and can
seed step 2–3.

**Primary skill:** `engineering-skills:senior-data-scientist` (trade-off design) + `senior-computer-vision`.

---

## Agents & skills — mapping

| Workstream | Best-fit skill / agent | When |
|---|---|---|
| Metrics (AP/PR-AUC), baselines, CIs, significance, power-law fit | `engineering-skills:senior-data-scientist` | M1 (primary) |
| Pipeline scale-up & parallelism (10K set) | `engineering-skills:senior-data-engineer` | M1–M2 |
| Config/packaging, README, deps | `engineering-skills:senior-data-engineer` | M2 |
| Property/regression tests | `engineering-skills:tdd-guide` | M2 |
| Residual features, learned CNN predictor | `engineering-skills:senior-computer-vision` | M3 |
| Model training / experiment tracking | `engineering-skills:senior-ml-engineer` | M3 |
| Codebase fan-out search; implementation design | `Explore` / `Plan` agents | each milestone start |
| Pre-merge review | `code-review`, `engineering-skills:adversarial-reviewer` | each milestone end |

encodes this project's conventions (data contract, metric set, the three mandatory controls, the ladder,
reproducibility rules, conseal API pointers) so future sessions invoke it directly instead of re-deriving.
