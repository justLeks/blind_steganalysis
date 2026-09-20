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

## Milestone 1 — Set the brackets & fix metrics  *(DONE 2026-06-25; see `experiments/localization_benchmark/FINDINGS.md`)*

**Why:** the current claims are uninterpretable without floor, ceiling, and honest metrics (Assessment §4.1–4.5).

- [x] **Uniform random baseline** → analytic floor in the benchmark; the physical `experiments/probing_uniform`
      run was generated 2026-07-01 with the distribution-comparison campaign.
- [x] **Oracle localizer** from `conseal`'s selection channel (`selection_channel.py`). Rank MiPOD by
      `mipod.probability` desc, HUGO by `hugo.compute_cost_adjusted` asc. Upper-bound curves + texture-vs-oracle gap.
- [x] **Cover-vs-stego ablation** — cover ≈ stego; *localization-not-detection* scope statement written.
- [x] **Metric upgrade**: AP/PR-AUC, recall@budget, lift@budget first-class (`localization_metrics.py`).
- [x] **Uncertainty**: bootstrap/percentile CIs on all headline numbers. (Scaling n past 100 on the 10K set
      remains open.)

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
- [x] **Decouple reporting** — superseded 2026-07-01: all paper-production tooling (Excel/DOCX/formula
      renderers, `make_probing_method_illustration.py`) and the legacy `probe.py` runner were **deleted**
      by project decision; reporting artifacts are recalculated from the tracked CSVs and grayscale
      matplotlib figures (`make_comparison_figures.py`). No config-as-globals scripts remain.

**Skills used:** `engineering-skills:tdd-guide` (tests) + `senior-data-engineer` (config/packaging).

## Milestone 3a — Hand-crafted score-map comparison  *(DONE 2026-07-01)*

**What:** widen the "current method" rung before any learning: a score-map registry
(`read_changes.SCORE_MAP_BUILDERS`) with `texture_energy`, `laplacian_residual`, `local_variance`
(targets MiPOD's variance model), `wavelet_energy` (WOW/S-UNIWARD 16-tap Daubechies-8 bank) and a
fixed-kernel `srm_residual`; per-distribution probing runs (`experiments/probing_<dist>/`); a
12-localizer bracketed benchmark (`experiments/distribution_benchmark/`); comparison CSVs + grayscale
figures + score-map panels (`experiments/distribution_comparison/`, see its `FINDINGS.md`).

## Milestone 3 — Estimator ladder  *(future; next-paper novelty, deferred)*

**Why:** texture-energy is a single hand-crafted feature; climbing the ladder is the novelty arc.

- [ ] **SRM-style residual localizer** — *fixed-kernel version shipped in M3a (`srm_residual`)*; the
      full SRM feature set / learned variant remains open.
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

## Milestone 5 — DESSERT-2026 revision campaign  *(2026-09; see `experiments/dessert2026/FINDINGS.md`)*

**Why:** the DESSERT-2026 review asked for a uniform baseline at every budget, strong texture/residual
baselines, low payloads, a third embedder, uncertainty and paired tests, normalized metrics, and full
reproducibility. Everything is answered from one n = 1000 campaign (seeded subset of the 10K set).

- [x] **S-UNIWARD** embedding + exact oracle (`embedding.py`, `selection_channel.py`).
- [x] **Two new score maps**: `gradient_magnitude` (Sobel) and `local_entropy` (9×9, 32 bins).
- [x] **Metrics**: `precision@B`, normalized area under the recall–budget curve `nAURC(b_max)`, fractional
      budget labels (`0p5`).
- [x] **Subset of record** `data/subsets/alaska10k_n1000_seed12345.txt` and `--cover-list` everywhere.
- [x] **Repeated sampling** (`--repeats K`) and `--workers` in the probing runner (K=1 stays bit-exact).
- [x] **Paired tests**: `run_paired_tests.py` (Wilcoxon signed-rank, Holm).
- [x] **Campaign**: `experiments/dessert2026/RUN.md`; paper figures/tables generators.

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
