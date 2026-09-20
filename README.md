# Blind Steganalysis — Carrier-Pixel Localization by Probing

Localization-oriented **blind** steganalysis: given a stego image and **no knowledge of the embedding
algorithm or the original cover**, estimate **where** the carrier (modified) pixels are under a fixed
inspection ("probing") budget `B`. Ground truth is obtained by simulating HUGO / MiPOD embedding with
[`conseal`](https://conseal.readthedocs.io) and recording the exact changed pixels, so every localizer
can be scored against a known carrier mask.

> **Scope.** This is carrier-pixel **localization**, not stego **detection**. It answers *"given that an
> image carries a payload, where are the carriers?"* — it does **not** decide whether an image is stego.

**Project docs:** [`ASSESSMENT.md`](ASSESSMENT.md) (audit) · [`ROADMAP.md`](ROADMAP.md) (milestones)
· [`experiments/localization_benchmark/FINDINGS.md`](experiments/localization_benchmark/FINDINGS.md)
(Milestone 1: texture-energy bracketing)
· [`experiments/distribution_comparison/FINDINGS.md`](experiments/distribution_comparison/FINDINGS.md)
(Milestone 3a: score-map comparison).

## Install

```bash
python3 -m pip install -r requirements.txt    # Python 3.11
```

## Data

| Path | Contents | In git? |
|---|---|---|
| `ALASKA_v2_TIFF_512_GrayScale_10K/` | ALASKA v2 grayscale 512×512, 10 000 cover TIFFs (full set). | no (external) |
| `ALASKA_v2_TIFF_512_GrayScale_50/` | **100**-image working subset (the `50` in the name is historical). All published results use it. | no (external) |
| `data/subsets/alaska10k_n1000_seed12345.txt` | The **1000-cover subset of record** for the DESSERT-2026 campaign (`select_cover_subset.py`, seed 12345); every runner takes it via `--cover-list`. | yes |
| `experiments/**/steganograms/` | Stego PNGs + `*_changes.npz` carrier records + `manifest.json`. | no — **re-derivable**: embedding is deterministic (per-image SHA-256 seeds from seed 12345), so any runner regenerates them on demand |
| `experiments/**` (CSV/JSON/PNG) | Experiment results and figures. | **yes** — results of record |

16-bit TIFFs are reduced to 8-bit (`/257`) on load (`embedding.to_u8_array`).

## Method: the estimator ladder and the three controls

Every probing approach ("score map") assigns each pixel a score (higher = more likely a carrier) and is
evaluated inside the same bracket:

```
random floor  →  hand-crafted score maps (this repo)  →  [SRM-learned → CNN, future]  →  conseal oracle ceiling
```

**No result is reported without three controls:**

1. **Random floor** — uniform probing has `E[recall] = B/N` exactly; report **lift = recall / (B/N)**.
2. **Oracle ceiling** — conseal's true selection channel (`selection_channel.py`; HUGO and S-UNIWARD via
   their adjusted costs + `_ternary.probability`, MiPOD via `mipod.probability`), the best any
   localizer could do.
3. **Cover-vs-stego ablation** — every map is scored on the stego image *and* on the cover. If the two
   match, the map carries no embedding-specific signal: it predicts the content-driven selection
   channel (still useful for localization, but say so honestly).

**Metrics** (`localization_metrics.py`, numpy-only): Average Precision / PR-AUC (primary — carriers are
a 0.4–10 % minority, so PR-AUC is honest where ROC-AUC flatters), recall@budget and lift@budget
(headline operating points), percentile-bootstrap 95 % CIs over images.

### Score maps (`read_changes.SCORE_MAP_BUILDERS`)

| Map | Score | Params | Borders | What it targets |
|---|---|---|---|---|
| `texture_energy` | \|dx\| + \|dy\| + \|3×3 Laplacian\| | `gamma`, `floor` | zero frame | the original published method |
| `gradient_magnitude` | sqrt(Gx² + Gy²), 3×3 Sobel | `gamma`, `floor` | reflect | the classic first-order texture baseline |
| `laplacian_residual` | \|3×3 Laplacian\| | `gamma`, `floor` | zero frame | ablation: Laplacian term alone |
| `local_variance` | windowed E[X²] − E[X]² | `window` (9), `gamma`, `floor` | reflect | MiPOD's variance-driven cost model |
| `local_entropy` | Shannon entropy (bits) of the windowed gray-level histogram, 32 bins | `window` (9), `bins` (32), `gamma`, `floor` | reflect | histogram-complexity baseline |
| `wavelet_energy` | \|LH\|+\|HL\|+\|HH\|, 16-tap Daubechies-8, undecimated | `gamma`, `floor` | reflect | the WOW / S-UNIWARD filter bank |
| `srm_residual` | Σ \|conv(x, K)\|/q over 1st-order diffs, 3×3 KB, 5×5 KV | `gamma`, `floor` | reflect | fixed-kernel preview of the SRM rung |

Each map serves two roles from one implementation: **ranking scores** for the benchmark (top-B,
deterministic) and **sampling probabilities** `(energy + floor)^gamma / Σ` for the stochastic probing
runs. `uniform` plus three location priors (`center_gaussian`, `center_laplace`, `edge_gaussian`)
complete the supported distributions.

## Pipeline

```
cover TIFFs ──embedding.py──▶ stego PNG + *_changes.npz ──┬─▶ run_probing_experiment.py    (γ=1 sequential probing, one run per distribution)
        (auto-run on demand by the runners)               └─▶ run_localization_benchmark.py (AP / recall@B / lift, 12 localizers, floor↔ceiling)
                                                                                 │
                                                              make_comparison_figures.py
                                                              (comparison CSVs + grayscale figures + score-map panels)
```

| Module | Role |
|---|---|
| `embedding.py` | Simulate HUGO/MiPOD via conseal; record changed pixels. Per-image SHA-256 seeds. CLI. |
| `read_changes.py` | Score-map registry, probing distributions, weighted sampling without replacement. CLI. |
| `selection_channel.py` | Exact conseal **oracle** (true per-pixel change probability). |
| `localization_metrics.py` | AP / PR-AUC, ROC-AUC, recall@budget, lift, bootstrap CIs. Self-test via `python3 localization_metrics.py`. |
| `run_probing_experiment.py` | Budgeted probing sweep over methods × alphas × budgets → `probe_raw.csv` / `probe_summary.csv`. |
| `run_localization_benchmark.py` | Bracketed benchmark: `uniform · <map>_{stego,cover} · oracle` → `per_image.csv` / `summary.csv`. |
| `make_comparison_figures.py` | Cross-distribution CSVs, grayscale metric figures, score-map panels. |
| `select_cover_subset.py` | Seeded random cover subset → cover-list file. |
| `run_paired_tests.py` | Paired Wilcoxon signed-rank tests (Holm-corrected) of one localizer against baselines over `per_image.csv`. |
| `make_paper_figures.py`, `make_illustration_figure.py`, `make_paper_tables.py` | DESSERT-2026 paper figures (recall vs budget, lift vs payload, cover-vs-stego ablation, illustration) and tables (CSV + Markdown). |

## Run the full pipeline

Steganograms are embedded automatically on first use (deterministic; ~min). Each command persists its
resolved config to a `metadata.json` next to its outputs.

```bash
# 1) Probing runs (γ=1 stochastic sequential probing), one experiment dir per distribution.
#    experiments/probing_only is the texture_energy run of record (committed defaults reproduce it).
python3 run_probing_experiment.py                                 # texture_energy → experiments/probing_only
for d in uniform laplacian_residual local_variance wavelet_energy srm_residual; do
  python3 run_probing_experiment.py --distribution $d \
    --experiment-root experiments/probing_$d \
    --steganogram-root experiments/probing_only/steganograms      # reuse the canonical steganograms
done

# 2) Localization benchmark (12 localizers, bootstrap CIs) → experiments/distribution_benchmark.
#    ~1 h for the full n=100 sweep (the conseal oracle dominates); use --limit for a quick pass.
python3 run_localization_benchmark.py
python3 run_localization_benchmark.py --limit 10                  # quick pass

# 3) Comparison CSVs + grayscale figures → experiments/distribution_comparison.
python3 make_comparison_figures.py
```

### DESSERT-2026 revision campaign (n = 1000, HUGO + MiPOD + S-UNIWARD, α from 0.001 bpp)

`experiments/dessert2026/RUN.md` is a runnable bash script: it embeds the 1000-cover subset for the 30
(method, α) configurations, runs the γ=1 probing sweeps with `--repeats 10` (texture_energy and uniform),
the 16-localizer benchmark (budgets from 0.5 %, plus precision@B and nAURC columns), and the paired tests.
Then `make_paper_figures.py`, `make_illustration_figure.py` and `make_paper_tables.py` derive the paper's
figures and tables. Findings of record: `experiments/dessert2026/FINDINGS.md`.

```bash
ROOT=experiments/dessert2026 bash experiments/dessert2026/RUN.md      # ~5 h with WORKERS=8
python3 make_paper_figures.py --root experiments/dessert2026
python3 make_illustration_figure.py --root experiments/dessert2026 --cover-root ALASKA_v2_TIFF_512_GrayScale_10K
python3 make_paper_tables.py --root experiments/dessert2026
```

## Outputs

**Per probing run** (`experiments/probing_<dist>/`):

- `probe_raw.csv` — one row per (image, budget): `config_id, record_name, method, alpha, probe_seed,
  probe_distribution, probe_budget_fraction, probe_budget_pixels, guessed_carrier_pixels,
  precision_hit_rate, carrier_recall, incremental_*`, …
- `probe_summary.csv` — aggregated per (method, alpha, budget): `mean_carrier_recall,
  mean_precision_hit_rate, mean_guessed_carrier_pixels`, …

**Benchmark** (`experiments/distribution_benchmark/`):

- `per_image.csv` — one row per (image, localizer): `average_precision, roc_auc, prevalence,
  recall_at_<pct>, lift_at_<pct>` for pct ∈ {1,5,10,20,30,40,50}.
- `summary.csv` — per (method, alpha, localizer): means with `*_ci_lo/_ci_hi` 95 % bootstrap CIs.

**Comparison** (`experiments/distribution_comparison/`):

- `comparison_ranking.csv` — per (method, alpha, localizer): AP ± CI, ROC, recall/lift@budget ± CI,
  plus joined `floor_ap`, `oracle_ap`, and `ap_efficiency = (AP − floor)/(oracle − floor)`.
- `comparison_sampling.csv` — per (distribution, method, alpha, budget): the γ=1 sampling view.
- `figures/` — grayscale-only charts (gray level × line style × marker; no color):
  `fig_recall_vs_budget_*`, `fig_lift_vs_budget_*` (α ∈ {0.05, 0.2}), `fig_ap_vs_alpha_*`,
  `fig_ap_efficiency_vs_alpha_*` (the headline: fraction of oracle AP captured),
  `fig_cover_vs_stego_*` (the ablation, per map), `fig_sampling_recall_vs_budget_*`.
- `panels/` — per-image grayscale panels: cover · true carrier mask · oracle p(change) · all five
  score maps, each independently clipped to [p1, p99]. Images picked deterministically at the
  10th/50th/90th prevalence percentiles.

## Tests

```bash
python3 -m unittest discover -s tests -v      # property tests (stdlib, no extra deps)
python3 localization_metrics.py               # metric self-tests
```

`tests/test_score_maps.py` carries frozen verbatim copies of the pre-registry texture-energy /
Laplacian implementations and asserts the current code is **bit-exact** against them.

## Reproducibility

- **Deterministic seeds.** Per-image embed seeds are SHA-256-derived (`embedding.derive_image_seed`);
  per-record probe seeds hash (run seed, method, alpha, record, distribution) — so runs are
  reproducible record-by-record (`--limit N` rows match the full run) and distributions get
  independent probe orders. Default run seed: 12345 everywhere.
- **Resolved config is persisted** to `metadata.json` next to every result, including the fully
  resolved score-map parameters.
- **Committed defaults reproduce the published experiment** (`run_probing_experiment.py` →
  `experiments/probing_only`, texture_energy, seed 12345) — re-verified after the registry refactor
  (field-for-field identical rows).
- **Benchmark tie-breaking caveat:** one RNG is shared sequentially across rows, so per-image
  recall/lift columns reproduce only under an identical (localizers, limit) configuration; AP and
  ROC-AUC are RNG-free. The Milestone-1 run is frozen in `experiments/localization_benchmark/`.
- **Pinned dependencies** (`requirements.txt`); a conseal/numpy bump can silently change carriers —
  re-run the regression above after any upgrade.
- **conseal JIT cache** is disabled at import via `embedding.prepare_conseal_import()`; override with
  `CONSEAL_DISABLE_NUMBA_CACHE` / `NUMBA_DISABLE_JIT`.

## Repo layout

```
embedding.py                     # conseal embedding + carrier records
read_changes.py                  # score-map registry + sampling
selection_channel.py             # oracle
localization_metrics.py          # metrics
run_probing_experiment.py        # γ=1 probing sweeps
run_localization_benchmark.py    # bracketed ranking benchmark
make_comparison_figures.py       # comparison CSVs + grayscale figures/panels
tests/                           # property tests (unittest)
experiments/
  probing_only/                  # texture_energy probing run of record (+ steganograms, git-ignored)
  probing_<distribution>/        # probing runs for uniform + the other score maps
  localization_benchmark/        # Milestone-1 benchmark (frozen record) + FINDINGS.md
  distribution_benchmark/        # 12-localizer benchmark
  distribution_comparison/       # comparison CSVs, figures/, panels/ + FINDINGS.md
  dessert2026/                   # DESSERT-2026 campaign: RUN.md, benchmark/, probing_*/, paired_tests/,
                                 #   comparison/figures/, paper/ (tables), FINDINGS.md
```
