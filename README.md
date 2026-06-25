# Blind Steganalysis — Texture-Oriented Carrier-Pixel Probing

Localization-oriented **blind** steganalysis: given a stego image and **no knowledge of the embedding
algorithm or the original cover**, estimate **where** the carrier (modified) pixels are under a fixed
inspection ("probing") budget. Ground truth is obtained by simulating HUGO / MiPOD embedding with
[`conseal`](https://conseal.readthedocs.io) and recording the exact changed pixels, so localizers can be
scored against a known carrier mask.

> **Scope.** This is carrier-pixel **localization**, not stego **detection**. It answers *"given that an
> image carries a payload, where are the carriers?"* — it does **not** decide whether an image is stego.
> See [`experiments/localization_benchmark/FINDINGS.md`](experiments/localization_benchmark/FINDINGS.md).

**Project docs:** [`ASSESSMENT.md`](ASSESSMENT.md) (audit) · [`ROADMAP.md`](ROADMAP.md) (plan & milestones)
· [`experiments/localization_benchmark/FINDINGS.md`](experiments/localization_benchmark/FINDINGS.md)
(Milestone-1 results).

## Install

```bash
python3 -m pip install -r requirements.txt    # Python 3.11
```

## Data layout (git-ignored — not in the repo)

| Directory | Contents |
|---|---|
| `ALASKA_v2_TIFF_512_GrayScale_10K/` | ALASKA v2 grayscale 512×512, 10 000 cover TIFFs (full set). |
| `ALASKA_v2_TIFF_512_GrayScale_50/` | 100-image working subset (the `50` in the name is historical). |
| `experiments/` | All generated outputs (steganograms, change records, CSVs, figures). |

The published experiment lives in `experiments/probing_only/` (n=100, `texture_energy`, seed 12345); the
Milestone-1 benchmark in `experiments/localization_benchmark/`.

## Pipeline

```
cover TIFFs ──embedding.py──▶ stego PNG + *_changes.npz (carrier mask) ──┐
                                                                          ├─▶ run_probing_experiment.py  (budgeted probing sweep, H(B))
                                                                          └─▶ run_localization_benchmark.py (AP / recall@budget / lift,
                                                                                bracketed by random floor & conseal oracle ceiling)
```

| Module | Role |
|---|---|
| `embedding.py` | Simulate HUGO/MiPOD via conseal; record changed pixels. Per-image SHA-256 seeds. CLI. |
| `read_changes.py` | Probing distributions + weighted sampling without replacement; per-record summaries. CLI. |
| `run_probing_experiment.py` | Probing sweep over methods × alphas × budgets → `probe_raw.csv` / `probe_summary.csv`. |
| `selection_channel.py` | Exact conseal **oracle** localizer (true per-pixel change probability). |
| `localization_metrics.py` | numpy-only AP / PR-AUC, ROC-AUC, recall@budget, lift, bootstrap CIs. |
| `run_localization_benchmark.py` | Bracketed benchmark: `uniform · texture_stego · texture_cover · oracle`. |
| `generate_*_excel.py`, `build_*_docx.py`, `render_formula*` | Paper-production tooling (optional, separate concern). |

### Run

```bash
# 1) Embed (writes stego + carrier records). Defaults: HUGO, alpha 0.3, seed 12345.
python3 embedding.py ALASKA_v2_TIFF_512_GrayScale_50 experiments/out --method HUGO --alpha 0.3

# 2) Probing sweep (reproduces experiments/probing_only with committed defaults; configurable via flags).
python3 run_probing_experiment.py
python3 run_probing_experiment.py --distribution uniform --experiment-root experiments/probing_uniform
python3 run_probing_experiment.py --limit 5            # quick pass

# 3) Localization benchmark (floor / texture / oracle), n=100, bootstrap CIs.
python3 run_localization_benchmark.py
python3 run_localization_benchmark.py --limit 10       # quick pass
```

## Reproducibility

- **Deterministic seeds.** Per-image seeds are SHA-256-derived (`embedding.derive_image_seed`); the probe
  run seed is fixed and logged. Same seeds ⇒ identical carriers and probes.
- **Resolved config is persisted** next to every result (`metadata.json`) — runs are taken from CLI args,
  not by editing module globals.
- **Committed defaults reproduce the published experiment** (`run_probing_experiment.py` →
  `experiments/probing_only`, `texture_energy`, seed 12345). (Earlier the defaults drifted to
  `laplacian_residual` / `experiments/probing_laplacian`; fixed in Milestone 2.)
- **conseal JIT cache** is disabled at import via `embedding.prepare_conseal_import()` (a documented
  workaround); override with the `CONSEAL_DISABLE_NUMBA_CACHE` / `NUMBA_DISABLE_JIT` env vars.

## Tests

```bash
python3 -m unittest discover -s tests -v      # property tests (stdlib, no extra deps)
python3 localization_metrics.py               # metric self-tests
```
