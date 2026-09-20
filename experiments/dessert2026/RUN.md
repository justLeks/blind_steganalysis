#!/usr/bin/env bash
# DESSERT-2026 revision campaign. This file is a runnable shell script kept with a .md extension so
# it renders on GitHub; run it with bash.
#
#   smoke: ROOT=experiments/dessert2026_smoke LIST=/tmp/list3.txt bash experiments/dessert2026/RUN.md
#   full:  ROOT=experiments/dessert2026 bash experiments/dessert2026/RUN.md
#
# Steganograms are embedded on first use under $ROOT/steganograms (deterministic, seed 12345) and
# reused by every later stage, so the script is resumable.
set -euo pipefail
ROOT=${ROOT:-experiments/dessert2026}
LIMIT=${LIMIT:-0}
WORKERS=${WORKERS:-8}
COVERS=${COVERS:-ALASKA_v2_TIFF_512_GrayScale_10K}
LIST=${LIST:-data/subsets/alaska10k_n1000_seed12345.txt}
METHODS="HUGO MIPOD SUNIWARD"
ALPHAS="0.001 0.0025 0.005 0.01 0.02 0.03 0.05 0.1 0.2 0.5"
BUDGETS="0.005 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5"
mkdir -p "$ROOT"
for DIST in texture_energy uniform; do
  python3 run_probing_experiment.py --cover-root "$COVERS" --cover-list "$LIST" \
    --methods $METHODS --alphas $ALPHAS --budgets $BUDGETS --repeats 10 --workers "$WORKERS" --limit "$LIMIT" \
    --distribution "$DIST" --experiment-root "$ROOT/probing_$DIST" --steganogram-root "$ROOT/steganograms" \
    2>&1 | tee "$ROOT/probing_$DIST.log"
done
python3 run_localization_benchmark.py --steganogram-root "$ROOT/steganograms" --cover-root "$COVERS" \
  --cover-list "$LIST" --methods $METHODS --alphas $ALPHAS --budgets $BUDGETS --workers "$WORKERS" --limit "$LIMIT" \
  --output-root "$ROOT/benchmark" 2>&1 | tee "$ROOT/benchmark.log"
python3 run_paired_tests.py --benchmark "$ROOT/benchmark" --out "$ROOT/paired_tests" 2>&1 | tee "$ROOT/paired.log"
