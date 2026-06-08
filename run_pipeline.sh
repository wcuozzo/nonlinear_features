#!/bin/bash
# Reproducible pipeline.
#
# From scratch: bootstrap → fix → push → recompile → check → notebooks
#
# Assumes CUDA-enabled box with 8 GPUs. Adjust --n-gpus and --K for your hardware.
#
# Usage:
#   ./run_pipeline.sh full        # everything from scratch
#   ./run_pipeline.sh fix         # only the violation-fix sweep
#   ./run_pipeline.sh push        # only the frontier push
#   ./run_pipeline.sh analyze     # only recompile + check + notebooks

set -euo pipefail

STORE_DIR="results_db"
N_GPUS=8
K=10
PUSH_K=30
BATCH_SIZE=8192
DEVICE="${DEVICE:-cuda:0}"
PY="${PY:-python3}"

mode="${1:-full}"

echo "Pipeline mode: $mode"
echo "Store: $STORE_DIR  GPUs: $N_GPUS  K: $K  push_K: $PUSH_K"

bootstrap_step() {
  echo "[step] bootstrap missing l=1 models"
  $PY code/bootstrap_l1.py --store-dir $STORE_DIR --n-gpus $N_GPUS --K 8
}

fix_step() {
  echo "[step] sweep violation fix"
  $PY code/sweep_violation_fix.py --store-dir $STORE_DIR --n-gpus $N_GPUS \
      --K $K --batch-size $BATCH_SIZE
}

push_step() {
  echo "[step] frontier push on suspicious configs"
  echo "(provide --configs as a space-separated list of n,m,l,S triples)"
  # The canonical list is recomputed each run via diagnostic;
  # for one-shot reproducibility, encode the original list:
  CONFIGS="128,16,2,0.85 128,64,2,0.9 128,16,2,0.95 128,64,2,0.95 \
128,64,3,0.85 128,64,3,0.9 128,64,3,0.95 128,64,4,0.85 \
128,64,4,0.9 128,64,4,0.95 16,4,2,0.9 128,2,3,0.9 128,8,3,0.9 \
128,32,3,0.9 128,32,3,0.95 128,32,4,0.85 128,32,4,0.9 128,32,4,0.95 \
16,4,3,0.85 16,2,3,0.95 16,4,3,0.9 16,2,4,0.95 \
32,2,2,0.85 32,2,3,0.85 32,4,2,0.85 32,8,2,0.85 32,8,2,0.9 32,8,3,0.85 \
32,8,3,0.9 64,16,2,0.9 64,16,4,0.9 64,16,4,0.95 64,2,3,0.9 \
64,32,2,0.95 64,32,3,0.85 64,32,3,0.9 64,32,3,0.95 \
64,32,4,0.85 64,32,4,0.9 64,32,4,0.95 \
64,8,2,0.85 64,8,2,0.9 64,8,3,0.95"
  $PY code/frontier_push.py --store-dir $STORE_DIR --n-gpus $N_GPUS \
      --K $PUSH_K --batch-size $BATCH_SIZE --configs $CONFIGS
}

analyze_step() {
  echo "[step] precise recompile (all .pt models, n_samples=200k)"
  $PY code/precise_recompile.py --store-dir $STORE_DIR --device $DEVICE

  echo "[step] comprehensive sanity check"
  $PY code/check_results.py --store-dir $STORE_DIR --precise

  echo "[step] phase diagrams + scaling laws"
  papermill notebooks/phase_diagrams_scaling.ipynb notebooks/phase_diagrams_scaling.ipynb \
            --kernel python3 --log-output 2>&1 | tail -10

  echo "[step] m=2 geometry"
  papermill notebooks/m2_geometry.ipynb notebooks/m2_geometry.ipynb \
            --kernel python3 --log-output 2>&1 | tail -10

  echo "[step] loss improvement journey"
  papermill notebooks/loss_improvement_journey.ipynb notebooks/loss_improvement_journey.ipynb \
            --kernel python3 --log-output 2>&1 | tail -10
}

case "$mode" in
  full)
    bootstrap_step
    fix_step
    push_step
    analyze_step
    ;;
  bootstrap) bootstrap_step ;;
  fix)       fix_step ;;
  push)      push_step ;;
  analyze)   analyze_step ;;
  *)
    echo "Unknown mode: $mode"
    echo "Try: full | bootstrap | fix | push | analyze"
    exit 1
    ;;
esac

echo "[pipeline] done"
