#!/bin/bash
# Single canonical sweep into a fresh, dated, self-contained results directory.
# Every (n, m, l, S) trained from scratch with the same recipe (near-warm-start
# arm enabled) and tagged with one master seed + one run_id.
#
# Usage:
#   ./canonical_sweep.sh                   # default seed=42, dated dir
#   MASTER_SEED=7 ./canonical_sweep.sh     # custom seed
#   STORE_DIR=my_run/ ./canonical_sweep.sh # custom output dir
#
# Output: $STORE_DIR/  with models/ + seeds/ + compiled/sweep_results_precise.csv

set -euo pipefail

DATE_TAG="$(date +%Y-%m-%d)"
MASTER_SEED="${MASTER_SEED:-42}"
STORE_DIR="${STORE_DIR:-results_canonical_sweep_${DATE_TAG}_seed${MASTER_SEED}}"
RUN_ID="canonical_${DATE_TAG}_seed${MASTER_SEED}"
N_GPUS=${N_GPUS:-8}
K=${K:-10}
PUSH_K=${PUSH_K:-30}
BATCH_SIZE=${BATCH_SIZE:-8192}
NWS_K=${NWS_K:-5}
NWS_LR_MULT=${NWS_LR_MULT:-0.3}
DEVICE="${DEVICE:-cuda:0}"
PY="${PY:-python3}"

# Whether to also run a frontier-push refinement pass over all l=4 configs
# (adds ~1 hour on 8x A100). Set to 0 to skip.
DO_FRONTIER_PUSH=${DO_FRONTIER_PUSH:-1}

echo "============================================================"
echo "Canonical sweep"
echo "  store_dir:      $STORE_DIR"
echo "  master_seed:    $MASTER_SEED"
echo "  run_id:         $RUN_ID"
echo "  n_gpus:         $N_GPUS"
echo "  K (per stage):  $K"
echo "  K_nws:          $NWS_K  (lr_mult $NWS_LR_MULT)"
echo "  push_K:         $PUSH_K"
echo "  frontier_push:  $DO_FRONTIER_PUSH"
echo "============================================================"

mkdir -p "$STORE_DIR"

# -------- 1. Bootstrap l=1 for every (n, m, S) --------
echo
echo "[1/5] Bootstrap l=1 for every (n, m, S)"
$PY code/bootstrap_l1.py --store-dir "$STORE_DIR" --n-gpus $N_GPUS --K $K \
    --batch-size $BATCH_SIZE --master-seed $MASTER_SEED --all-groups

# -------- 2. Progressive l=1 -> l=4 chain on every group with near-warm-start --------
echo
echo "[2/5] Progressive chain (l=1 -> l=4) on every (n, m, S) group, with near-warm-start arm"
$PY code/sweep_violation_fix.py --store-dir "$STORE_DIR" --n-gpus $N_GPUS \
    --K $K --batch-size $BATCH_SIZE \
    --master-seed $MASTER_SEED --run-id "$RUN_ID" \
    --near-warm-start-K $NWS_K --near-warm-start-lr-mult $NWS_LR_MULT \
    --all-groups

# -------- 3. (Optional) Frontier-push on every l=4 config --------
if [ "$DO_FRONTIER_PUSH" = "1" ]; then
  echo
  echo "[3/5] Frontier-push K=$PUSH_K + near-warm-start on every l>=2 config"
  # Build the full config list (n, m, l, S) with m < n
  CONFIGS=$($PY -c "
ns = [16, 32, 64, 128]
ms = [2, 4, 8, 16, 32, 64]
ls = [2, 3, 4]
Ss = [0.85, 0.9, 0.95]
out = []
for n in ns:
  for m in ms:
    if m >= n: continue
    for l in ls:
      for S in Ss:
        out.append(f'{n},{m},{l},{S}')
print(' '.join(out))
")
  $PY code/frontier_push.py --store-dir "$STORE_DIR" --n-gpus $N_GPUS \
      --K $PUSH_K --batch-size $BATCH_SIZE \
      --grad-clip 1.0 --steps-mult 3.0 --ema-decay 0.999 \
      --master-seed $MASTER_SEED \
      --near-warm-start-K $NWS_K --near-warm-start-lr-mult $NWS_LR_MULT \
      --configs $CONFIGS
else
  echo
  echo "[3/5] Frontier-push: SKIPPED (DO_FRONTIER_PUSH=0)"
fi

# -------- 4. Precise re-eval at n=200k -> canonical CSV --------
echo
echo "[4/5] Precise recompile (n_samples=200k) + enforce monotonicity"
$PY code/precise_recompile.py --store-dir "$STORE_DIR" --device $DEVICE
$PY code/enforce_monotonicity.py --store-dir "$STORE_DIR" --device $DEVICE
$PY code/precise_recompile.py --store-dir "$STORE_DIR" --device $DEVICE

# -------- 5. Sanity check --------
echo
echo "[5/5] Sanity check"
$PY code/check_results.py --store-dir "$STORE_DIR" --precise \
    || echo "[canonical] sanity check flagged issues (continuing)"

echo
echo "============================================================"
echo "Canonical sweep complete."
echo "Canonical results:  $STORE_DIR/compiled/sweep_results_precise.csv"
echo "Models:             $STORE_DIR/models/  ($(ls $STORE_DIR/models/ 2>/dev/null | wc -l) files)"
echo "============================================================"
