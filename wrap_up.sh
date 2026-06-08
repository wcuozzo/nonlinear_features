#!/bin/bash
# Post-frontier-push: refresh canonical CSVs and regenerate all notebooks.
# Run on the GPU box (assumes frontier_push.py just finished there).
#
# Then sync results back to local with:
#   rsync -avz -e "ssh -p 44188" \
#       root@HOST:/root/nonlinear_features/results_db/ results_db/
#
set -euo pipefail

cd /root/nonlinear_features

echo "[wrap-up] precise recompile (re-eval all .pt at n_samples=200k)"
python3 code/precise_recompile.py --device cuda:0

echo "[wrap-up] enforce monotonicity (replace any l with identity-embed of l-1 if l > l-1)"
python3 code/enforce_monotonicity.py --device cuda:0

echo "[wrap-up] precise recompile (post-enforce)"
python3 code/precise_recompile.py --device cuda:0

echo "[wrap-up] sanity check"
python3 code/check_results.py --precise || echo "[wrap-up] sanity check flagged issues (continuing)"

cd notebooks
echo "[wrap-up] phase diagrams + scaling laws"
papermill phase_diagrams_scaling.ipynb phase_diagrams_scaling.ipynb \
          --kernel python3 --log-output 2>&1 | tail -5

echo "[wrap-up] m=2 geometry"
papermill m2_geometry.ipynb m2_geometry.ipynb \
          --kernel python3 --log-output 2>&1 | tail -5

echo "[wrap-up] loss improvement journey"
papermill loss_improvement_journey.ipynb loss_improvement_journey.ipynb \
          --kernel python3 --log-output 2>&1 | tail -5
cd ..

echo "[wrap-up] done — refresh complete"
