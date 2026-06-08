# Project summary at a glance

Detailed README in `README.md`. Detailed timestamped notes in `RUN_NOTES.md`.

## TL;DR

This project sweeps `(n, m, l, S)` autoencoders and pushes every config to (or near) its true minimum, then analyzes the resulting landscape.

**Key methodological result**: a particular training recipe — progressive `l=1 → l=4` chain with batched K=30 seeds, gradient clipping, 3× steps, optionally EMA — beats the original random-init sweep by up to **469×** on individual configs and gives full monotonicity (0 violations) across all 4 axes.

**Key empirical result**: a clean per-regime scaling law for MSE in the bottleneck regime:
```
log10(MSE) ≈ −0.10·log(m/n) + 1.24·log(1−S) − 0.30·l − 0.16·log(m/n)·l − 0.39
R² = 0.95, CV R² = 0.93  (90 bottleneck-regime configs)
```

**Key conceptual result**: depth doesn't add directional rank — `l=2` already finds the optimal direction arrangement (regular n-gon, matching Elhage et al. Toy Models). Going deeper INCREASES magnitude superposition: features collapse into shared directions at different magnitudes.

## Canonical training recipe

```bash
python frontier_push.py --n-gpus 8 --K 30 --batch-size 8192 \
    --grad-clip 1.0 --steps-mult 3.0 --configs <list>
```

Optionally add `--ema-decay 0.999` for Polyak weight averaging.

## Pipeline

```
bootstrap_l1.py → sweep_violation_fix.py → frontier_push.py → precise_recompile.py → check_results.py
```

Or single command: `./run_pipeline.sh full`

## Key figures

- `fig_phase_diagram_mse.png` — MSE heatmap over (n, m) at each (l, S)
- `fig_phase_diagram_gain.png` — nonlinear gain heatmap
- `fig_loss_vs_compression.png` — MSE vs compression curves per (l, S)
- `fig_scaling_per_regime.png` — predicted vs observed scaling law per regime
- `fig_scaling_pareto.png` — simplicity-vs-accuracy Pareto for scaling laws
- `fig_highlight_m2_n16_S0.95.png` — clean m=2 feature geometry across depths
- `fig_m2_vs_theoretical_ngon.png` — learned features vs Elhage n-gon overlay
- `fig_basin_easiness.png` — bimodal distribution of basin difficulty
- `fig_per_feature_mse.png` — per-feature reconstruction Gini histograms
- `fig_effective_rank.png` — effective rank of encoder Jacobian vs m
- `fig_K_seeds_spread.png` — K=30+ seed distribution per config
- `fig_technique_comparison.png` — optimization techniques side-by-side
- `fig_improvement_histogram.png` — cumulative loss reduction vs original baseline

## Files

- **Canonical code**: `core.py`, `results_store.py`, `warm_start.py`, `sweep_violation_fix.py`, `frontier_push.py`, `bootstrap_l1.py`, `precise_recompile.py`, `check_results.py`
- **Analysis notebooks**: `phase_diagrams_scaling.ipynb`, `m2_geometry.ipynb`, `loss_improvement_journey.ipynb`
- **Data**: `results_db/` (seeds + models + canonical CSV)
- **Notes**: `RUN_NOTES.md` (timestamped improvement iterations), `lessons-learned.md` (rigor standards), `improvement_backlog.md` (idea queue)

## Numbers at a glance

- 216 configs in the sweep (`n ∈ {16,32,64,128}, m ∈ {2,4,8,16,32,64}, l ∈ {1,2,3,4}, S ∈ {0.85,0.9,0.95}`)
- 186 saved best `.pt` models (some `m=n` excluded)
- 0 monotonicity violations under precise eval at n=200k
- All 4 axes (depth / bottleneck / input-dim / sparsity) clean
- Bootstrap 95% CI: median relative width 2.3%, all apparent residual violations within CI overlap
