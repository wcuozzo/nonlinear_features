# Improvement Backlog (auto-managed)

Each /loop heartbeat picks ONE item, tries it, logs the result in RUN_NOTES.md,
and marks it consumed. Tagged by category and rough effort.

**No priority by category.** Each heartbeat picks the item most likely to move the
needle given the current state. Optimization improvements can be most impactful
when there's a specific reason to believe a particular strategy will help (e.g.
high seed variance on a particular config family). Non-optimization items shine
when the recipe is mature and the bottleneck is communication or analysis depth.

## CLEAN_UP / CODE
- [code] consolidate train loops: train_stage in sweep_violation_fix.py and train_diverse_pool in frontier_push.py share ~70% of code → extract a single train_K_batched helper to a new module
- [code] move BatchedAutoencoder out of run_sweep_gpu.py into its own batched_ae.py so the import dependency is clearer
- [code] inline the wrap_up.sh logic into a Python entry point (so it's cross-platform)
- [code] add type hints to public functions in sweep_violation_fix.py, frontier_push.py, warm_start.py
- [code] write a tiny test_smoke.py that runs each pipeline stage on a single n=8 config

## VISUALIZATION / FIGURES
- [viz] m=2 highlight figure: compare with toy-models-paper theoretical n-gon arrangement (overlay)
- [viz] add titles + axis labels with proper units to all phase diagrams
- [viz] make a "best-K seeds spread" plot — for K=30 frontier-push, distribution of mse across seeds per config
- [viz] add the loss_improvement_journey scatter to README directly (it's currently a separate notebook)
- [viz] training-curve panel: pick 8 representative configs, plot loss curves side-by-side at consistent scale
- [viz] re-do the m2 trajectories with thicker lines + sharper colors + bigger panels per (n, l)
- [viz] color m2 region plot by `dominant feature MAGNITUDE` instead of identity — shows where multiple features compete

## ANALYSIS / RIGOR
- [analysis] add 95% bootstrap CIs to the precise MSEs (resample data, refit eval)
- [analysis] per-feature reconstruction error breakdown: which features get reconstructed well/poorly per config
- [analysis] compute and plot the effective rank of the encoder Jacobian, see how it relates to actual m
- [analysis] for each config, % of seeds within 1.05x of the minimum — measures how "easy" the basin is to find
- [analysis] for the scaling law, add cross-validation R² (currently train R²)
- [analysis] separately fit scaling laws per regime (linear m/n≥0.5 vs bottleneck m/n≤0.125) — may give cleaner laws
- [analysis] hunt for a cleaner scaling-law functional form. Current best is 5 params + intercept (R²=0.82, CV=0.81): log10(MSE) ≈ a·log(m/n) + b·log(1−S) + c·l + d·log(n) + e·log(m/n)·l + f. Try: (1) Chinchilla-style additive per-axis sub-laws L = (m_c/m)^α + ((1-S)/S_c)^β + γ/l + E; (2) information-theoretic / rate-distortion forms; (3) compressed-sensing-motivated compression metric m/(k·log n) instead of m/n separately + log(n); (4) per-regime fits (linear vs bottleneck) — see prior item. Standard is the 6-param fit honestly reflects genuine cross-axis interactions, but ML scaling laws are usually cleaner so worth a real hunt.

## DOCS / README
- [docs] add a "How the optimization recipe was discovered" debug-narrative section to README
- [docs] add a "Common pitfalls" section listing the bugs we fixed (floor-init, chain handoff, eval noise)
- [docs] add a one-page "results overview" markdown with all key figures inlined
- [docs] CLAUDE.md is outdated re: project structure — update to point to new canonical files

## OPTIMIZATION

Includes both signal-motivated picks AND "let's try this standard ML technique
we haven't tried yet" picks.

### Sweep / search-budget moves
- [opt] more_seeds  K=60 multi-source on top-variance configs
- [opt] fresh_random  pure random init K=100, longer training
- [opt] bigger_batch  batch_size=32k on slow-converging configs
- [opt] longer_train  3x max_steps on configs still descending
- [opt] depth_5_test  try l=5 on configs where l=4 had big improvement

### Optimizer / objective tricks
- [opt] lion  Lion optimizer instead of AdamW
- [opt] nadam  NAdam (Nesterov-momentum Adam)
- [opt] adafactor  Adafactor (memory-efficient, well-behaved on small models)
- [opt] sgd_nesterov  SGD with Nesterov momentum (sometimes finds flatter minima)
- [opt] adam_w_decoupled_only  AdamW with weight decay but no L2
- [opt] grad_clipping  clip grad norm to 1.0 — sometimes stabilizes warm-starts
- [opt] sam  Sharpness-aware minimization step
- [opt] anti_collapse  penalty on small feature norms during training
- [opt] gradient_centralization  zero-mean gradient on conv/linear weights (Yong et al. 2020)
- [opt] sophia  SOPHIA optimizer (Liu et al. 2023) — second-order light, popular in 2024

### LR schedule variants
- [opt] warm_restart  SGDR cosine with restarts
- [opt] linear_warmup_decay  linear warmup + linear decay (transformers)
- [opt] one_cycle  fastai-style one-cycle (LR up then down, momentum opposite)
- [opt] reduce_on_plateau  ReduceLROnPlateau-style (drops LR when validation flatlines)
- [opt] noam  Noam schedule (sqrt step^-1, transformer-style)

### Weight averaging / ensembling
- [opt] polyak_ema  EMA of weights instead of point estimate
- [opt] swa  Stochastic Weight Averaging (Izmailov et al.) — average snapshots near end of training
- [opt] lookahead  Lookahead optimizer (wraps AdamW)
- [opt] seed_ensemble_avg  average weights of top-K seeds at convergence (Wortsman "model soups")

### Initialization
- [opt] orthogonal_init  orthogonal init on encoder weights
- [opt] kaiming_init  Kaiming uniform/normal explicitly
- [opt] custom_init_from_pca  init encoder weights from PCA of sparse data
- [opt] scaled_residual_init  zero-out the last layer of each residual-style block

### Architectural variations (within the (n, m, l) constraint)
- [opt] layer_norm  add LayerNorm between linear layers
- [opt] residual_connections  add residual skips in encoder/decoder (l>=2 only)
- [opt] bias_init_zero  explicitly zero all biases at start
- [opt] activation_gelu  GELU instead of ReLU (then re-check non-negativity)
- [opt] activation_softplus  SoftPlus (smooth ReLU) — smoother loss landscape
- [opt] spectral_norm  spectral normalization on each linear layer
- [opt] bf16  bfloat16 mixed-precision training (faster, sometimes better generalization)

### Curriculum / data tricks
- [opt] curriculum_sparsity  start at low S, ramp up to target during training
- [opt] curriculum_n  start training at small n, gradually scale up (probably not applicable)
- [opt] knowledge_distill  use the best existing model as a teacher with soft-label loss
- [opt] mixup_features  per-batch convex combo of two samples (rare for AE but worth trying)

## CONSUMED
(items get moved here when done; with notes on outcome)
- [opt] more_seeds — first heartbeat run (in flight on box at start)
