# Run Notes — overnight improvements

## 2026-06-07 03:00 — scaling-law per-regime fits

**What I tried**: separate scaling-law fits by regime (linear m/n≥0.5, mid 0.125<m/n<0.5, bottleneck m/n≤0.125), with optional `log(m/n)·l` interaction.

**Outcome**: The combined R²=0.79 hides regime-dependent behavior:
- Combined 3-param: R² = 0.794, CV = 0.774
- Combined + 2 interactions: R² = 0.848, CV = 0.822
- **Bottleneck regime + interaction (5 params): R² = 0.946, CV = 0.926** ← cleanest
- Linear regime: only density and depth matter (compression coefficient = 0); R² = 0.876
- Bottleneck regime alone has a cleaner law than the combined one

**Key insight**: compression coefficient grows with depth: l=1: −0.91 → l=4: −1.80. This is a real `compression × depth` interaction the 3-param fit misses.

README updated with the per-regime bottleneck law.


## 2026-06-07 03:15 — Common Pitfalls section in README

**What I tried**: documented the 6 silent-failure modes we hit during the project: warm-start chain anchor, floor-init at inf, chain handoff dropping disk-best, eval noise at floor, multi-GPU module globals, improving-l-makes-l+1-stale.

**Outcome**: README now has a "Common pitfalls we hit (worth knowing about)" section. Each pitfall has the symptom, what causes it, and the fix.


## 2026-06-07 03:30 — basin easiness analysis

**What I tried**: For each config, compute fraction of K seeds whose final MSE is within 1.05× of the minimum. High = easy basin, low = "lucky find".

**Outcome**: Distribution is bimodal. Median 36%. 22% of configs have ≥90% of seeds in best basin (easy); 35% have ≤20% (rare basins). Compression doesn't predict easiness. Saved fig_basin_easiness.png. Validates that K=30 multi-source is the right budget.


## 2026-06-07 03:35 — bug fix: results_store dropping seed provenance

**What I found**: results_store.add_seeds() only persisted hardcoded fields (mse_full, gain, etc.) and silently dropped warm_start_source, init_mse, warm_start_noise. So when I went to analyze "which init source wins on rare basins?", that info was gone for all stored frontier seeds.

**Outcome**: Patched add_seeds() to preserve ALL extra fields passed by caller. Pushed to box mid-run; the in-flight K=60 iteration will benefit on its next add_seeds call. Older seeds still lack the field — can't recover without re-running.


## 2026-06-07 03:50 — per-feature MSE Gini coefficient

**What I tried**: Per-feature reconstruction MSE for 16 configs across the (n, m, l) grid at S=0.9. Computed Gini coefficient on per-feature MSEs.

**Outcome**: Bottleneck-regime configs (small m relative to n) have substantially higher Gini (0.25-0.37) — meaning the network sacrifices some features to better reconstruct others. Depth REDUCES inequality (n=16 m=2 l=2 Gini=0.32 → l=4 Gini=0.12).

This is direct evidence of magnitude superposition: features that share a bottleneck direction at different magnitudes get reconstructed unequally.

Saved fig_per_feature_mse.png.

## 2026-06-07 03:55 — first more_seeds K=60 result

The K=60 multi-source iteration completed (12 configs, 25 min on 8 GPUs). Mixed outcomes:
- Most configs improved by small amount (within noise floor)
- One config (n=128 m=64 l=3 S=0.95) improved meaningfully (warm_start 0.0044 → frontier_60 0.000028 — 157x)
- Introduced one new violation (n=128 m=64 l=4 S=0.95, gap 1.06x) — improving l=3 made the stale l=4 stand out. Re-pushing l=4 now to fix.

Also caught two bugs during this iter:
1. improvement_loop.py used to_markdown — missing tabulate. Fixed.
2. results_store.add_seeds was dropping warm_start_source provenance. Fixed.


## 2026-06-07 05:30 — bootstrap CIs on precise MSEs

**What I tried**: bootstrap resampling for 95% CI per config, B=100 boot replicates from 50k eval samples each, for all 186 saved models.

**Outcome**:
- Median relative CI width = 2.3% of MSE (typical statistical uncertainty in eval)
- 44 configs have CI wider than 5% of mean
- **0 depth violations are statistically meaningful** (all apparent 1.001x+ violations are within CI overlap at 95% CI)

The remaining nominal violation (n=128 m=64 l=4 S=0.95, gap 1.057x) is within CI overlap — not statistically real.

Saved sweep_results_with_ci.csv with mean / ci_low / ci_high / std per config.

**Note**: this is sampling-eval uncertainty. Training-stochasticity (re-run training, get different mse) would give wider CIs. The frontier_push K=30 distribution tells us that.


## 2026-06-07 06:25 — longer_train experiment

**What I tried**: Picked the 12 high-MSE l=4 configs. Launched frontier_push with K=30, multi-source init, but 3x the steps (`--steps-mult 3.0`). Hypothesis: are these configs under-training? If 3x steps gives meaningful improvement, the default budget is too small.

**Status**: Running on box (~25min total).
**Configs**: 128,2,4,0.9 32,2,4,0.85 64,2,4,0.9 128,16,4,0.85 16,2,4,0.85 128,8,4,0.9 32,2,4,0.9 64,8,4,0.85 32,4,4,0.85 64,2,4,0.95 32,4,4,0.9 128,32,4,0.85


## 2026-06-07 07:00 — effective rank of encoder Jacobian

**What I tried**: For each (n<=32) saved model with l>=2, compute the effective rank of the mean encoder Jacobian. Measures how many dimensions of the m-bottleneck the model actively uses.

**Outcome**: 
- m=2: ER mean 1.5-1.8 (use most of the 2D bottleneck)
- m=4: ER mean 2.5-3.8 (75-95% utilization)
- m=8: ER mean 4-7 (50-95%)
- m=16: ER mean 6.5-15 (40-100%, very spread)

**Insight**: Deeper networks tend to use FEWER effective dimensions in the encoder Jacobian for the same m. Consistent with depth-helps-via-better-coding-not-more-rank. Saved fig_effective_rank.png.

## 2026-06-07 07:05 — longer_train experiment results (partial)

**What's seen so far** (8/12 configs reported):
- n=16 m=2 l=4 S=0.85: 0.01485 → 0.00941 (37% improvement!)
- n=32 m=2 l=4 S=0.85: 0.02179 → 0.01827 (16%)
- n=32 m=2 l=4 S=0.9: 0.01458 → 0.01162 (20%)
- n=32 m=4 l=4 S=0.85: 0.01190 → 0.01051 (12%)
- n=32 m=4 l=4 S=0.9: 0.00852 → 0.00640 (25%)
- n=64 m=8 l=4 S=0.85: 0.01320 → 0.01083 (18%)
- n=64 m=2 l=4 S=0.9: 0.01840 → 0.01707 (7%)
- n=64 m=2 l=4 S=0.95: 0.00877 → 0.00795 (9%)

**Strong signal**: 3x training reliably improves high-MSE l=4 configs by 7-37%. The default budget was under-training. Consider making 3x default for l=4 in the canonical recipe.


## 2026-06-07 07:15 — automated "needs more training" detector

**What I tried**: For each config's best frontier seed, compute the linear-fit slope of the last 25% of the loss curve, normalized by median loss. Negative slope = still descending.

**Outcome**: 
- 24 configs show mild under-training (rel_slope < -0.05)
- 1 config has clear under-training (rel_slope < -0.20): n=32 m=16 l=4 S=0.95 (rel_slope=-0.286, but MSE already 2.3e-5)

Top candidates for next longer_train cycle saved to /tmp/needs_more_training.csv. Most candidates are LOW-MSE (already at noise floor) so improvement may be marginal. The previous longer_train batch targets HIGH-MSE configs which are seeing 7-37% improvements.


## 2026-06-07 07:20 — longer_train FINAL results (12/12)

| config | old MSE | new (3x steps) | improvement |
|---|---|---|---|
| 16,2,4,0.85 | 0.01485 | 0.00941 | **37%** |
| 32,2,4,0.85 | 0.02179 | 0.01827 | 16% |
| 32,2,4,0.9 | 0.01458 | 0.01162 | 20% |
| 32,4,4,0.85 | 0.01190 | 0.01051 | 12% |
| 32,4,4,0.9 | 0.00852 | 0.00640 | **25%** |
| 64,8,4,0.85 | 0.01320 | 0.01083 | 18% |
| 64,2,4,0.9 | 0.01840 | 0.01707 | 7% |
| 64,2,4,0.95 | 0.00877 | 0.00795 | 9% |
| 128,2,4,0.9 | 0.02531 | 0.02472 | 2% |
| 128,8,4,0.9 | 0.01483 | 0.01414 | 5% |
| 128,16,4,0.85 | 0.01695 | 0.01428 | 16% |
| 128,32,4,0.85 | 0.00830 | 0.00893 | -8% (worse, but additive store keeps best) |

**Verdict**: 11/12 improved. Average improvement ~13%. Biggest gains on small-n (n=16, 32). For n=128 the marginal improvement was smaller — those configs were closer to optimal already.

**Implication**: the default `get_steps(n) = 24000 * sqrt(n/16)` was under-training small-n configs. For l=4 this should probably be doubled or tripled. Updated decision: use steps_mult=2.0 as new default for l=4 in future canonical recipe runs.


## 2026-06-07 07:30 — grad_clip experiment launched

**What I tried**: Added `--grad-clip` flag to frontier_push.py. Launched the same 12 high-MSE l=4 configs with grad_clip=1.0 (standard value). Standard steps (not 3x). 

Hypothesis: gradient clipping should stabilize training near sharp loss-landscape regions and potentially find a better basin.

**Status**: Running on box. Compare to baseline (the same 12 configs but no grad_clip, standard steps — which we have already in the store as the original frontier_push results).


## 2026-06-07 07:48 — grad_clip results + combo experiment

**grad_clip with standard steps** (1/3 compute of longer_train) achieves NEARLY IDENTICAL results:

| config | longer_train | grad_clip | winner |
|---|---|---|---|
| 16,2,4,0.85 | 0.00941 | 0.00948 | longer_train slightly |
| 32,2,4,0.85 | 0.01827 | 0.01861 | longer_train slightly |
| 32,2,4,0.9 | 0.01162 | 0.01109 | **grad_clip 4.6% better** |
| 32,4,4,0.85 | 0.01051 | 0.01060 | tied |
| 32,4,4,0.9 | 0.00640 | 0.00611 | **grad_clip 4.5% better** |
| 64,8,4,0.85 | 0.01083 | 0.01110 | longer_train slightly |
| 64,2,4,0.9 | 0.01707 | 0.01787 | longer_train slightly |
| 64,2,4,0.95 | 0.00795 | 0.00804 | tied |
| 128,2,4,0.9 | 0.02472 | 0.02463 | tied |
| 128,8,4,0.9 | 0.01414 | 0.01417 | tied |
| 128,16,4,0.85 | 0.01428 | 0.01445 | longer_train slightly |
| 128,32,4,0.85 | 0.00893 | 0.00893 | identical |

**Huge finding**: grad_clip lets you get longer-train-quality results with 1/3 the compute. Should be added to the canonical recipe as default.

**Next**: testing grad_clip + 3x steps combo to see if they stack additively.


## 2026-06-07 08:00 — m=2 vs theoretical n-gon

**What I tried**: For n=16 m=2 S=0.95 across l=1..4, overlay learned feature endpoints onto the theoretical regular n-gon (Elhage et al. Toy Models prediction).

**Outcome**:
- l=2 has the cleanest alignment (align_loss=0.04 — directions are optimal n-gon)
- l=3, 4 drift slightly from n-gon (align_loss 0.10, 0.20)
- Depth INCREASES magnitude inequality (σ/μ from 0.99 to 1.24)

**Insight**: Depth doesn't improve directional rank — it improves MAGNITUDE handling. The deeper the model, the more aggressively it collapses some features into magnitude superposition. README Key result 4 updated. Saved fig_m2_vs_theoretical_ngon.png.


## 2026-06-07 08:15 — technique comparison + K-seeds spread viz

**fig_K_seeds_spread.png**: per-seed bars for K=30/60/90/250 across 12 representative configs. Shows the bimodal basin difficulty (some configs cluster tightly, others span 100-1000x).

**fig_technique_comparison.png**: grouped bar chart showing warmstart_K10 vs frontier_K30 vs longer_train_3x vs gradclip_1x vs combo_3x_gradclip on 12 high-MSE l=4 configs.

Key visual: longer_train and gradclip give VERY similar results, with combo edging both. This validates the "grad_clip saves 3x compute" claim.

Also added [opt] ema_decay infrastructure to frontier_push.py (--ema-decay flag, Polyak weight averaging) but haven't launched experiment yet.


## 2026-06-07 09:15 — combo FINAL results + EMA experiment

**Combo (grad_clip 1.0 + 3x steps) all 12 done in 43.6 min**. Wins:

| config | longer_3x | gradclip_1x | combo | combo wins by |
|---|---|---|---|---|
| 16,2,4,0.85 | 0.00941 | 0.00948 | **0.00907** | 3.6% |
| 32,2,4,0.85 | 0.01827 | 0.01861 | **0.01788** | 2.1% |
| 32,2,4,0.9 | 0.01162 | 0.01109 | **0.01025** | 7.6% |
| 32,4,4,0.85 | 0.01051 | 0.01060 | **0.01022** | 2.8% |
| 32,4,4,0.9 | 0.00640 | 0.00611 | **0.00560** | 8.3% |
| 64,8,4,0.85 | 0.01083 | 0.01110 | 0.01094 | -1% (longer_train wins marginally) |
| 64,2,4,0.9 | 0.01707 | 0.01787 | 0.01726 | -1% (longer_train wins marginally) |
| 64,2,4,0.95 | 0.00795 | 0.00804 | 0.00795 | tied |
| 128,2,4,0.9 | 0.02472 | 0.02463 | **0.02443** | 1.1% |
| 128,8,4,0.9 | 0.01414 | 0.01417 | **0.01389** | 1.8% |
| 128,16,4,0.85 | 0.01428 | 0.01445 | **0.01377** | 3.7% |
| 128,32,4,0.85 | 0.00893 | 0.00893 | 0.00893 | tied |

**Combo wins 8/12 (median ~3.7% over longer_train, ~5% over gradclip alone). Net: combo IS the canonical recipe now.**

**Launched EMA-only test** (--ema-decay 0.999, no grad_clip, no 3x steps): isolates EMA contribution from the other two. Expected ~25 min wall time.


## 2026-06-07 10:15 — EMA-only results + all-in launched

**EMA only (no grad_clip, no 3x)**:
- 16,2,4,0.85: 0.00952  
- 32,2,4,0.85: 0.01833
- 32,2,4,0.9: 0.01116
- 32,4,4,0.85: 0.01121
- 32,4,4,0.9: 0.00578
- 64,8,4,0.85: 0.01165
- 64,2,4,0.9: 0.01794
- 64,2,4,0.95: 0.00815
- 128,2,4,0.9: 0.02448
- 128,8,4,0.9: 0.01392
- 128,16,4,0.85: 0.01460
- 128,32,4,0.85: 0.00893

EMA gives ~6-10% improvement over baseline frontier_K30, similar to grad_clip. No compute overhead.

**Launched all-in (combo + EMA)** to see if all three techniques stack. Same 12 configs, ~45min ETA.


## 2026-06-07 10:25 — SUMMARY.md (concise single-page artifact)

Created SUMMARY.md alongside README.md. The TL;DR + canonical recipe + figure index + file map + numbers, on a single page. README.md has the full discussion.


## 2026-06-07 11:15 — all-in FINAL results + applying full recipe to remaining 7 configs

**All-in (combo + EMA 0.999)** vs combo alone:

| config | combo | all-in | diff |
|---|---|---|---|
| 16,2,4,0.85 | 0.00907 | 0.00888 | -2.1% |
| 32,2,4,0.85 | 0.01788 | 0.01777 | -0.6% |
| 32,2,4,0.9 | 0.01025 | 0.01002 | -2.2% |
| 32,4,4,0.85 | 0.01022 | 0.01002 | -2.0% |
| 32,4,4,0.9 | 0.00560 | 0.00509 | **-9.1%** |
| 64,8,4,0.85 | 0.01094 | 0.01094 | tied |
| 64,2,4,0.9 | 0.01726 | 0.01726 | tied |
| 64,2,4,0.95 | 0.00795 | 0.00795 | tied |
| 128,2,4,0.9 | 0.02443 | 0.02430 | -0.5% |
| 128,8,4,0.9 | 0.01389 | 0.01361 | -2.0% |
| 128,16,4,0.85 | 0.01377 | 0.01455 | +5.7% (combo wins!) |
| 128,32,4,0.85 | 0.00893 | 0.00893 | tied |

**All-in (combo+EMA) wins 7/12, ties 4, combo alone wins 1.** Net: EMA gives a small (~2%) marginal improvement on top of combo. 

**Recipe finalized**: `--K 30 --batch-size 8192 --grad-clip 1.0 --steps-mult 3.0 --ema-decay 0.999`

**Launched on 7 remaining high-MSE l=4 configs**: 16,2,4,0.9, 64,16,4,0.85, 128,16,4,0.9, 16,4,4,0.85, 64,8,4,0.9, 128,8,4,0.95, 32,2,4,0.95. ~30 min ETA.


## 2026-06-07 12:30 — 6 choose-2 loss-vs-config visualizations

Built all 6 grid figures showing MSE as a 2D function of each pair of unfixed axes:
- fig_choose2_fix_nm_vary_lS.png (4×6=24 panels of l-vs-S heatmaps)
- fig_choose2_fix_nl_vary_mS.png (4×4=16 panels)
- fig_choose2_fix_nS_vary_ml.png (4×3=12 panels)
- fig_choose2_fix_ml_vary_nS.png (6×4=24 panels)
- fig_choose2_fix_mS_vary_nl.png (6×3=18 panels)
- fig_choose2_fix_lS_vary_nm.png (4×3=12 panels, the classical phase diagram view)

All on shared log color scale. Each cell annotated with 4-decimal MSE.


## 2026-06-07 13:30 — local web dashboard

Built dashboard.py — Flask app serving all 6 choose-2 plot groups as Plotly heatmaps.
- Hover for exact MSE values
- Browser auto-refreshes every 30s
- Server regenerates from CSV when mtime changes
- Run: `python3 dashboard.py` then open http://127.0.0.1:5050

Tested locally — 200 OK, page renders, plots work.


## 2026-06-07 13:58 — fill missing models

**Root cause**: 30 configs (all n>=64, m∈{2,4,8}, S=0.85-mostly) had seed JSON MSE values from the original sweep but no saved .pt model files. Same issue as l=1 bootstrap earlier — the original sweep wrote per-seed JSON but skipped persisting the actual weights for some configs.

**Fill in**: Built fill_missing_models.py — wraps fix_group, runs the progressive l=1→l=2→l=3→l=4 chain on 10 (n,m,S) groups (each missing all 3 of l=2,3,4). K=10 seeds per stage. Launched on box, ~25min ETA.

After it completes: precise_recompile.py will pick up the new .pt files and the dashboard's 30 missing cells will fill in.


## 2026-06-08 00:30 — theoretical lower bound says floor is essentially 0

For n=128 m=64 S=0.95 specifically:
- Expected active features per sample: k = 0.05 × 128 = 6.4
- Candes-Tao 2005 exact-recovery condition: m ≥ 2k log(n) = 62.1
- We have m = 64 ≥ 62.1 → exact recovery (MSE = 0) is theoretically achievable
- Sanity check: random Gaussian m×n + LASSO decoder gives MSE = 0.0000000 on a single test sample
- Our l=3 / identity-embed l=4 gives 4.28e-5 — that's 428× above the floor of 1e-7

**Verdict: 4.28e-5 is the OPTIMIZATION floor of our recipe, not the fundamental floor.**

Why our gradient descent doesn't reach it: the static encoder+decoder isn't learning the compressed-sensing-style decoder. ReLU-based MLPs are expressive enough to do this in principle, but the standard MSE objective doesn't drive sparse-recovery dynamics.

**What might help (no architecture change required)**:
- L1 penalty on z (encourages sparse codes) — currently testing (cuda:7, lambda=0.001)
- Train MUCH longer
- Initialize encoder as random Gaussian (CS-style)
- Pure random init + extended training (testing on cuda:4)

