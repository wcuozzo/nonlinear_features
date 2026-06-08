# Results Log

## 2026-03-09: z(0) Constraint Experiment

**Question**: Are the bias terms (where z(0) ends up) doing anything? Could they do anything even in theory?

**Method**: Compared autoencoders with vs without encoder biases across 7 configurations. Without encoder biases, z(0) = 0 is guaranteed (ReLU(W @ 0) = 0). With encoder biases, z(0) can be anywhere in latent space.

### Configurations Tested

| n | m | l | S | Effect Size | Diff % | p-value |
|---|---|---|---|-------------|--------|---------|
| 5 | 2 | 2 | 0.9 | negligible (d=-0.07) | -6.21% | 0.7892 |
| 5 | 2 | 4 | 0.9 | negligible (d=0.07) | +5.41% | 0.7813 |
| 10 | 3 | 2 | 0.9 | small (d=0.36) | +19.27% | 0.1848 |
| 10 | 5 | 2 | 0.9 | medium (d=-0.51) | -32.09% | 0.0704 |
| 10 | 3 | 4 | 0.9 | small (d=-0.43) | -18.29% | 0.1169 |
| 5 | 2 | 2 | 0.95 | negligible (d=-0.13) | -11.11% | 0.6327 |
| 10 | 3 | 2 | 0.95 | negligible (d=0.15) | +7.57% | 0.5693 |

### Summary Statistics

- **Configurations tested**: 7
- **Effect sizes**: 4 negligible, 2 small, 1 medium
- **Statistically significant**: 0 (none at p < 0.05)
- **Average difference**: -5.07%
- **Average best-model difference**: +12.94%

### Key Observations

1. **z(0) locations are non-trivial**: Models with encoder bias place z(0) at non-zero locations (mean norm 0.4-1.5, max norm up to 3.4), showing the bias freedom IS being used.

2. **No consistent performance advantage**: The direction of the effect flips between configurations (sometimes with-bias is better, sometimes without-bias is better).

3. **Best-model comparisons also mixed**: When comparing the best model from each condition (what you'd do in practice), results are similarly inconsistent.

4. **Effect sizes are small**: Most configurations show negligible or small Cohen's d values.

### Conclusion

**The z(0) degree of freedom (encoder biases) has no consistent practical impact on autoencoder performance.**

While models with encoder biases DO place z(0) at non-zero locations (the freedom is being used), this doesn't translate to better reconstruction performance. The nonlinearity in these autoencoders does NOT fundamentally depend on shifting the origin in latent space.

**Theoretical interpretation**: The benefits of depth come from input-dependent transformations (how features interact and get encoded differently based on context), not from global offsets in latent space. The z(0) location is essentially a "free" degree of freedom that optimization can use however it wants, but it doesn't provide representational benefits.

**Practical implication**: You can constrain z(0) = 0 without hurting model performance. This could be useful for interpretability (knowing exactly where the "baseline" is in latent space).

---

## 2026-03-10: Methodology Improvements (Reanalysis Pending)

**Problem**: The original z(0) analysis had methodological gaps:
1. Optimization variance (~100x) dominated by failed runs, not properly handled
2. Outlier exclusion used both tails (wrong - failures are one-tailed)
3. Best-of-k comparison noisy, no bootstrap CIs
4. Visual observations (arc/chord differences) not quantified
5. Training curve differences not investigated

**Improvements Implemented** (in `z0_constraint_exploration.ipynb`):

### Unified Quantile Analysis
- Replaced ad-hoc outlier exclusion + best-of-k with quantile progression
- Reports at top-1, top-3, top-25%, median, all (mean)
- Bootstrap CIs at each quantile level
- Interpretation: where does the effect appear/disappear in the distribution?

### Empirical Minimum Scaling
- Plot E[min | k] vs k for both conditions
- Hierarchical subsampling to observe scaling at multiple k values
- Shows whether conditions converge or diverge as you run more seeds

### Training Curve Analysis
- Grid plot of individual training curves
- Auto-classification: smooth_converged, bumpy, plateau_escape, failed, oscillating
- Chi-square test for category distribution differences

### Arc/Chord Ratio Quantification
- Measures trajectory curvature: arc_length / chord_length
- 1 = perfectly linear, >1 = curved
- Mann-Whitney U test for distribution comparison
- Separate analysis for best models

### Comprehensive Summary
- Multi-pronged verdict combining all analyses
- Confidence level based on convergent evidence

**Status**: Analysis complete.

### Results (n=5, m=2, l=2, S=0.9, 20 seeds)

#### 1. Quantile Progression
| Quantile | With Bias | Without Bias | Diff | Significant? |
|----------|-----------|--------------|------|--------------|
| top_1 | 0.00215 | 0.00217 | +0.000016 | NO |
| top_3 | 0.00249 | 0.00236 | -0.000136 | NO |
| top_25pct | 0.00371 | 0.00266 | -0.001050 | NO |
| median | 0.01061 | 0.00725 | -0.003352 | NO |
| all (mean) | 0.01465 | 0.00930 | -0.005348 | NO |

**Key insight**: No significant difference at ANY quantile level. The naive mean comparison (without appears 37% better) is misleading - at the frontier (best models), the difference is <1%.

#### 2. Minimum Scaling
| k | E[min|k] with | E[min|k] without | Diff |
|---|---------------|------------------|------|
| 1 | 0.0147 | 0.0093 | -0.0053 |
| 5 | 0.0037 | 0.0029 | -0.0008 |
| 10 | 0.0024 | 0.0023 | -0.00005 |
| 20 | 0.00215 | 0.00217 | +0.00002 |

**Key insight**: Curves converge as k increases. At k=20, difference is +0.7% - essentially identical at the frontier.

#### 3. Training Dynamics
| Category | With Bias | Without Bias |
|----------|-----------|--------------|
| bumpy | 20 | 20 |
| failed | 0 | 0 |

**Key insight**: All runs classified as "bumpy" (converged with some noise). No failures in either condition - we're comparing successful training runs.

#### 4. Arc/Chord Ratio
| Condition | Mean Ratio | Std |
|-----------|------------|-----|
| With bias | 1.0202 | 0.0452 |
| Without bias | 1.0000 | 0.0000 |

Mann-Whitney U test: p < 0.0001 (significant)

**Key insight**: Statistically significant but practically negligible. Without-bias trajectories are perfectly linear (z(0)=0 forces this). With-bias trajectories are only 2% more curved on average.

### Critical Correction (2026-03-11)

The above analysis was **overconfident**. Re-examination reveals:

#### Problem 1: Insufficient Statistical Power
The 95% CIs are ~10x wider than the point estimates:
- top_1: observed +6.3%, CI spans [-185%, +10%]
- median: observed -1.5%, CI spans [-71%, +42%]

**"Not significant" means "we don't know", not "no effect"**. We'd need ~400+ seeds to bound effects to ±5%.

#### Problem 2: Results Depend on Training Parameters
- With n_steps=20000: 0 failures in both conditions
- With n_steps=10000: 0 failures with-bias, 3 failures without-bias

The "identical training dynamics" claim was wrong.

#### Problem 3: Arc/Chord Ratio = 1.0 Exactly Demands Explanation

**Key theoretical insight**: ReLU networks without biases are **positively homogeneous**: f(t·x) = t·f(x) for t > 0.

Proof: For encoder without biases:
```
z(t·e_i) = W @ ReLU(... ReLU(t · W1[:,i])...)
         = W @ ReLU(... t · ReLU(W1[:,i])...)  # ReLU(t·x) = t·ReLU(x) for t>0
         = t · W @ ReLU(... ReLU(W1[:,i])...)
         = t · z(e_i)
```

This means **removing biases doesn't just change z(0) - it fundamentally changes the geometry**:
- Without biases: all feature trajectories are perfectly straight lines through origin
- With biases: trajectories can curve (ReLU kinks become visible)

This is a **real representational difference**, not just "moving z(0)".

### Revised Conclusion

**VERDICT: INCONCLUSIVE**
**Confidence: LOW**

What we can say:
1. Performance differences at the frontier are small (<10%), but CIs are too wide to bound precisely
2. Without-bias models may fail more often (needs more investigation)
3. Without-bias models have fundamentally different geometry (positive homogeneity)

What we cannot say:
- Whether the effect is "negligible" vs "small but real"
- Whether training stability differs systematically

### Open Questions
1. Does positive homogeneity help or hurt representation learning?
2. Why do some runs fail? What's the mechanism?
3. With enough seeds, can we bound the frontier difference to <5%?

### Final Note: z(0) = 0 Constraint Isn't Worth Enforcing

The z(0) location (where the zero vector maps in latent space) is an arbitrary coordinate choice:

- **If enforced during training**: Requires computing `encoder(0)` every forward pass and subtracting it — extra computation for no benefit.
- **If done post-hoc**: Just define `encoder_new(x) = encoder(x) - c` and `decoder_new(z) = decoder(z + c)` where `c = encoder(0)` — this is a trivial coordinate shift that doesn't change the representation.

**Conclusion**: Don't constrain z(0). The interesting question is instead whether "write-linear + read-nonlinear" is sufficient (see `write_linear_experiment.ipynb`).

---

