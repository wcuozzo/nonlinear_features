# Lessons Learned

Hard-won insights from this project. Reference when designing experiments or drawing conclusions.

## z(0) Experiment (2026-03-11)

### Theoretical Insight: Positive Homogeneity of Bias-Free ReLU Networks

**Key property**: ReLU networks without biases satisfy f(t·x) = t·f(x) for all t > 0.

**Proof**:
1. For a single ReLU unit: ReLU(t·a) = t·ReLU(a) for t > 0
   - If a ≥ 0: ReLU(t·a) = t·a = t·ReLU(a) ✓
   - If a < 0: ReLU(t·a) = 0 = t·0 = t·ReLU(a) ✓
2. For a linear layer: W·(t·x) = t·(W·x)
3. Composing layers: Each layer preserves the factor of t
4. Therefore: encoder(t·x) = t·encoder(x) for the entire bias-free network

**Consequences**:
1. **Feature trajectories are perfectly linear**: z(t·eᵢ) = t·z(eᵢ) — straight lines through origin
2. **Arc/chord ratio = 1.0 exactly**: This is a mathematical certainty, not an empirical observation
3. **Multiple encoder layers collapse to single linear projection**: If f(t·x) = t·f(x) for all t > 0, then f must be linear on each ray from the origin. With ReLU activations, this means the encoder is effectively piecewise-linear with the same linear function on all positive rays.

**Implication**: Removing encoder biases doesn't just change where z(0) is — it fundamentally changes the geometry. The encoder becomes "write-linear" regardless of depth.

**Example verification** (from experiments):
```python
# For a bias-free encoder:
encoder(0.5 * e_i) == 0.5 * encoder(e_i)  # Exactly true
encoder(2.0 * e_i) == 2.0 * encoder(e_i)  # Exactly true
```

This led to the follow-up question: Is "write-linear + read-nonlinear" sufficient? → See `write_linear_experiment.ipynb`

### Statistical Mistakes Made
1. **Claimed "no effect" with wide CIs**: CIs were ~10x wider than point estimates. "Not significant" means "we don't know", not "no effect".
2. **Didn't estimate required sample size**: Would need ~400+ seeds to bound effects to ±5%.
3. **Results changed with parameters**: 0 vs 3 failures depending on n_steps. Conclusions weren't robust.

### Qualitative Analysis Gaps
1. **Didn't investigate individual failures**: Why did specific runs fail? What's the mechanism?
2. **Didn't explain surprising values**: arc/chord = 1.0 exactly should have triggered theoretical investigation immediately.
3. **Conflated analyses from different runs**: Training dynamics results contradicted across parameter settings.

---

## Experimental Standards (Detailed)

### 1. Quantile Analysis (Not Just Means)
- Report at multiple quantiles: top-1, top-3, top-25%, median, all
- Use bootstrap CIs at each quantile
- Interpret the progression: where does the effect appear/disappear?

### 2. Minimum Scaling Curves
- Plot E[min | k] vs k for both conditions
- Key question: do conditions converge or diverge as you run more seeds?

### 3. Training Dynamics
- Plot individual runs before aggregating
- Auto-classify curve shapes
- Separate "did training succeed?" from "given success, which is better?"

### 4. Quantify Visuals
- Never conclude from visual patterns without measurement
- Use appropriate statistical tests

### 5. Outlier Handling
- Don't use symmetric IQR exclusion for optimization outcomes
- Failed runs are one-tailed; use quantile progression instead

### 6. Statistical Reporting
- Report effect sizes alongside p-values
- Include bootstrap CIs
- State confidence level with main source of uncertainty

### 7. Power Requirements
- Before "no effect": Can you bound effect to <5%?
- "Not significant" ≠ "no effect"
- Estimate samples needed: n ≈ 4 × current_n × (CI_width / target_width)²

### 8. Anomalous Results
- Exact values demand theoretical explanation
- Unexpected results are often most informative

### 9. Qualitative Differences
- Quantify HOW things differ, not just that they differ
- Understand failure modes

### 10. Contradictions
- Investigate before moving on
- Note parameter dependencies

---

## Pre-Conclusion Checklist

### Statistical Rigor
- [ ] CIs narrow enough to bound effect to actionable threshold?
- [ ] If wide CIs, stated "insufficient power" rather than "no effect"?
- [ ] Reported both point estimates AND uncertainty?

### Theoretical Grounding
- [ ] Exact/surprising values have mathematical explanation?
- [ ] Derived predictions BEFORE experiments?
- [ ] Interpretation matches math?

### Qualitative Investigation
- [ ] Looked at individual examples, not just aggregates?
- [ ] Understand failure/outlier mechanisms?
- [ ] Plotted raw data?

### Reproducibility
- [ ] Conclusions hold across parameter settings?
- [ ] Noted parameter dependencies?
- [ ] Would different seed change conclusion?

### Intellectual Honesty
- [ ] Confidence levels accurate?
- [ ] Investigated surprising results?
- [ ] Distinguishing "we found X" from "X is true"?
