# Final Summary: Nonlinear Features in Bottleneck Autoencoders

## Key Findings (Ranked by Importance)

### 1. Progressive Training Shatters the Depth Ceiling (Exp 20)
Standard training collapses to zero nonlinear gain at depth l≥8. Progressive depth training (grow network layer-by-layer) achieves **gain=0.039 at l=10**—the highest observed in 20 experiments. The "universal depth ceiling" was an initialization problem, not an architectural limit.

### 2. Larger Input Dimension → Higher Nonlinear Gain (Exp 5, 9, 10)
With scaled training (steps ∝ n), nonlinear gain scales as ~n^(+1 to +2). The apparent negative scaling from fixed training was entirely underfitting. **Implication for LLMs: massive scale + extensive training should push deep into nonlinear territory.**

### 3. Compression Ratio is the Dominant Driver (Exp 3, 8)
Compression ratio (n/m) correlates r=0.83 with nonlinear gain—the strongest predictor. Higher compression forces nonlinear encoding regardless of sparsity level.

### 4. Depth and Compression are Substitutes (Exp 8)
Strong negative interaction (r=-0.93): high compression weakens depth benefit; high depth weakens compression benefit. **Both are alternative paths to nonlinearity.** Phase boundary approximates: l × log2(CR) ≈ 10 (shifts with n).

### 5. LeakyReLU α=0.2 Outperforms ReLU by 116% (Exp 11-12)
Optimal negative slope is 0.2, far from PyTorch default (0.01). Tanh **never** achieves positive nonlinear gain at any depth—bounded activations prevent nonlinear encoding.

### 6. Variance Concentration Mechanism (Exp 1-2)
Top-1 variance fraction (r=0.62) predicts nonlinear gain better than rank or Gini coefficient. Compression forces variance concentration, which enables nonlinearity.

### 7. Critical Training Threshold Exists (Exp 7)
All configurations transition from negative to positive nonlinear gain at ~120-200 steps. Gradual S-curve, not sharp transition. Deeper networks reach threshold ~20 steps earlier.

### 8. Optimal Sparsity ≈ 0.10 (Exp 3)
Dense data (sparsity=0.30) shows NO nonlinear benefit—linear encoding wins. Peak gain at moderate sparsity (10%). Sparsity is prerequisite but doesn't shift compression threshold.

---

## Phase Diagram (ASCII)

```
                    COMPRESSION RATIO (n/m)
                 4       8      16      32
              ┌──────┬──────┬──────┬──────┐
            1 │  -   │  -   │  -   │  ~   │  LINEAR REGIME
              ├──────┼──────┼──────┼──────┤  (gain < 0)
            2 │  -   │  -   │  ~   │  +   │
    D         ├──────┼──────┼──────┼──────┤
    E       3 │  -   │  ~   │  +   │  ++  │  TRANSITION
    P         ├──────┼──────┼──────┼──────┤  ZONE (~)
    T       4 │  ~   │  +   │  ++  │  ++  │
    H         ├──────┼──────┼──────┼──────┤
    (l)     5 │  +   │  ++  │  ++  │  ++  │  NONLINEAR REGIME
              ├──────┼──────┼──────┼──────┤  (gain > 0)
            6 │  ++  │  ++  │  ++  │  ++  │
              ├──────┼──────┼──────┼──────┤
          ≥7  │  X   │  X   │  X   │  X   │  DEPTH CEILING*
              └──────┴──────┴──────┴──────┘

Legend:  -  = negative gain (linear wins)
         ~  = near-zero gain (transition)
         +  = positive gain (nonlinear wins)
         ++ = strong positive gain
         X  = collapse to zero (standard training)

*Ceiling overcome with progressive training → gains continue to l=10+
```

**Key boundary:** Phase transition occurs approximately where l × log2(CR) ≈ 10

**Scale dependence:** Larger n lowers the critical threshold and increases peak gain.

---

## Linear vs Nonlinear Encoding Regions

### Clearly Linear (nonlinear encoding provides no benefit):
- Any depth + sparsity ≥ 0.30 (dense data)
- Tanh activation at any configuration
- l ≥ 8 with standard training (ceiling collapse)
- Low compression (CR ≤ 4) + shallow depth (l ≤ 2)

### Clearly Nonlinear (significant encoding benefit):
- High compression (CR ≥ 16) + moderate depth (l = 4-6)
- Progressive training + any depth up to l = 10+
- Larger n (128-256) with scaled training budget
- LeakyReLU α ≈ 0.1-0.3

### Transition Zone (marginal benefit, sensitive to hyperparameters):
- CR = 8, l = 3-4
- Very small n (32) regardless of compression
- Sparsity extremes (0.03 or 0.20)

---

## Core Hypothesis Assessment

**Hypothesis:** Autoencoders learn nonlinear encodings as a function of (n, m, l).

### Strong Evidence FOR:
1. **Compression drives nonlinearity (r=0.83)** — Most robust finding across all experiments
2. **Scale amplifies nonlinearity** — n=256 achieves 20× higher gain than n=32 when properly trained
3. **Depth enables nonlinearity (r=0.94 with scaled training)** — Near-perfect correlation when training is adequate
4. **Progressive training removes depth ceiling** — Architecture can support deep nonlinear encoding; limitation was optimization

### Evidence AGAINST (or qualifying):
1. **Depth ceiling is real for standard training** — Without progressive training, l≥8 universally fails
2. **Bounded activations (Tanh) cannot achieve nonlinear encoding** — Architectural choice can prevent nonlinearity entirely
3. **Dense data prefers linear encoding** — Sparsity is a prerequisite for nonlinear benefit
4. **Small n shows weak effects** — At n=32, correlations are weaker and gains are smaller

### Verdict:
**The hypothesis is strongly supported** with important qualifications: nonlinear encoding requires (a) sparse data, (b) unbounded activations, and (c) sufficient training. Given these conditions, the (n, m, l) relationship is predictable and follows a clear phase diagram.

---

## Recommended Next Experiments

### Tier 1: High Priority (extend breakthrough findings)
1. **Progressive training + α=0.2** — Combine the two biggest improvements (Exp 12 + 20)
2. **Progressive training to l=15-20** — Find where progressive training saturates (if ever)
3. **Progressive training at n=512, 1024** — Verify progressive training scales to LLM-relevant sizes

### Tier 2: Mechanistic Understanding
4. **What happens at the training threshold?** — Visualize latent representations before/after the ~160-step transition
5. **Why does progressive training work?** — Compare gradient norms, Hessian spectra, loss landscapes between standard vs progressive
6. **Why does Tanh fail?** — Test Tanh at very low compression where nonlinearity should be "easy"

### Tier 3: Architecture Extensions
7. **Sparse autoencoders** — Does the phase diagram change with L1 sparsity penalty?
8. **Real data (MNIST, CIFAR)** — Test whether findings generalize beyond synthetic sparse vectors
9. **Attention layers** — Add attention to bottleneck and test if depth ceiling shifts

### Tier 4: LLM Connections
10. **Train toy transformer with progressive depth** — Test if the depth ceiling finding applies to transformers
11. **Analyze pretrained LLM layers** — Measure "nonlinear gain" equivalent in transformer blocks
12. **Feature superposition analysis** — Connect to Anthropic's superposition work—does compression create superposition?

---

## Summary Statistics

| Metric | Best Configuration | Value |
|--------|-------------------|-------|
| Highest nonlinear gain | n=128, l=10, progressive training | 0.039 |
| Strongest predictor | Compression ratio (CR) | r = 0.83 |
| Optimal activation | LeakyReLU α=0.2 | +116% vs ReLU |
| Optimal sparsity | 0.10 | Peak gain |
| Depth ceiling (standard) | l ≈ 6-7 | Universal |
| Depth ceiling (progressive) | None observed | l=10 works |
| Scale benefit | n scaling exponent | +1 to +2 |
