# Nonlinear Feature Encoding - Experiment Log

This log tracks overnight experiments investigating when and why neural networks store features as nonlinear functions of activations.

## Background

The core question: Under what conditions do autoencoders learn nonlinear vs linear encodings?

**Prior work in notebooks:**
- `nonlinear_features_v0.ipynb`: Basic phase diagrams over (n, m, l), linearity/nonlinearity metrics
- `metric_exploration.ipynb`: Searching for predictive metrics (found n^a/m^b forms work reasonably)
- `latent_viz_2d.ipynb`: 2D visualizations showing feature trajectories, depth effects

**Key findings so far:**
- Higher compression ratio (n/m) tends to increase nonlinear gain
- Deeper networks show more nonlinear encodings
- Sparsity affects the transition

---

## Experiment 1: Feature Interference and Superposition Analysis

**Date:** 2026-02-26

**Hypothesis:** Nonlinear encodings emerge when the network needs to resolve interference between features that would collide in a purely linear encoding.

**Approach:**
1. Train autoencoders across (n, m, l) configurations: n∈{16,32,64,128}, m∈{4,8,16,32}, l∈{1,2,3}
2. Measure feature interference in both linear (weight-product) and learned encodings
3. Compute pairwise nonlinearity: deviation from additivity when encoding pairs of features
4. Track effective features ratio (features with low interference)

**Key Metrics Introduced:**
- `interference_reduction`: How much the learned encoding reduces feature interference vs. linear baseline
- `avg_pairwise_nonlinearity`: Non-additivity when encoding feature pairs (high = nonlinear)
- `effective_features_ratio`: Fraction of features distinguishable in latent space

**Results:**

| Correlation | Value |
|-------------|-------|
| nonlinear_gain vs pairwise_nonlinearity | **0.70** |
| nonlinear_gain vs interference_reduction | 0.09 |
| nonlinear_gain vs compression_ratio | -0.13 |
| interference_reduction vs compression | 0.42 |
| effective_features vs nonlinear_gain | -0.45 |

**Key Findings:**

1. **Pairwise nonlinearity is the strongest predictor of nonlinear gain (r=0.70)**
   - Networks that show non-additive feature encoding (z(x+y) ≠ z(x)+z(y)) also show higher MSE improvement from nonlinear encoding
   - This suggests nonlinearity is specifically used for feature interaction, not just general compression

2. **Interference reduction has weak correlation with nonlinear gain (r=0.09)**
   - Contrary to initial hypothesis, networks don't primarily use nonlinearity to reduce feature interference
   - The learned interference is often *similar* to linear baseline, suggesting nonlinearity serves a different purpose

3. **Effective features ratio negatively correlates with nonlinear gain (r=-0.45)**
   - High nonlinear gain regimes have *fewer* cleanly separable features
   - Suggests nonlinearity enables denser packing where features don't need to be individually separable

4. **High nonlinear gain regime characteristics:**
   - Average compression ratio: 5.4
   - Average depth: 1.9 (shallower than expected)
   - Average interference reduction: 0.014 (minimal)

**Insights:**
- Nonlinearity appears to serve **feature interaction** more than interference reduction
- The network may use nonlinear encoding to represent *combinations* of features efficiently
- Low depth (l=1-2) can achieve high nonlinear gain if compression is sufficient
- This aligns with the idea that nonlinearity enables a form of "computation in superposition"

**Output Files:**
- `experiment_1/main_results.png` - Correlation plots
- `experiment_1/interference_matrices.png` - Example linear vs learned interference
- `experiment_1/results.csv` - Full data
- `experiment_1/correlations.json` - Computed correlations

**Next Steps (suggested):**
1. Investigate *which* feature pairs show highest pairwise nonlinearity
2. Test whether co-occurring features (in training data) develop more nonlinear interactions
3. Explore the role of depth more carefully - why does shallow+compressed beat deep?

---

## Experiment 2: Feature Co-occurrence and Nonlinear Encoding

**Date:** 2026-02-26

**Hypothesis:** Nonlinear encodings develop specifically for features that frequently co-occur in training data. Building on Experiment 1's finding that pairwise nonlinearity (r=0.70) strongly predicts nonlinear gain, we hypothesized that co-occurring features would develop more nonlinear interactions because the network needs to disentangle commonly co-activated features.

**Approach:**
Four sub-experiments testing the co-occurrence hypothesis:
- **Exp A:** Vary co-occurrence strength (0.0 to 1.0) with fixed cluster structure
- **Exp B:** Test if within-cluster (high co-occurrence) pairs show more nonlinear encoding than between-cluster pairs
- **Exp C:** 2D sweep over compression ratio × co-occurrence strength
- **Exp D:** Compare independent features vs structured co-occurrence across multiple (n, m) configurations

**Key Metrics:**
- Correlation between co-occurrence and pairwise encoding nonlinearity
- Within-cluster vs between-cluster nonlinearity
- Nonlinear gain difference: structured vs independent features

**Results:**

| Metric | Value |
|--------|-------|
| Correlation: co-occurrence strength vs nonlinear gain | **-0.78** |
| Correlation: co-occurrence vs pairwise nonlinearity | 0.03 (p=0.48) |
| Within-cluster nonlinearity | 0.062 ± 0.055 |
| Between-cluster nonlinearity | 0.059 ± 0.038 |
| T-test (within vs between) p-value | 0.48 |
| Avg nonlinear gain increase with structure | **-0.12** |

**Key Findings:**

1. **Co-occurrence DECREASES nonlinear gain (r=-0.78)**
   - Contrary to our hypothesis, higher feature co-occurrence leads to LOWER nonlinear encoding benefit
   - At co-occurrence strength=0 (independent features): nonlinear_gain=0.096
   - At co-occurrence strength=1 (fully coupled clusters): nonlinear_gain=0.005
   - This is a strong negative correlation, the opposite of what we predicted

2. **No significant within-cluster vs between-cluster difference (p=0.48)**
   - Features that co-occur (within clusters) do NOT develop more nonlinear encodings
   - The network treats high-cooccurrence and low-cooccurrence pairs similarly

3. **Structured data shows LOWER nonlinear gain than independent data**
   - Across all (n, m) configurations tested, structured co-occurrence reduced nonlinear gain
   - Average reduction: -0.12 (substantial)
   - Example: n=64, m=8: independent=0.14, structured≈0.00

4. **Compression ratio still matters, but interacts with co-occurrence**
   - Highest nonlinear gain at m=8 (high compression) with cooc_strength=0 (independent features)
   - With high co-occurrence, even high compression yields near-linear encodings

**Insights:**

- **Independence drives nonlinearity, not co-occurrence:** Networks develop nonlinear encodings to handle *independent* features that must be compressed, not co-occurring ones
- **Co-occurring features are "easier" to encode:** When features co-occur, the network can effectively treat them as a single composite feature, enabling more linear encoding
- **Nonlinearity serves separation, not composition:** Experiment 1 showed nonlinearity correlates with pairwise non-additivity. Experiment 2 clarifies this is NOT about encoding co-occurring features together, but about SEPARATING independent features that must share the same bottleneck space
- **Implications for LLMs:** If this holds at scale, nonlinear feature representations should emerge more strongly for independent/uncorrelated concepts than for semantically related ones

**Revised Understanding:**
The original interpretation from Experiment 1 - that nonlinearity enables "computation in superposition" - needs refinement. Nonlinearity enables *separation* in superposition, not composition. When features co-occur, they effectively reduce the dimensionality of the data distribution, making linear encoding sufficient.

**Output Files:**
- `experiment_2/main_results.png` - 6-panel visualization of all experiments
- `experiment_2/cooccurrence_scatter.png` - Scatter plot of co-occurrence vs nonlinearity
- `experiment_2/exp_a_cooccurrence_strength.csv` - Strength sweep results
- `experiment_2/exp_c_compression_interaction.csv` - Compression × co-occurrence grid
- `experiment_2/exp_d_independent_vs_structured.csv` - Comparison results
- `experiment_2/summary.json` - Key statistics

**Next Steps (suggested):**
1. Test the "separation hypothesis" directly: measure how well the network can decode individual features vs feature combinations
2. Investigate the role of feature magnitude distributions (not just presence/absence)
3. Explore whether depth helps with *independent* features specifically
4. Consider non-binary co-occurrence structures (continuous correlations, hierarchical)

---

## Experiment 3: Depth-Separation Interaction Analysis

**Date:** 2026-02-26

**Hypothesis:** Depth's benefit for nonlinear encoding depends critically on the effective dimensionality of the data. When features are truly independent (high effective dim), deeper networks should show more benefit. When features co-occur (lower effective dim), depth should matter less. This builds on Experiment 2's finding that independent features drive nonlinearity.

**Approach:**
Four sub-experiments investigating depth × effective dimensionality interaction:
- **Exp A:** Depth sweep (1-6) for independent vs correlated features (varying n_true from 8 to 64 in n=64 space)
- **Exp B:** Find "critical depth" threshold where nonlinear encoding emerges (depth 1-8, multiple compression ratios)
- **Exp C:** Quantify depth benefit scaling with effective dimensionality
- **Exp D:** Test if depth improves feature separation specifically (not just lower MSE)

**Key Metrics:**
- Nonlinear gain across depth × effective dim grid
- Critical depth (minimum depth for nonlinear_gain > 0.05)
- Depth benefit: Δ nonlinear_gain between shallow and deep networks
- Separation score: 1 - avg cosine similarity between single-feature encodings

**Results:**

| Metric | Value |
|--------|-------|
| Max nonlinear gain (independent, n_true=64) | **0.155** |
| Max nonlinear gain (correlated, n_true=8) | **0.0016** |
| Critical depth (m=4, high compression) | 1 |
| Critical depth (m=32, low compression) | 3 |
| Depth benefit vs effective_dim correlation | **-0.56** |
| Separation vs nonlinear_gain correlation | 0.22 |
| Depth vs separation correlation | **-0.76** |

**Key Findings:**

1. **Independent features show ~100x more nonlinear gain than correlated features**
   - At n_true=64 (independent): max nonlinear_gain = 0.155
   - At n_true=8 (12.5% effective dim): max nonlinear_gain = 0.0016
   - This strongly confirms Experiment 2: independence drives nonlinearity, not co-occurrence

2. **Critical depth depends on compression ratio, not raw parameters**
   - High compression (m=4, m=8, m=16): Critical depth = 1 (even shallow networks go nonlinear)
   - Low compression (m=32): Critical depth = 3 (needs more depth to trigger nonlinearity)
   - Interpretation: When compression is severe, even depth-1 networks must use nonlinearity to fit the data

3. **Depth benefit is NEGATIVELY correlated with effective dimensionality (r=-0.56)**
   - Contrary to hypothesis! Deeper networks show MORE benefit with LOWER effective dim
   - This is counterintuitive but explainable: with highly correlated features, the data lies on a low-dim manifold, and deeper networks can learn this structure better
   - With independent features, even shallow networks already achieve high nonlinear gain

4. **Depth DECREASES feature separation (r=-0.76)**
   - Deeper networks have LOWER separation scores (features become more similar in latent space)
   - This suggests depth doesn't improve nonlinearity by better separation
   - Instead, depth may enable more sophisticated feature mixing/sharing of latent dimensions

5. **Separation score has weak positive correlation with nonlinear gain (r=0.22)**
   - Some link exists, but separation is not the primary mechanism
   - Networks can achieve high nonlinear gain with low separation (dense feature packing)

**Insights:**

- **Compression dominates depth:** The critical depth threshold is primarily determined by compression ratio, not raw dimensions. Highly compressed networks "must" go nonlinear at any depth.

- **Depth enables manifold learning, not separation:** The negative depth-separation correlation suggests deeper networks learn to represent the data manifold more compactly, sharing latent dimensions across features rather than separating them.

- **Independence vs depth trade-off:** With independent features, shallow+compressed networks achieve high nonlinear gain. Depth adds value primarily when features have correlational structure that can be exploited.

- **Revised model of nonlinearity:**
  - Shallow networks: Nonlinearity emerges from compression + independence (forced to pack independent features)
  - Deep networks: Nonlinearity emerges from manifold learning (even with correlated features)
  - Both paths lead to nonlinear encoding but through different mechanisms

**Output Files:**
- `experiment_3/main_results.png` - 6-panel summary figure
- `experiment_3/exp_a_results.png` - Depth × effective dim analysis
- `experiment_3/exp_b_results.png` - Critical depth thresholds
- `experiment_3/exp_c_results.png` - Depth benefit scaling
- `experiment_3/exp_d_results.png` - Separation quality analysis
- `experiment_3/exp_*.csv` - Raw data for all experiments
- `experiment_3/summary.json` - Key statistics

**Next Steps (suggested):**
1. Investigate the "manifold learning" hypothesis: do deeper networks learn lower-rank representations?
2. Test whether the depth-manifold effect holds for non-linear ground truth data distributions
3. Explore the transition: at what effective dimensionality does depth switch from harmful to helpful?
4. Scale up to larger n to see if these patterns hold or if new regimes emerge

---

