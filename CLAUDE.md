# Nonlinear Feature Encoding Phase Diagram

## Project Overview
This project explores when autoencoders learn nonlinear vs linear encodings as a function of input dimension (n), bottleneck dimension (m), and depth (l). The goal is to map out a "phase diagram" of encoding behavior and relate findings to LLM-scale models.

## Structure
- `core.py` — Shared module with Autoencoder class, training, and measurement functions
- `nonlinear_features_v0.ipynb` — Main notebook for phase diagram sweeps
- `toy_models_sanity_check.ipynb` — Reproducing Elhage et al. Toy Models of Superposition
- `metric_exploration.ipynb` — Exploring compression metrics beyond n/m ratio
- `z0_constraint_exploration.ipynb` — Investigating z(0) degree of freedom (concluded: not worth constraining)
- `write_linear_experiment.ipynb` — Testing "write linear + read nonlinear" hypothesis
- `signed_features.ipynb` — Analyzing signed (Uniform[-1,1]) vs unsigned features: geometry, nonlinear gain, homogeneity breaking
- `feature_similarity_geometry.ipynb` — Testing whether bottleneck geometry reflects input feature similarity structure (block, cyclic, hierarchical); connects to Engels/Tegmark circular features and SAE feature splitting
- `theoretical_optimal_compression.ipynb` — Theoretical analysis of optimal compression geometry: n-gon packing, onion features, angular vs radial tradeoffs
- `computation_in_superposition.ipynb` — Testing whether nonlinear features are used in computation (AND, product) vs just representation; connects Hänni et al. (2408.05451), LawrenceC's "features aren't computational primitives", and the Nanda evidence hierarchy
- `feature_taxonomy.ipynb` — 1D-linear vs multi-D-linear vs genuinely nonlinear feature taxonomy; connects Engels/Tegmark, Shafran MFA, Luo diffusion meta-model, SpaDE, Circuits Updates July 2024
- `compositional_splitting.ipynb` — Compositional/hierarchical feature data → endogenous splitting; tests whether compression ratio controls granularity level, dendrogram recovery, SAE dictionary splitting, depth comparison; connects to SAE feature splitting phenomenon
- `representational_geometry.ipynb` — Neuroscience representational geometry applied to our toy models; RDMs, frame operator spectra, manifold capacity, testing "geometry beyond superposition" (jake_mendel/Sharkey critique)
- `parameter_decomposition_spd.ipynb` — SPD (Bushnaq/Braun/Sharkey 2025) vs SAE: does parameter-space decomposition handle nonlinear features better than activation-space? Tests parameter additivity violation, ReLU regime diversity, ablation selectivity across phase diagram
- `rate_distortion_theory.ipynb` — Rate-distortion theory analysis: spike+uniform as deviation from Gaussian, effective compression ratio (1-S)n/m, phase transition predictions, Gaussian vs sparse control experiments
- `slt_training_dynamics.ipynb` — SLT/training dynamics: LLC estimation during training, phase transitions in feature development, Hessian spectrum degeneracy, depth/compression effects on singular structure; connects Lau et al. LLC, Chen et al. SLT toy models, Watanabe RLCT
- `lessons-learned.md` — Detailed experimental standards and past mistakes

## Key Concepts
- **Autoencoder architecture**: `n -> [n]*l -> m -> [n]*l -> n` with ReLU activations
- **Nonlinear gain**: Relative MSE improvement over best linear approximation
- **Positive homogeneity**: Bias-free ReLU networks satisfy f(t·x) = t·f(x), making feature trajectories perfectly linear

## Experimental Rigor (Summary)

Three principles; see `lessons-learned.md` for details:

1. **"Not significant" ≠ "no effect"**: Wide CIs mean we don't know. State power limitations explicitly.

2. **Exact/surprising values demand explanation**: If something is exactly 1.0 or unexpectedly clean, derive why mathematically before moving on.

3. **Look at individuals before aggregates**: Plot individual runs, understand failure mechanisms, quantify qualitative observations.

## Dependencies
Python 3, PyTorch, NumPy, Matplotlib, tqdm, pandas

## Notes
- Data generated on-the-fly (sparse random vectors), no external datasets
- GPU used if available, otherwise CPU
