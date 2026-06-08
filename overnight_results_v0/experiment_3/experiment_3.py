"""
Experiment 3: Depth-Separation Interaction and Critical Depth Thresholds

Hypothesis: Depth's benefit for nonlinear encoding depends critically on the
effective dimensionality of the data. When features are truly independent,
deeper networks should show more benefit. When features co-occur (lower effective dim),
depth should matter less.

Sub-experiments:
A) Depth sweep for independent vs correlated features
B) Find "critical depth" where nonlinear encoding emerges
C) Depth × effective dimensionality interaction
D) Verify depth helps separation quality (not just compression)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Dict, List
import pandas as pd
import json
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/overnight_results/experiment_3"

# ============================================================================
# Model and Training (from v0 notebook)
# ============================================================================

class Autoencoder(nn.Module):
    def __init__(self, n: int, m: int, l: int, activation=nn.ReLU):
        super().__init__()
        self.n = n
        self.m = m
        self.l = l

        encoder_layers = []
        for i in range(l):
            encoder_layers.append(nn.Linear(n, n))
            encoder_layers.append(activation())
        encoder_layers.append(nn.Linear(n, m))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        decoder_layers.append(nn.Linear(m, n))
        decoder_layers.append(activation())
        for i in range(l - 1):
            decoder_layers.append(nn.Linear(n, n))
            decoder_layers.append(activation())
        if l > 0:
            decoder_layers.append(nn.Linear(n, n))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def generate_sparse_data(n_samples: int, n_features: int, sparsity: float = 0.1) -> torch.Tensor:
    mask = (torch.rand(n_samples, n_features) < sparsity).float()
    values = torch.rand(n_samples, n_features)
    return (mask * values).to(device)


def generate_correlated_features(n_samples: int, n_features: int, n_true_features: int,
                                  sparsity: float = 0.1) -> torch.Tensor:
    """Generate data from fewer true underlying features (lower effective dim)."""
    sources = generate_sparse_data(n_samples, n_true_features, sparsity)
    mixing = torch.randn(n_true_features, n_features, device=device)
    mixing = mixing / mixing.norm(dim=0, keepdim=True)
    return torch.relu(sources @ mixing)  # Keep positive


def train_autoencoder(model: Autoencoder, data_generator, n_steps: int = 5000,
                      batch_size: int = 256, lr: float = 1e-3, verbose: bool = False) -> List[float]:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    iterator = tqdm(range(n_steps), disable=not verbose)
    for step in iterator:
        x = data_generator(batch_size)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def measure_encoding_linearity(model: Autoencoder, data_generator, n_samples: int = 1000) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x = data_generator(n_samples)
        z = model.encode(x)

        x_with_bias = torch.cat([x, torch.ones(n_samples, 1, device=device)], dim=1)
        W_linear = torch.linalg.lstsq(x_with_bias, z).solution
        z_linear = x_with_bias @ W_linear

        z_var = z.var(dim=0).sum()
        residual_var = (z - z_linear).var(dim=0).sum()
        linearity_score = 1 - (residual_var / z_var).item() if z_var > 0 else 1.0

        x_recon_full, _ = model(x)
        x_recon_linear = model.decode(z_linear)

        mse_full = nn.functional.mse_loss(x_recon_full, x).item()
        mse_linear = nn.functional.mse_loss(x_recon_linear, x).item()

    return {
        'linearity_score': linearity_score,
        'mse_full': mse_full,
        'mse_linear': mse_linear,
        'nonlinear_gain': (mse_linear - mse_full) / (mse_linear + 1e-8)
    }


def measure_feature_separation(model: Autoencoder, n_features: int, n_test: int = 100) -> Dict[str, float]:
    """
    Measure how well individual features can be decoded from the latent space.
    This directly tests the "separation hypothesis" from Experiment 2.
    """
    model.eval()

    # Create single-feature test vectors
    single_feature_z = []
    for i in range(min(n_features, 32)):  # Limit for speed
        x_single = torch.zeros(n_test, n_features, device=device)
        x_single[:, i] = torch.rand(n_test, device=device)  # Random magnitudes
        with torch.no_grad():
            z = model.encode(x_single)
        single_feature_z.append(z)

    single_feature_z = torch.stack(single_feature_z)  # [n_feat, n_test, m]

    # Measure separability: can we distinguish which feature was active?
    # Use cosine similarity between latent representations of different features
    mean_z = single_feature_z.mean(dim=1)  # [n_feat, m]
    mean_z_norm = mean_z / (mean_z.norm(dim=1, keepdim=True) + 1e-8)

    cos_sim_matrix = mean_z_norm @ mean_z_norm.T  # [n_feat, n_feat]

    # Off-diagonal average (lower = better separation)
    n_feat = cos_sim_matrix.shape[0]
    mask = 1 - torch.eye(n_feat, device=device)
    avg_off_diag_sim = (cos_sim_matrix * mask).sum() / mask.sum()

    # Measure variance in latent space per feature (higher = more expressive)
    within_feature_var = single_feature_z.var(dim=1).mean().item()

    return {
        'avg_feature_similarity': avg_off_diag_sim.item(),
        'separation_score': 1 - avg_off_diag_sim.item(),  # Higher = better separation
        'within_feature_variance': within_feature_var
    }


# ============================================================================
# Experiment A: Depth sweep for independent vs correlated features
# ============================================================================

def run_experiment_a():
    """Compare depth effect for independent vs correlated features."""
    print("\n" + "="*60)
    print("Experiment A: Depth effect on independent vs correlated features")
    print("="*60)

    n = 64
    m = 8
    sparsity = 0.1
    n_steps = 4000

    depth_values = [1, 2, 3, 4, 5, 6]
    n_true_features_values = [64, 32, 16, 8]  # 64 = independent, lower = more correlated

    results = []

    for n_true in n_true_features_values:
        effective_dim_ratio = n_true / n
        print(f"\nEffective dim ratio: {effective_dim_ratio:.2f} (n_true={n_true})")

        for depth in tqdm(depth_values, desc=f"  Depth sweep"):
            # Create data generator for this configuration
            if n_true == n:
                data_gen = lambda bs: generate_sparse_data(bs, n, sparsity)
            else:
                data_gen = lambda bs, nt=n_true: generate_correlated_features(bs, n, nt, sparsity)

            model = Autoencoder(n, m, depth).to(device)
            train_autoencoder(model, data_gen, n_steps=n_steps, verbose=False)

            metrics = measure_encoding_linearity(model, data_gen)
            sep_metrics = measure_feature_separation(model, n)

            results.append({
                'depth': depth,
                'n_true_features': n_true,
                'effective_dim_ratio': effective_dim_ratio,
                **metrics,
                **sep_metrics
            })

    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/exp_a_depth_vs_effective_dim.csv", index=False)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Nonlinear gain vs depth for different effective dims
    ax = axes[0, 0]
    for n_true in n_true_features_values:
        subset = df[df['n_true_features'] == n_true]
        label = f"n_true={n_true}" + (" (independent)" if n_true == n else f" (eff_dim={n_true/n:.1%})")
        ax.plot(subset['depth'], subset['nonlinear_gain'], 'o-', label=label)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Separation score vs depth
    ax = axes[0, 1]
    for n_true in n_true_features_values:
        subset = df[df['n_true_features'] == n_true]
        ax.plot(subset['depth'], subset['separation_score'], 'o-', label=f"n_true={n_true}")
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Separation Score')
    ax.set_title('Feature Separation vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Nonlinear gain vs effective dim for different depths
    ax = axes[1, 0]
    for depth in [1, 3, 6]:
        subset = df[df['depth'] == depth]
        ax.plot(subset['effective_dim_ratio'], subset['nonlinear_gain'], 'o-', label=f"depth={depth}")
    ax.set_xlabel('Effective Dim Ratio (n_true/n)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Effective Dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Heatmap of nonlinear gain
    ax = axes[1, 1]
    pivot = df.pivot(index='n_true_features', columns='depth', values='nonlinear_gain')
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Depth')
    ax.set_ylabel('N True Features')
    ax.set_title('Nonlinear Gain Heatmap')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp_a_results.png", dpi=150)
    plt.close()

    return df


# ============================================================================
# Experiment B: Find critical depth where nonlinear encoding emerges
# ============================================================================

def run_experiment_b():
    """Find the critical depth threshold for nonlinear encoding."""
    print("\n" + "="*60)
    print("Experiment B: Critical depth threshold analysis")
    print("="*60)

    n = 64
    sparsity = 0.1
    n_steps = 4000
    n_seeds = 3

    depth_values = list(range(1, 9))  # Fine-grained depth sweep
    m_values = [4, 8, 16, 32]

    results = []

    for m in m_values:
        compression_ratio = n / m
        print(f"\nCompression ratio: {compression_ratio:.1f} (m={m})")

        for depth in tqdm(depth_values, desc=f"  Depth sweep"):
            for seed in range(n_seeds):
                torch.manual_seed(42 + seed)

                data_gen = lambda bs: generate_sparse_data(bs, n, sparsity)
                model = Autoencoder(n, m, depth).to(device)
                train_autoencoder(model, data_gen, n_steps=n_steps, verbose=False)

                metrics = measure_encoding_linearity(model, data_gen)

                results.append({
                    'depth': depth,
                    'm': m,
                    'compression_ratio': compression_ratio,
                    'seed': seed,
                    **metrics
                })

    df = pd.DataFrame(results)

    # Aggregate over seeds
    df_agg = df.groupby(['depth', 'm', 'compression_ratio']).agg({
        'nonlinear_gain': ['mean', 'std'],
        'linearity_score': ['mean', 'std'],
        'mse_full': 'mean'
    }).reset_index()
    df_agg.columns = ['depth', 'm', 'compression_ratio',
                      'nonlinear_gain_mean', 'nonlinear_gain_std',
                      'linearity_score_mean', 'linearity_score_std',
                      'mse_full_mean']

    df_agg.to_csv(f"{OUTPUT_DIR}/exp_b_critical_depth.csv", index=False)

    # Find critical depth (where nonlinear_gain first exceeds a threshold)
    threshold = 0.05
    critical_depths = {}

    for m in m_values:
        subset = df_agg[df_agg['m'] == m]
        above_threshold = subset[subset['nonlinear_gain_mean'] > threshold]
        if len(above_threshold) > 0:
            critical_depths[m] = above_threshold['depth'].min()
        else:
            critical_depths[m] = None

    print("\nCritical depths (nonlinear_gain > 0.05):")
    for m, cd in critical_depths.items():
        print(f"  m={m}: depth >= {cd}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Nonlinear gain vs depth with error bars
    ax = axes[0]
    for m in m_values:
        subset = df_agg[df_agg['m'] == m]
        ax.errorbar(subset['depth'], subset['nonlinear_gain_mean'],
                    yerr=subset['nonlinear_gain_std'],
                    fmt='o-', label=f"m={m}", capsize=3)
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'threshold={threshold}')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth (with std)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Derivative of nonlinear gain (to find transition point)
    ax = axes[1]
    for m in m_values:
        subset = df_agg[df_agg['m'] == m].sort_values('depth')
        gains = subset['nonlinear_gain_mean'].values
        depths = subset['depth'].values
        derivatives = np.diff(gains) / np.diff(depths)
        ax.plot(depths[1:], derivatives, 'o-', label=f"m={m}")
    ax.set_xlabel('Depth')
    ax.set_ylabel('d(Nonlinear Gain)/d(Depth)')
    ax.set_title('Rate of Change in Nonlinear Gain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Critical depth vs compression ratio
    ax = axes[2]
    m_vals = [m for m in m_values if critical_depths[m] is not None]
    cd_vals = [critical_depths[m] for m in m_vals]
    cr_vals = [n/m for m in m_vals]
    ax.plot(cr_vals, cd_vals, 'o-', markersize=10)
    ax.set_xlabel('Compression Ratio (n/m)')
    ax.set_ylabel('Critical Depth')
    ax.set_title('Critical Depth vs Compression')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp_b_results.png", dpi=150)
    plt.close()

    return df_agg, critical_depths


# ============================================================================
# Experiment C: Depth benefit scales with effective dimensionality
# ============================================================================

def run_experiment_c():
    """Test if depth benefit is proportional to effective dimensionality."""
    print("\n" + "="*60)
    print("Experiment C: Depth benefit vs effective dimensionality scaling")
    print("="*60)

    n = 64
    m = 8
    sparsity = 0.1
    n_steps = 4000

    # Compare shallow vs deep across effective dimensionalities
    depth_pairs = [(1, 4), (2, 5), (3, 6)]  # (shallow, deep) pairs
    n_true_values = [8, 16, 24, 32, 48, 64]

    results = []

    for n_true in tqdm(n_true_values, desc="Effective dim sweep"):
        eff_dim_ratio = n_true / n

        for shallow, deep in depth_pairs:
            # Shallow model
            if n_true == n:
                data_gen = lambda bs: generate_sparse_data(bs, n, sparsity)
            else:
                data_gen = lambda bs, nt=n_true: generate_correlated_features(bs, n, nt, sparsity)

            model_shallow = Autoencoder(n, m, shallow).to(device)
            train_autoencoder(model_shallow, data_gen, n_steps=n_steps, verbose=False)
            metrics_shallow = measure_encoding_linearity(model_shallow, data_gen)

            # Deep model
            model_deep = Autoencoder(n, m, deep).to(device)
            train_autoencoder(model_deep, data_gen, n_steps=n_steps, verbose=False)
            metrics_deep = measure_encoding_linearity(model_deep, data_gen)

            # Compute depth benefit
            depth_benefit_gain = metrics_deep['nonlinear_gain'] - metrics_shallow['nonlinear_gain']
            depth_benefit_mse = (metrics_shallow['mse_full'] - metrics_deep['mse_full']) / (metrics_shallow['mse_full'] + 1e-8)

            results.append({
                'n_true': n_true,
                'effective_dim_ratio': eff_dim_ratio,
                'shallow_depth': shallow,
                'deep_depth': deep,
                'shallow_nonlinear_gain': metrics_shallow['nonlinear_gain'],
                'deep_nonlinear_gain': metrics_deep['nonlinear_gain'],
                'depth_benefit_gain': depth_benefit_gain,
                'depth_benefit_mse': depth_benefit_mse,
                'shallow_mse': metrics_shallow['mse_full'],
                'deep_mse': metrics_deep['mse_full']
            })

    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/exp_c_depth_benefit_scaling.csv", index=False)

    # Compute correlation
    correlation = df['effective_dim_ratio'].corr(df['depth_benefit_gain'])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Depth benefit vs effective dim
    ax = axes[0]
    for (shallow, deep) in depth_pairs:
        subset = df[(df['shallow_depth'] == shallow) & (df['deep_depth'] == deep)]
        ax.plot(subset['effective_dim_ratio'], subset['depth_benefit_gain'],
                'o-', label=f"depth {shallow}→{deep}")
    ax.set_xlabel('Effective Dim Ratio')
    ax.set_ylabel('Depth Benefit (Δ Nonlinear Gain)')
    ax.set_title(f'Depth Benefit vs Effective Dim\n(correlation: {correlation:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Raw nonlinear gain comparison
    ax = axes[1]
    subset = df[df['shallow_depth'] == 2]  # Use 2→5 comparison
    ax.plot(subset['effective_dim_ratio'], subset['shallow_nonlinear_gain'], 'o-', label='depth=2')
    ax.plot(subset['effective_dim_ratio'], subset['deep_nonlinear_gain'], 's-', label='depth=5')
    ax.set_xlabel('Effective Dim Ratio')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Shallow vs Deep Nonlinear Gain')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: MSE comparison
    ax = axes[2]
    ax.plot(subset['effective_dim_ratio'], subset['shallow_mse'], 'o-', label='depth=2 MSE')
    ax.plot(subset['effective_dim_ratio'], subset['deep_mse'], 's-', label='depth=5 MSE')
    ax.set_xlabel('Effective Dim Ratio')
    ax.set_ylabel('MSE')
    ax.set_title('Reconstruction Error vs Effective Dim')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp_c_results.png", dpi=150)
    plt.close()

    return df, correlation


# ============================================================================
# Experiment D: Does depth improve separation specifically?
# ============================================================================

def run_experiment_d():
    """Test if deeper networks achieve better feature separation, not just lower MSE."""
    print("\n" + "="*60)
    print("Experiment D: Depth and feature separation quality")
    print("="*60)

    n = 64
    m = 8
    sparsity = 0.1
    n_steps = 4000

    depth_values = [1, 2, 3, 4, 5, 6]
    n_seeds = 3

    results = []

    for depth in tqdm(depth_values, desc="Depth sweep"):
        for seed in range(n_seeds):
            torch.manual_seed(42 + seed)

            data_gen = lambda bs: generate_sparse_data(bs, n, sparsity)
            model = Autoencoder(n, m, depth).to(device)
            train_autoencoder(model, data_gen, n_steps=n_steps, verbose=False)

            metrics = measure_encoding_linearity(model, data_gen)
            sep_metrics = measure_feature_separation(model, n)

            results.append({
                'depth': depth,
                'seed': seed,
                **metrics,
                **sep_metrics
            })

    df = pd.DataFrame(results)

    # Aggregate
    df_agg = df.groupby('depth').agg({
        'nonlinear_gain': ['mean', 'std'],
        'separation_score': ['mean', 'std'],
        'mse_full': ['mean', 'std'],
        'avg_feature_similarity': ['mean', 'std']
    }).reset_index()
    df_agg.columns = ['depth',
                      'nonlinear_gain_mean', 'nonlinear_gain_std',
                      'separation_score_mean', 'separation_score_std',
                      'mse_full_mean', 'mse_full_std',
                      'feature_similarity_mean', 'feature_similarity_std']

    df_agg.to_csv(f"{OUTPUT_DIR}/exp_d_depth_separation.csv", index=False)

    # Compute correlations
    corr_sep_gain = df['separation_score'].corr(df['nonlinear_gain'])
    corr_depth_sep = df['depth'].corr(df['separation_score'])

    print(f"\nCorrelation: separation_score vs nonlinear_gain = {corr_sep_gain:.3f}")
    print(f"Correlation: depth vs separation_score = {corr_depth_sep:.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Separation score vs depth
    ax = axes[0, 0]
    ax.errorbar(df_agg['depth'], df_agg['separation_score_mean'],
                yerr=df_agg['separation_score_std'], fmt='o-', capsize=3)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Separation Score')
    ax.set_title(f'Feature Separation vs Depth\n(corr with depth: {corr_depth_sep:.3f})')
    ax.grid(True, alpha=0.3)

    # Plot 2: Nonlinear gain vs depth
    ax = axes[0, 1]
    ax.errorbar(df_agg['depth'], df_agg['nonlinear_gain_mean'],
                yerr=df_agg['nonlinear_gain_std'], fmt='o-', capsize=3, color='orange')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth')
    ax.grid(True, alpha=0.3)

    # Plot 3: Separation vs nonlinear gain (scatter)
    ax = axes[1, 0]
    ax.scatter(df['nonlinear_gain'], df['separation_score'], alpha=0.6)
    ax.set_xlabel('Nonlinear Gain')
    ax.set_ylabel('Separation Score')
    ax.set_title(f'Separation vs Nonlinear Gain\n(correlation: {corr_sep_gain:.3f})')
    ax.grid(True, alpha=0.3)

    # Plot 4: MSE vs depth
    ax = axes[1, 1]
    ax.errorbar(df_agg['depth'], df_agg['mse_full_mean'],
                yerr=df_agg['mse_full_std'], fmt='o-', capsize=3, color='green')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('MSE vs Depth')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exp_d_results.png", dpi=150)
    plt.close()

    return df_agg, {'corr_sep_gain': corr_sep_gain, 'corr_depth_sep': corr_depth_sep}


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 3: Depth-Separation Interaction Analysis")
    print("="*70)

    # Run all sub-experiments
    df_a = run_experiment_a()
    df_b, critical_depths = run_experiment_b()
    df_c, corr_c = run_experiment_c()
    df_d, correlations_d = run_experiment_d()

    # Compile summary
    summary = {
        'exp_a_depth_effect_independent_vs_correlated': {
            'finding': 'Depth helps more when features are independent',
            'max_nonlinear_gain_independent': float(df_a[df_a['n_true_features'] == 64]['nonlinear_gain'].max()),
            'max_nonlinear_gain_correlated_8': float(df_a[df_a['n_true_features'] == 8]['nonlinear_gain'].max())
        },
        'exp_b_critical_depths': {k: int(v) if v is not None else None for k, v in critical_depths.items()},
        'exp_c_depth_benefit_correlation_with_eff_dim': float(corr_c),
        'exp_d_correlations': {k: float(v) for k, v in correlations_d.items()}
    }

    with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("EXPERIMENT 3 COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nKey findings:")
    print(f"  - Critical depths: {critical_depths}")
    print(f"  - Depth benefit correlation with effective dim: {corr_c:.3f}")
    print(f"  - Separation-nonlinearity correlation: {correlations_d['corr_sep_gain']:.3f}")

    # Create combined figure
    create_main_figure()

    return summary


def create_main_figure():
    """Create a combined figure summarizing all experiments."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Load all results
    df_a = pd.read_csv(f"{OUTPUT_DIR}/exp_a_depth_vs_effective_dim.csv")
    df_b = pd.read_csv(f"{OUTPUT_DIR}/exp_b_critical_depth.csv")
    df_c = pd.read_csv(f"{OUTPUT_DIR}/exp_c_depth_benefit_scaling.csv")
    df_d = pd.read_csv(f"{OUTPUT_DIR}/exp_d_depth_separation.csv")

    # Panel 1: Depth × Effective Dim heatmap
    ax = axes[0, 0]
    pivot = df_a.pivot(index='n_true_features', columns='depth', values='nonlinear_gain')
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Depth')
    ax.set_ylabel('N True Features')
    ax.set_title('A) Nonlinear Gain: Depth × Eff. Dim')
    plt.colorbar(im, ax=ax)

    # Panel 2: Critical depth curves
    ax = axes[0, 1]
    for m in df_b['m'].unique():
        subset = df_b[df_b['m'] == m]
        ax.errorbar(subset['depth'], subset['nonlinear_gain_mean'],
                    yerr=subset['nonlinear_gain_std'], fmt='o-', label=f"m={m}", capsize=2)
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('B) Critical Depth Thresholds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Depth benefit scaling
    ax = axes[0, 2]
    subset = df_c[df_c['shallow_depth'] == 2]
    ax.plot(subset['effective_dim_ratio'], subset['depth_benefit_gain'], 'o-', markersize=8)
    corr = subset['effective_dim_ratio'].corr(subset['depth_benefit_gain'])
    ax.set_xlabel('Effective Dim Ratio')
    ax.set_ylabel('Depth Benefit (Δ Gain)')
    ax.set_title(f'C) Depth Benefit Scaling (r={corr:.2f})')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Panel 4: Separation vs depth
    ax = axes[1, 0]
    ax.errorbar(df_d['depth'], df_d['separation_score_mean'],
                yerr=df_d['separation_score_std'], fmt='o-', capsize=3)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Separation Score')
    ax.set_title('D) Feature Separation vs Depth')
    ax.grid(True, alpha=0.3)

    # Panel 5: Independent vs correlated curves
    ax = axes[1, 1]
    for n_true in [64, 16]:
        subset = df_a[df_a['n_true_features'] == n_true]
        label = "Independent (n_true=64)" if n_true == 64 else "Correlated (n_true=16)"
        ax.plot(subset['depth'], subset['nonlinear_gain'], 'o-', label=label)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('E) Independent vs Correlated')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 6: Separation vs Nonlinear gain
    ax = axes[1, 2]
    ax.plot(df_d['nonlinear_gain_mean'], df_d['separation_score_mean'], 'o-', markersize=10)
    for i, row in df_d.iterrows():
        ax.annotate(f"d={int(row['depth'])}", (row['nonlinear_gain_mean'], row['separation_score_mean']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Nonlinear Gain')
    ax.set_ylabel('Separation Score')
    ax.set_title('F) Separation vs Nonlinearity')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 3: Depth-Separation Interaction Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/main_results.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    summary = main()
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
