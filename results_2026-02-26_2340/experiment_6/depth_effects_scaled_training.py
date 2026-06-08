"""
Experiment 6: Depth Effects with Scaled Training

Key question: Does depth matter MORE or LESS when properly trained?

Prior findings:
- Exp 1-2: Depth correlates with nonlinear gain (r=0.49-0.53) at fixed 200 steps
- Exp 5: Fixed training causes massive underfitting at larger n
- Need to re-examine depth effects with scaled training budget

Design:
- Test l ∈ {1, 2, 3, 4, 5} at n=128 (good balance of signal and speed)
- Compare fixed vs scaled training (scaled = base_steps * (l + 1))
- Use optimal compression ratio CR=16
- Hypothesis: Depth effects will be STRONGER with proper training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Output directory
OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_6"


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


def train_autoencoder(model, n_steps, batch_size=256, sparsity=0.1, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        x = generate_sparse_data(batch_size, model.n, sparsity)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def measure_encoding_linearity(model, n_samples=1000, sparsity=0.1):
    model.eval()

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)
        z = model.encode(x)

        x_with_bias = torch.cat([x, torch.ones(n_samples, 1, device=device)], dim=1)
        W_linear = torch.linalg.lstsq(x_with_bias, z).solution
        z_linear = x_with_bias @ W_linear

        z_var = z.var(dim=0).sum()
        residual_var = (z - z_linear).var(dim=0).sum()
        linearity_score = 1 - (residual_var / z_var).item()

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


def measure_latent_stats(model, n_samples=1000, sparsity=0.1):
    """Measure latent space statistics (variance concentration, rank)."""
    model.eval()

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)
        z = model.encode(x)

        # Compute variance per latent dimension
        variances = z.var(dim=0).cpu().numpy()
        total_var = variances.sum()

        # Top-1 variance fraction
        top1_var = variances.max() / (total_var + 1e-8)

        # Effective dimensionality (participation ratio)
        eff_dim = (total_var ** 2) / ((variances ** 2).sum() + 1e-8)
        eff_dim_ratio = eff_dim / model.m

        # Rank estimation (using SVD)
        z_centered = z - z.mean(dim=0)
        U, S, V = torch.linalg.svd(z_centered, full_matrices=False)
        S = S.cpu().numpy()
        cumvar = np.cumsum(S**2) / (S**2).sum()
        rank_95 = np.searchsorted(cumvar, 0.95) + 1
        rank_ratio = rank_95 / model.m

    return {
        'top1_var': top1_var,
        'eff_dim': eff_dim,
        'eff_dim_ratio': eff_dim_ratio,
        'rank_95': rank_95,
        'rank_ratio': rank_ratio
    }


def run_depth_experiment():
    """Main experiment: depth effects with fixed vs scaled training."""

    n = 128
    compression_ratio = 16
    m = n // compression_ratio  # m = 8
    l_values = [1, 2, 3, 4, 5]
    sparsity = 0.1
    base_steps = 100
    n_seeds = 3

    results = []

    print("=" * 60)
    print("Experiment 6: Depth Effects with Scaled Training")
    print("=" * 60)
    print(f"n={n}, m={m}, CR={compression_ratio}")
    print(f"Depth values: {l_values}")
    print(f"Base steps: {base_steps}")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    for l in l_values:
        # Scaled steps: proportional to depth + 1 (more layers = more parameters = more training)
        # Also factor in n: use n/32 scaling from Exp 5
        scaled_steps = int(base_steps * (n / 32) * (l + 1) / 2)  # Moderate scaling with depth

        print(f"\n--- Depth l={l} ---")
        print(f"  Fixed steps: {base_steps}, Scaled steps: {scaled_steps}")

        for condition in ['fixed', 'scaled']:
            n_steps = base_steps if condition == 'fixed' else scaled_steps

            for seed in range(n_seeds):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                model = Autoencoder(n, m, l).to(device)
                n_params = sum(p.numel() for p in model.parameters())

                losses = train_autoencoder(model, n_steps, sparsity=sparsity)
                metrics = measure_encoding_linearity(model, sparsity=sparsity)
                latent_stats = measure_latent_stats(model, sparsity=sparsity)

                result = {
                    'n': n,
                    'm': m,
                    'l': l,
                    'compression_ratio': compression_ratio,
                    'condition': condition,
                    'n_steps': n_steps,
                    'seed': seed,
                    'n_params': n_params,
                    'final_loss': np.mean(losses[-10:]),
                    **metrics,
                    **latent_stats
                }
                results.append(result)

                print(f"  [{condition}] seed {seed}: nonlinear_gain={metrics['nonlinear_gain']:.6f}, "
                      f"mse={result['final_loss']:.6f}, top1_var={latent_stats['top1_var']:.3f}")

    return results


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    import pandas as pd
    df = pd.DataFrame(results)

    # Aggregate by condition and depth
    summary = df.groupby(['l', 'condition']).agg({
        'nonlinear_gain': ['mean', 'std'],
        'mse_full': ['mean', 'std'],
        'linearity_score': 'mean',
        'top1_var': 'mean',
        'eff_dim_ratio': 'mean',
        'rank_ratio': 'mean',
        'n_steps': 'first',
        'n_params': 'first'
    }).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Nonlinear gain by depth and condition
    ax = axes[0, 0]
    for condition in ['fixed', 'scaled']:
        subset = df[df['condition'] == condition]
        means = subset.groupby('l')['nonlinear_gain'].mean()
        stds = subset.groupby('l')['nonlinear_gain'].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='o', label=f'{condition} steps', capsize=5, markersize=8)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot 2: MSE by depth and condition
    ax = axes[0, 1]
    for condition in ['fixed', 'scaled']:
        subset = df[df['condition'] == condition]
        means = subset.groupby('l')['mse_full'].mean()
        stds = subset.groupby('l')['mse_full'].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='s', label=f'{condition} steps', capsize=5, markersize=8)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('MSE (Full Nonlinear)')
    ax.set_title('Reconstruction MSE vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Top-1 variance by depth
    ax = axes[0, 2]
    for condition in ['fixed', 'scaled']:
        subset = df[df['condition'] == condition]
        means = subset.groupby('l')['top1_var'].mean()
        stds = subset.groupby('l')['top1_var'].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='^', label=f'{condition} steps', capsize=5, markersize=8)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Top-1 Variance Fraction')
    ax.set_title('Variance Concentration vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Correlation between top1_var and nonlinear_gain
    ax = axes[1, 0]
    for condition, marker in [('fixed', 'o'), ('scaled', 's')]:
        subset = df[df['condition'] == condition]
        ax.scatter(subset['top1_var'], subset['nonlinear_gain'],
                  alpha=0.7, s=50, marker=marker, label=condition)
    ax.set_xlabel('Top-1 Variance Fraction')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Variance Concentration → Nonlinear Gain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot 5: Parameter count vs nonlinear gain
    ax = axes[1, 1]
    for condition, marker in [('fixed', 'o'), ('scaled', 's')]:
        subset = df[df['condition'] == condition]
        ax.scatter(subset['n_params'], subset['nonlinear_gain'],
                  alpha=0.7, s=50, marker=marker, label=condition)
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Model Capacity → Nonlinear Gain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot 6: Effective dimensionality ratio by depth
    ax = axes[1, 2]
    for condition in ['fixed', 'scaled']:
        subset = df[df['condition'] == condition]
        means = subset.groupby('l')['eff_dim_ratio'].mean()
        stds = subset.groupby('l')['eff_dim_ratio'].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='d', label=f'{condition} steps', capsize=5, markersize=8)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Effective Dim Ratio')
    ax.set_title('Latent Space Efficiency vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/depth_effects_scaled.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/depth_effects_scaled.png")

    # Compute key statistics
    stats = {}

    for condition in ['fixed', 'scaled']:
        subset = df[df['condition'] == condition]

        # Depth vs nonlinear gain correlation
        r_depth_gain = np.corrcoef(subset['l'], subset['nonlinear_gain'])[0, 1]
        stats[f'{condition}_depth_vs_gain'] = r_depth_gain

        # Top1 var vs nonlinear gain correlation
        r_var_gain = np.corrcoef(subset['top1_var'], subset['nonlinear_gain'])[0, 1]
        stats[f'{condition}_top1var_vs_gain'] = r_var_gain

        # Best depth
        best_l = subset.groupby('l')['nonlinear_gain'].mean().idxmax()
        best_gain = subset.groupby('l')['nonlinear_gain'].mean().max()
        stats[f'{condition}_best_depth'] = best_l
        stats[f'{condition}_best_gain'] = best_gain

    # Improvement ratio
    fixed_best = stats['fixed_best_gain']
    scaled_best = stats['scaled_best_gain']
    stats['scaled_vs_fixed_ratio'] = scaled_best / (fixed_best + 1e-10)

    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save results
    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump({'results': results, 'stats': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in stats.items()}}, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR}/results.json")

    return summary, stats


if __name__ == "__main__":
    results = run_depth_experiment()
    summary, stats = analyze_and_plot(results)
