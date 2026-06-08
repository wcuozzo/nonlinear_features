"""
Experiment 2: Variance Concentration Mechanism

Hypothesis: Concentrating variance in fewer latent dimensions (not just low rank)
enables nonlinear encoding. We test this by:
1. Measuring variance concentration profiles (top-k variance explained)
2. Sweeping compression ratio to see if severe compression forces variance concentration
3. Testing whether variance concentration predicts nonlinear gain better than rank

Based on Exp 1 finding: rank doesn't predict nonlinearity (r=-0.012), but
top-1 variance does (r=0.48).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

SAVE_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_2"

# ====================
# Model and Data
# ====================

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


def train_autoencoder(model: Autoencoder, n_steps: int = 500, batch_size: int = 256,
                      sparsity: float = 0.1, lr: float = 1e-3) -> List[float]:
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


# ====================
# Variance Concentration Metrics
# ====================

def measure_variance_concentration(model: Autoencoder, n_samples: int = 1000,
                                   sparsity: float = 0.1) -> Dict[str, float]:
    """
    Measure how concentrated variance is in latent space.
    Returns top-1, top-2, top-4 variance fractions and Gini coefficient.
    """
    model.eval()

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)
        z = model.encode(x)

        # Per-dimension variance
        z_centered = z - z.mean(dim=0)
        variances = (z_centered ** 2).mean(dim=0)
        total_var = variances.sum()

        # Sort by variance (descending)
        sorted_vars, _ = torch.sort(variances, descending=True)
        sorted_vars = sorted_vars / total_var  # normalize

        # Cumulative variance explained
        cumvar = torch.cumsum(sorted_vars, dim=0)

        # Top-k variance fractions
        m = model.m
        top1_var = cumvar[0].item() if m >= 1 else 0
        top2_var = cumvar[min(1, m-1)].item() if m >= 2 else top1_var
        top4_var = cumvar[min(3, m-1)].item() if m >= 4 else top2_var
        tophalf_var = cumvar[m//2].item()

        # Gini coefficient (measure of concentration)
        # Perfect concentration: Gini = 1, uniform: Gini = 0
        n_dims = len(sorted_vars)
        indices = torch.arange(1, n_dims + 1, device=device, dtype=torch.float)
        gini = (2 * (indices * sorted_vars).sum() / (n_dims * sorted_vars.sum())
                - (n_dims + 1) / n_dims).item()

        # Effective dimensionality (using entropy)
        sorted_vars_safe = sorted_vars + 1e-10
        entropy = -torch.sum(sorted_vars_safe * torch.log(sorted_vars_safe))
        effective_dim = torch.exp(entropy).item()

    return {
        'top1_var': top1_var,
        'top2_var': top2_var,
        'top4_var': top4_var,
        'tophalf_var': tophalf_var,
        'gini_concentration': gini,
        'effective_dim': effective_dim,
        'effective_dim_ratio': effective_dim / m,
    }


def measure_encoding_metrics(model: Autoencoder, n_samples: int = 1000,
                             sparsity: float = 0.1) -> Dict[str, float]:
    """Measure linearity and reconstruction quality."""
    model.eval()

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)
        z = model.encode(x)

        # Fit best linear encoder
        x_with_bias = torch.cat([x, torch.ones(n_samples, 1, device=device)], dim=1)
        W_linear = torch.linalg.lstsq(x_with_bias, z).solution
        z_linear = x_with_bias @ W_linear

        # Linearity score
        z_var = z.var(dim=0).sum()
        residual_var = (z - z_linear).var(dim=0).sum()
        linearity_score = 1 - (residual_var / z_var).item()

        # Reconstruction quality
        x_recon_full, _ = model(x)
        x_recon_linear = model.decode(z_linear)

        mse_full = nn.functional.mse_loss(x_recon_full, x).item()
        mse_linear = nn.functional.mse_loss(x_recon_linear, x).item()
        nonlinear_gain = (mse_linear - mse_full) / (mse_linear + 1e-8)

    return {
        'linearity_score': linearity_score,
        'mse_full': mse_full,
        'mse_linear': mse_linear,
        'nonlinear_gain': nonlinear_gain,
    }


def run_single_experiment(n: int, m: int, l: int, sparsity: float = 0.1,
                          n_steps: int = 500, seed: int = 42) -> Dict:
    """Run one experiment configuration."""
    torch.manual_seed(seed)
    model = Autoencoder(n, m, l).to(device)

    losses = train_autoencoder(model, n_steps=n_steps, sparsity=sparsity)

    var_metrics = measure_variance_concentration(model, sparsity=sparsity)
    enc_metrics = measure_encoding_metrics(model, sparsity=sparsity)

    return {
        'n': n, 'm': m, 'l': l, 'sparsity': sparsity, 'seed': seed,
        'final_loss': np.mean(losses[-50:]),
        'compression_ratio': n / m,
        **var_metrics,
        **enc_metrics,
    }


# ====================
# Main Experiment
# ====================

def run_variance_concentration_experiment():
    """
    Test whether variance concentration predicts nonlinear gain.
    Sweep over compression ratios at fixed n, varying depth.
    """
    print("=" * 60)
    print("Experiment 2: Variance Concentration Mechanism")
    print("=" * 60)

    # Parameters chosen for speed (15 min budget)
    n = 64
    m_values = [4, 8, 12, 16, 24, 32]  # varying compression
    l_values = [1, 2, 3, 4]
    sparsity = 0.1
    n_steps = 200  # Fast for exploration
    n_seeds = 3

    results = []
    total = len(m_values) * len(l_values) * n_seeds

    print(f"\nRunning {total} configurations (n={n}, n_steps={n_steps})...")

    pbar = tqdm(total=total)
    for m in m_values:
        for l in l_values:
            for seed in range(n_seeds):
                try:
                    res = run_single_experiment(n, m, l, sparsity, n_steps, seed)
                    results.append(res)
                except Exception as e:
                    print(f"Error at m={m}, l={l}, seed={seed}: {e}")
                pbar.update(1)
    pbar.close()

    df = pd.DataFrame(results)

    # Aggregate by (m, l)
    agg_df = df.groupby(['m', 'l']).agg({
        'compression_ratio': 'first',
        'top1_var': ['mean', 'std'],
        'gini_concentration': ['mean', 'std'],
        'effective_dim_ratio': ['mean', 'std'],
        'nonlinear_gain': ['mean', 'std'],
        'linearity_score': ['mean', 'std'],
        'mse_full': ['mean', 'std'],
    }).reset_index()
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns]

    # Compute correlations
    print("\n" + "=" * 60)
    print("CORRELATIONS")
    print("=" * 60)

    correlations = {}
    for var in ['top1_var', 'gini_concentration', 'effective_dim_ratio', 'compression_ratio']:
        r = np.corrcoef(df[var], df['nonlinear_gain'])[0, 1]
        correlations[var] = r
        print(f"  {var} vs nonlinear_gain: r = {r:.3f}")

    # Depth correlations
    print("\nDepth correlations:")
    r_depth_gain = np.corrcoef(df['l'], df['nonlinear_gain'])[0, 1]
    r_depth_gini = np.corrcoef(df['l'], df['gini_concentration'])[0, 1]
    r_depth_top1 = np.corrcoef(df['l'], df['top1_var'])[0, 1]
    print(f"  depth vs nonlinear_gain: r = {r_depth_gain:.3f}")
    print(f"  depth vs gini_concentration: r = {r_depth_gini:.3f}")
    print(f"  depth vs top1_var: r = {r_depth_top1:.3f}")

    correlations['depth_vs_nonlinear_gain'] = r_depth_gain
    correlations['depth_vs_gini'] = r_depth_gini
    correlations['depth_vs_top1_var'] = r_depth_top1

    # ====================
    # Plots
    # ====================
    print("\nGenerating plots...")

    # Plot 1: Variance concentration vs Nonlinear Gain (scatter)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    scatter = ax.scatter(df['top1_var'], df['nonlinear_gain'],
                         c=df['compression_ratio'], cmap='viridis', alpha=0.7)
    ax.set_xlabel('Top-1 Variance Fraction')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title(f'Top-1 Var vs Nonlinear Gain (r={correlations["top1_var"]:.3f})')
    plt.colorbar(scatter, ax=ax, label='Compression Ratio')

    ax = axes[0, 1]
    scatter = ax.scatter(df['gini_concentration'], df['nonlinear_gain'],
                         c=df['compression_ratio'], cmap='viridis', alpha=0.7)
    ax.set_xlabel('Gini Concentration')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title(f'Gini vs Nonlinear Gain (r={correlations["gini_concentration"]:.3f})')
    plt.colorbar(scatter, ax=ax, label='Compression Ratio')

    ax = axes[1, 0]
    scatter = ax.scatter(df['effective_dim_ratio'], df['nonlinear_gain'],
                         c=df['l'], cmap='plasma', alpha=0.7)
    ax.set_xlabel('Effective Dim Ratio (eff_dim / m)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title(f'Eff Dim Ratio vs Nonlinear Gain (r={correlations["effective_dim_ratio"]:.3f})')
    plt.colorbar(scatter, ax=ax, label='Depth (l)')

    ax = axes[1, 1]
    scatter = ax.scatter(df['compression_ratio'], df['nonlinear_gain'],
                         c=df['l'], cmap='plasma', alpha=0.7, s=50)
    ax.set_xlabel('Compression Ratio (n/m)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title(f'Compression vs Nonlinear Gain (r={correlations["compression_ratio"]:.3f})')
    plt.colorbar(scatter, ax=ax, label='Depth (l)')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'variance_concentration_scatter.png'), dpi=150)
    plt.close()

    # Plot 2: Heatmap of variance concentration by (m, l)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Group means for heatmap
    heatmap_data = df.groupby(['m', 'l']).agg({
        'top1_var': 'mean',
        'gini_concentration': 'mean',
        'nonlinear_gain': 'mean',
    }).reset_index()

    for idx, (metric, title) in enumerate([
        ('top1_var', 'Top-1 Variance'),
        ('gini_concentration', 'Gini Concentration'),
        ('nonlinear_gain', 'Nonlinear Gain')
    ]):
        ax = axes[idx]
        pivot = heatmap_data.pivot(index='m', columns='l', values=metric)
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('Depth (l)')
        ax.set_ylabel('Bottleneck (m)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'variance_concentration_heatmap.png'), dpi=150)
    plt.close()

    # Plot 3: Variance concentration profiles by depth
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each depth, average the concentration metrics at high compression (m=4)
    high_comp_df = df[df['m'] == 4]
    for l_val in sorted(high_comp_df['l'].unique()):
        subset = high_comp_df[high_comp_df['l'] == l_val]
        avg_top1 = subset['top1_var'].mean()
        avg_top2 = subset['top2_var'].mean()
        avg_top4 = subset['top4_var'].mean()
        avg_tophalf = subset['tophalf_var'].mean()

        ax.plot([1, 2, 4, 'm/2'], [avg_top1, avg_top2, avg_top4, avg_tophalf],
                'o-', label=f'l={l_val}', linewidth=2, markersize=8)

    ax.set_xlabel('Top-k dimensions')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Variance Concentration Profile at High Compression (m=4)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'variance_profile_by_depth.png'), dpi=150)
    plt.close()

    # Save data
    df.to_csv(os.path.join(SAVE_DIR, 'results.csv'), index=False)
    agg_df.to_csv(os.path.join(SAVE_DIR, 'aggregated_results.csv'), index=False)

    print(f"\nSaved plots and data to {SAVE_DIR}")

    # Summary stats for log
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Find highest nonlinear gain configurations
    top_gain = df.nlargest(5, 'nonlinear_gain')[['m', 'l', 'top1_var', 'gini_concentration', 'nonlinear_gain']]
    print("\nTop 5 nonlinear gain configurations:")
    print(top_gain.to_string(index=False))

    # Find highest variance concentration configurations
    top_conc = df.nlargest(5, 'gini_concentration')[['m', 'l', 'top1_var', 'gini_concentration', 'nonlinear_gain']]
    print("\nTop 5 variance concentration configurations:")
    print(top_conc.to_string(index=False))

    return df, correlations


if __name__ == "__main__":
    df, correlations = run_variance_concentration_experiment()
