#!/usr/bin/env python3
"""
Experiment 4: Scaling Behavior with Input Dimension (n)

Hypothesis: The compression-driven nonlinearity finding (r=0.833) should scale with n.
Specifically:
1. The RATIO n/m (compression ratio) should matter, not absolute m values
2. Larger n should enable higher nonlinear gain (more features to compress)
3. The optimal depth may increase with n (more computation needed for larger problems)

Prior findings (all at n=64):
- Compression ratio dominates (r=0.833)
- Sweet spot at m=4 (compression=16x), l=2-3
- Sparsity ~0.10 optimal

We'll test n ∈ {32, 64, 128, 256} with matched compression ratios.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

SAVE_DIR = Path(__file__).parent
print(f"Saving to: {SAVE_DIR}")

# ===== Model and Data =====

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


def train_autoencoder(model, n_steps=200, batch_size=256, sparsity=0.1, lr=1e-3):
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


def measure_metrics(model, n_samples=500, sparsity=0.1):
    model.eval()

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)
        z = model.encode(x)

        # Fit linear encoder
        x_with_bias = torch.cat([x, torch.ones(n_samples, 1, device=device)], dim=1)
        W_linear = torch.linalg.lstsq(x_with_bias, z).solution
        z_linear = x_with_bias @ W_linear

        # Linearity score
        z_var = z.var(dim=0).sum()
        residual_var = (z - z_linear).var(dim=0).sum()
        linearity_score = 1 - (residual_var / z_var).item() if z_var > 0 else 1.0

        # Reconstruction quality
        x_recon_full, _ = model(x)
        x_recon_linear = model.decode(z_linear)

        mse_full = nn.functional.mse_loss(x_recon_full, x).item()
        mse_linear = nn.functional.mse_loss(x_recon_linear, x).item()
        nonlinear_gain = (mse_linear - mse_full) / (mse_linear + 1e-8)

        # Variance concentration
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (n_samples - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.flip(0)  # descending
        total_var = eigenvalues.sum()
        top1_var = (eigenvalues[0] / total_var).item() if total_var > 0 else 0

        # Effective dimensionality (participation ratio)
        eigenvalues_norm = eigenvalues / total_var
        eff_dim = (total_var ** 2 / (eigenvalues ** 2).sum()).item() if total_var > 0 else 0

    return {
        'linearity_score': linearity_score,
        'nonlinear_gain': nonlinear_gain,
        'mse_full': mse_full,
        'mse_linear': mse_linear,
        'top1_variance': top1_var,
        'effective_dim': eff_dim,
    }


def run_single(n, m, l, sparsity, n_steps=100, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model = Autoencoder(n, m, l).to(device)
    train_autoencoder(model, n_steps=n_steps, sparsity=sparsity)
    metrics = measure_metrics(model, sparsity=sparsity)

    return {
        'n': n, 'm': m, 'l': l, 'sparsity': sparsity,
        'compression_ratio': n / m,
        **metrics
    }


# ===== Main Experiment =====

if __name__ == "__main__":
    start_time = time.time()

    # Fixed parameters
    sparsity = 0.1  # optimal from Exp 3
    n_steps = 100   # reduced for speed (was 150-200)
    n_seeds = 2     # reduced for speed

    # Input dimension scaling
    n_values = [32, 64, 128, 256]

    # Test compression ratios (not absolute m)
    # compression_ratio = n / m, so m = n / compression_ratio
    compression_ratios = [4, 8, 16, 32]

    # Depths - test if optimal depth scales with n
    l_values = [1, 2, 3, 4]

    results = []
    total = len(n_values) * len(compression_ratios) * len(l_values) * n_seeds

    print(f"Running {total} configurations...")
    print(f"n values: {n_values}")
    print(f"compression ratios: {compression_ratios}")
    print(f"depths: {l_values}")

    pbar = tqdm(total=total)

    for n in n_values:
        for cr in compression_ratios:
            m = n // cr  # bottleneck dimension
            if m < 1:
                m = 1
            for l in l_values:
                for seed in range(n_seeds):
                    # Time check - stop if taking too long
                    elapsed = time.time() - start_time
                    if elapsed > 4 * 60:  # 4 minute hard limit
                        print(f"\nTime limit reached ({elapsed/60:.1f} min), stopping early...")
                        break

                    res = run_single(n, m, l, sparsity, n_steps, seed=seed)
                    res['seed'] = seed
                    res['actual_compression'] = n / m
                    results.append(res)
                    pbar.update(1)
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break

    pbar.close()

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")

    df = pd.DataFrame(results)
    df.to_csv(SAVE_DIR / 'results.csv', index=False)
    print(f"Saved {len(results)} results to {SAVE_DIR / 'results.csv'}")

    # ===== Analysis =====
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Aggregate over seeds
    agg = df.groupby(['n', 'm', 'l', 'compression_ratio']).agg({
        'nonlinear_gain': ['mean', 'std'],
        'top1_variance': 'mean',
        'linearity_score': 'mean',
        'mse_full': 'mean',
        'effective_dim': 'mean',
    }).reset_index()
    agg.columns = ['n', 'm', 'l', 'compression_ratio', 'nonlinear_gain_mean',
                   'nonlinear_gain_std', 'top1_variance', 'linearity_score',
                   'mse_full', 'effective_dim']

    # Key correlations
    print("\nKey Correlations (across all configs):")
    print(f"  n vs Nonlinear Gain: r = {df['n'].corr(df['nonlinear_gain']):.3f}")
    print(f"  Compression Ratio vs Nonlinear Gain: r = {df['compression_ratio'].corr(df['nonlinear_gain']):.3f}")
    print(f"  Depth vs Nonlinear Gain: r = {df['l'].corr(df['nonlinear_gain']):.3f}")
    print(f"  Top-1 Var vs Nonlinear Gain: r = {df['top1_variance'].corr(df['nonlinear_gain']):.3f}")
    print(f"  Effective Dim vs Nonlinear Gain: r = {df['effective_dim'].corr(df['nonlinear_gain']):.3f}")

    # Does nonlinear gain scale with n at fixed compression ratio?
    print("\nNonlinear gain by n (at each compression ratio):")
    print("-" * 60)
    for cr in sorted(df['compression_ratio'].unique()):
        subset = df[df['compression_ratio'] == cr]
        print(f"\n  Compression ratio = {cr}:")
        for n in sorted(subset['n'].unique()):
            sub2 = subset[subset['n'] == n]
            mean_gain = sub2['nonlinear_gain'].mean()
            print(f"    n={n:3d}, m={n//int(cr):2d}: nonlinear_gain = {mean_gain:.5f}")

        # Correlation within this compression ratio
        if len(subset) > 3:
            corr_n_gain = subset['n'].corr(subset['nonlinear_gain'])
            print(f"    [n vs gain within CR={cr}: r = {corr_n_gain:.3f}]")

    # Optimal depth by n
    print("\nOptimal depth by input dimension n:")
    print("-" * 40)
    for n in sorted(df['n'].unique()):
        subset = agg[agg['n'] == n]
        best_idx = subset['nonlinear_gain_mean'].idxmax()
        best = subset.loc[best_idx]
        print(f"  n={n:3d}: optimal l={int(best['l'])}, CR={best['compression_ratio']:.0f}, "
              f"gain={best['nonlinear_gain_mean']:.5f}")

    # ===== Plots =====

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Nonlinear gain vs n for each compression ratio
    ax = axes[0, 0]
    for cr in sorted(df['compression_ratio'].unique()):
        subset = agg[agg['compression_ratio'] == cr]
        subset = subset.groupby('n')['nonlinear_gain_mean'].mean().reset_index()
        ax.plot(subset['n'], subset['nonlinear_gain_mean'], 'o-',
                label=f'CR={int(cr)}', linewidth=2, markersize=8)
    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Input Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Plot 2: Nonlinear gain vs compression ratio for each n
    ax = axes[0, 1]
    for n in sorted(df['n'].unique()):
        subset = agg[agg['n'] == n]
        subset = subset.groupby('compression_ratio')['nonlinear_gain_mean'].mean().reset_index()
        ax.plot(subset['compression_ratio'], subset['nonlinear_gain_mean'], 'o-',
                label=f'n={n}', linewidth=2, markersize=8)
    ax.set_xlabel('Compression Ratio (n/m)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Compression Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Heatmap of nonlinear gain (n x compression_ratio), averaged over depth
    ax = axes[1, 0]
    pivot = agg.groupby(['n', 'compression_ratio'])['nonlinear_gain_mean'].mean().reset_index()
    pivot = pivot.pivot(index='n', columns='compression_ratio', values='nonlinear_gain_mean')
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([int(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Compression Ratio (n/m)')
    ax.set_ylabel('Input Dimension (n)')
    ax.set_title('Phase Diagram: n × Compression')
    plt.colorbar(im, ax=ax, label='Nonlinear Gain')

    # Plot 4: Optimal depth by n
    ax = axes[1, 1]
    optimal_l_by_n = []
    for n in sorted(df['n'].unique()):
        subset = agg[agg['n'] == n]
        best_idx = subset['nonlinear_gain_mean'].idxmax()
        best = subset.loc[best_idx]
        optimal_l_by_n.append({'n': n, 'optimal_l': best['l'],
                               'max_gain': best['nonlinear_gain_mean']})
    opt_df = pd.DataFrame(optimal_l_by_n)

    ax2 = ax.twinx()
    ax.bar(range(len(opt_df)), opt_df['optimal_l'], alpha=0.7, color='blue', label='Optimal depth')
    ax2.plot(range(len(opt_df)), opt_df['max_gain'], 'ro-', linewidth=2, markersize=10, label='Max gain')
    ax.set_xticks(range(len(opt_df)))
    ax.set_xticklabels(opt_df['n'])
    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Optimal Depth (l)', color='blue')
    ax2.set_ylabel('Max Nonlinear Gain', color='red')
    ax.set_title('Optimal Depth and Max Gain by n')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'scaling_input_dimension.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {SAVE_DIR / 'scaling_input_dimension.png'}")

    # Additional plot: Scaling laws
    fig, ax = plt.subplots(figsize=(8, 6))

    # For each n, get the max nonlinear gain achieved
    max_gains = df.groupby('n')['nonlinear_gain'].max().reset_index()
    ax.loglog(max_gains['n'], max_gains['nonlinear_gain'], 'bo-',
              linewidth=2, markersize=10, label='Max nonlinear gain')

    # Fit power law
    if len(max_gains) > 2 and max_gains['nonlinear_gain'].min() > 0:
        log_n = np.log(max_gains['n'].values)
        log_gain = np.log(max_gains['nonlinear_gain'].values.clip(min=1e-10))
        slope, intercept = np.polyfit(log_n, log_gain, 1)
        fit_gain = np.exp(intercept + slope * log_n)
        ax.loglog(max_gains['n'], fit_gain, 'r--', linewidth=2,
                  label=f'Power law: gain ∝ n^{slope:.2f}')
        print(f"\nScaling law: nonlinear_gain ∝ n^{slope:.2f}")

    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Max Nonlinear Gain')
    ax.set_title('Scaling of Nonlinear Benefit with Input Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(SAVE_DIR / 'scaling_law.png', dpi=150, bbox_inches='tight')
    print(f"Saved scaling law plot")

    plt.show()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total configs tested: {len(results)}")
    print(f"Best overall nonlinear gain: {df['nonlinear_gain'].max():.5f}")
    best_row = df.loc[df['nonlinear_gain'].idxmax()]
    print(f"  at n={int(best_row['n'])}, m={int(best_row['m'])}, l={int(best_row['l'])}, CR={best_row['compression_ratio']:.0f}")
