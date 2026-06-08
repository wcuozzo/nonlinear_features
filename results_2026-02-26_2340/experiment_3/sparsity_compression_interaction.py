#!/usr/bin/env python3
"""
Experiment 3: Sparsity-Compression Interaction

Hypothesis: Sparsity modulates how much compression is needed to trigger nonlinear encoding.
- Prior work: Independence drives nonlinearity (r=-0.78)
- Exp 2: Compression is primary driver (r=0.605), sweet spot at m=4
- Theory: Sparser data should require LESS compression for nonlinearity (fewer features to encode)
         OR sparser data should require MORE compression (need to merge sparse activations)

We'll sweep sparsity x compression to find the phase boundary.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Save directory
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

        # Variance concentration (top-1 variance fraction)
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (n_samples - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.flip(0)  # descending
        total_var = eigenvalues.sum()
        top1_var = (eigenvalues[0] / total_var).item() if total_var > 0 else 0

    return {
        'linearity_score': linearity_score,
        'nonlinear_gain': nonlinear_gain,
        'mse_full': mse_full,
        'mse_linear': mse_linear,
        'top1_variance': top1_var,
    }


def run_single(n, m, l, sparsity, n_steps=150, seed=None):
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
    # Fixed parameters
    n = 64
    l = 2  # Use depth=2 (sweet spot from Exp 2)
    n_steps = 150
    n_seeds = 3

    # Sweep parameters
    sparsity_values = [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
    m_values = [4, 6, 8, 12, 16, 24, 32]

    results = []
    total = len(sparsity_values) * len(m_values) * n_seeds

    print(f"Running {total} configurations...")
    pbar = tqdm(total=total)

    for sparsity in sparsity_values:
        for m in m_values:
            for seed in range(n_seeds):
                res = run_single(n, m, l, sparsity, n_steps, seed=seed)
                res['seed'] = seed
                results.append(res)
                pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)
    df.to_csv(SAVE_DIR / 'results.csv', index=False)
    print(f"\nSaved {len(results)} results to {SAVE_DIR / 'results.csv'}")

    # ===== Analysis =====
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Aggregate over seeds
    agg = df.groupby(['sparsity', 'm', 'compression_ratio']).agg({
        'nonlinear_gain': ['mean', 'std'],
        'top1_variance': 'mean',
        'linearity_score': 'mean',
        'mse_full': 'mean'
    }).reset_index()
    agg.columns = ['sparsity', 'm', 'compression_ratio', 'nonlinear_gain_mean',
                   'nonlinear_gain_std', 'top1_variance', 'linearity_score', 'mse_full']

    # Key correlations
    print("\nKey Correlations (across all configs):")
    corr_sparsity_gain = df['sparsity'].corr(df['nonlinear_gain'])
    corr_compression_gain = df['compression_ratio'].corr(df['nonlinear_gain'])
    corr_top1_gain = df['top1_variance'].corr(df['nonlinear_gain'])

    print(f"  Sparsity vs Nonlinear Gain: r = {corr_sparsity_gain:.3f}")
    print(f"  Compression vs Nonlinear Gain: r = {corr_compression_gain:.3f}")
    print(f"  Top-1 Var vs Nonlinear Gain: r = {corr_top1_gain:.3f}")

    # Find optimal compression for each sparsity level
    print("\nOptimal compression ratio by sparsity:")
    print("-"*50)

    optimal_per_sparsity = []
    for sp in sparsity_values:
        subset = agg[agg['sparsity'] == sp]
        best_idx = subset['nonlinear_gain_mean'].idxmax()
        best_row = subset.loc[best_idx]
        optimal_per_sparsity.append({
            'sparsity': sp,
            'optimal_m': best_row['m'],
            'optimal_compression': best_row['compression_ratio'],
            'max_nonlinear_gain': best_row['nonlinear_gain_mean'],
            'top1_variance': best_row['top1_variance']
        })
        print(f"  sparsity={sp:.2f}: optimal m={int(best_row['m'])}, "
              f"compression={best_row['compression_ratio']:.1f}, "
              f"nonlinear_gain={best_row['nonlinear_gain_mean']:.5f}")

    optimal_df = pd.DataFrame(optimal_per_sparsity)

    # Correlation between sparsity and optimal compression
    corr_sparsity_opt_compression = optimal_df['sparsity'].corr(optimal_df['optimal_compression'])
    corr_sparsity_max_gain = optimal_df['sparsity'].corr(optimal_df['max_nonlinear_gain'])

    print(f"\n  Sparsity vs Optimal Compression: r = {corr_sparsity_opt_compression:.3f}")
    print(f"  Sparsity vs Max Achievable Gain: r = {corr_sparsity_max_gain:.3f}")

    # ===== Plots =====

    # Plot 1: Heatmap of nonlinear gain (sparsity x compression)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Heatmap
    ax = axes[0]
    pivot = agg.pivot(index='sparsity', columns='m', values='nonlinear_gain_mean')
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{s:.2f}' for s in pivot.index])
    ax.set_xlabel('Bottleneck dim (m)')
    ax.set_ylabel('Sparsity')
    ax.set_title('Nonlinear Gain')
    plt.colorbar(im, ax=ax)

    # Line plot: nonlinear gain vs compression for each sparsity
    ax = axes[1]
    for sp in sparsity_values[::2]:  # every other for clarity
        subset = agg[agg['sparsity'] == sp].sort_values('compression_ratio')
        ax.plot(subset['compression_ratio'], subset['nonlinear_gain_mean'],
                marker='o', label=f'sparsity={sp:.2f}')
    ax.set_xlabel('Compression Ratio (n/m)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Compression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Optimal compression vs sparsity
    ax = axes[2]
    ax.scatter(optimal_df['sparsity'], optimal_df['optimal_compression'], s=100, c='blue')
    ax.plot(optimal_df['sparsity'], optimal_df['optimal_compression'], 'b--', alpha=0.5)

    ax2 = ax.twinx()
    ax2.scatter(optimal_df['sparsity'], optimal_df['max_nonlinear_gain'], s=100, c='red', marker='^')
    ax2.plot(optimal_df['sparsity'], optimal_df['max_nonlinear_gain'], 'r--', alpha=0.5)

    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Optimal Compression Ratio', color='blue')
    ax2.set_ylabel('Max Nonlinear Gain', color='red')
    ax.set_title('Optimal Settings by Sparsity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'sparsity_compression_interaction.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {SAVE_DIR / 'sparsity_compression_interaction.png'}")

    # Plot 2: 3D surface view (optional)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df['compression_ratio'],
        df['sparsity'],
        c=df['nonlinear_gain'],
        s=50,
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label='Nonlinear Gain')
    ax.set_xlabel('Compression Ratio (n/m)')
    ax.set_ylabel('Sparsity')
    ax.set_title('Phase Diagram: Compression x Sparsity')
    plt.savefig(SAVE_DIR / 'phase_scatter.png', dpi=150, bbox_inches='tight')

    plt.show()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total configs tested: {len(results)}")
    print(f"Best overall nonlinear gain: {df['nonlinear_gain'].max():.5f}")
    best_row = df.loc[df['nonlinear_gain'].idxmax()]
    print(f"  at m={int(best_row['m'])}, sparsity={best_row['sparsity']:.2f}")
