"""
Experiment 5: Scaled Training Steps to Test Underfitting Hypothesis

Key question from Experiment 4: Is the n^-1.8 scaling of nonlinear gain REAL
or an artifact of underfitting larger models?

Design:
- Test n ∈ {32, 64, 128, 256}
- Scale training steps proportionally: n_steps = base_steps * (n / 32)
- Compare with fixed steps as control
- Use optimal compression ratio (CR=16) from prior experiments
- Depth l=2 (sweet spot from Experiment 2)
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
OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_5"

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


def run_scaled_training_experiment():
    """Main experiment: compare fixed vs scaled training steps."""

    n_values = [32, 64, 128, 256]
    compression_ratio = 16  # Optimal from prior experiments
    l = 2  # Sweet spot depth
    sparsity = 0.1
    base_steps = 100  # Base for n=32
    n_seeds = 3

    results = []

    print("=" * 60)
    print("Experiment 5: Scaled Training Steps")
    print("=" * 60)
    print(f"n values: {n_values}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"Depth: {l}")
    print(f"Base steps (at n=32): {base_steps}")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    for n in n_values:
        m = n // compression_ratio

        # Scaled steps: proportional to n
        scaled_steps = int(base_steps * (n / 32))

        print(f"\n--- n={n}, m={m} ---")
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
                    **metrics
                }
                results.append(result)

                print(f"  [{condition}] seed {seed}: nonlinear_gain={metrics['nonlinear_gain']:.6f}, "
                      f"mse={result['final_loss']:.6f}")

    return results


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    import pandas as pd
    df = pd.DataFrame(results)

    # Aggregate by condition and n
    summary = df.groupby(['n', 'condition']).agg({
        'nonlinear_gain': ['mean', 'std'],
        'mse_full': ['mean', 'std'],
        'linearity_score': 'mean',
        'n_steps': 'first'
    }).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    # Key comparison: fixed vs scaled
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Nonlinear gain by n and condition
    ax = axes[0]
    for condition in ['fixed', 'scaled']:
        subset = df[df['condition'] == condition]
        means = subset.groupby('n')['nonlinear_gain'].mean()
        stds = subset.groupby('n')['nonlinear_gain'].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='o', label=f'{condition} steps', capsize=5, markersize=8)
    ax.set_xlabel('Input dimension (n)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain: Fixed vs Scaled Training')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: MSE by n and condition
    ax = axes[1]
    for condition in ['fixed', 'scaled']:
        subset = df[df['condition'] == condition]
        means = subset.groupby('n')['mse_full'].mean()
        stds = subset.groupby('n')['mse_full'].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker='s', label=f'{condition} steps', capsize=5, markersize=8)
    ax.set_xlabel('Input dimension (n)')
    ax.set_ylabel('MSE (Full Nonlinear)')
    ax.set_title('Reconstruction MSE by Condition')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Plot 3: Scaling behavior (log-log)
    ax = axes[2]
    for condition in ['fixed', 'scaled']:
        subset = df[df['condition'] == condition]
        means = subset.groupby('n')['nonlinear_gain'].mean()
        # Fit power law
        log_n = np.log(means.index)
        log_gain = np.log(np.clip(means.values, 1e-8, None))
        valid = np.isfinite(log_gain)
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(log_n[valid], log_gain[valid], 1)
            ax.scatter(means.index, means.values, s=100, label=f'{condition}: slope={slope:.2f}')
            fit_line = np.exp(intercept) * np.array(means.index) ** slope
            ax.plot(means.index, fit_line, '--', alpha=0.5)
    ax.set_xlabel('Input dimension (n)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Scaling Behavior (log-log)')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/scaled_training_comparison.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/scaled_training_comparison.png")

    # Compute key statistics
    stats = {}

    for condition in ['fixed', 'scaled']:
        subset = df[df['condition'] == condition]
        means = subset.groupby('n')['nonlinear_gain'].mean()
        log_n = np.log(means.index)
        log_gain = np.log(np.clip(means.values, 1e-8, None))
        valid = np.isfinite(log_gain)
        if valid.sum() >= 2:
            slope, _ = np.polyfit(log_n[valid], log_gain[valid], 1)
            stats[f'{condition}_scaling_exponent'] = slope

        # Compute correlation with n
        r = np.corrcoef(subset['n'], subset['nonlinear_gain'])[0, 1]
        stats[f'{condition}_n_vs_gain_corr'] = r

    # Ratio of gains at largest n
    fixed_at_256 = df[(df['condition'] == 'fixed') & (df['n'] == 256)]['nonlinear_gain'].mean()
    scaled_at_256 = df[(df['condition'] == 'scaled') & (df['n'] == 256)]['nonlinear_gain'].mean()
    stats['scaled_vs_fixed_ratio_at_n256'] = scaled_at_256 / (fixed_at_256 + 1e-10)

    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump({'results': results, 'stats': stats}, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR}/results.json")

    return summary, stats


if __name__ == "__main__":
    results = run_scaled_training_experiment()
    summary, stats = analyze_and_plot(results)
