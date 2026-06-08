"""
Experiment 14: Extended Depth Search at the Sweet Spot n=128

Key question: Does the depth-gain relationship plateau, or does it continue increasing beyond l=5-6?

Prior findings:
- Exp 10: At n=128, optimal depth was l=6 (our maximum tested), achieving gain=0.00845
- Exp 13: n=128 with α=0.1, l=5 achieved gain=0.01204 (highest ever)
- All prior experiments capped depth at l=5 or l=6

Hypothesis:
1. At n=128 (the sweet spot), nonlinear gain may continue increasing with depth beyond l=6
2. There should be an eventual plateau or diminishing returns
3. Very deep networks may require even more training to realize their potential

Design:
- Fixed n=128, m=8 (CR=16), α=0.1 (best from Exp 13 at n=128)
- Test l ∈ {3, 4, 5, 6, 7, 8, 9, 10}
- Scaled training: steps ∝ (l+1) to account for depth (more generous than prior)
- 3 seeds for reliability
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_14"


class Autoencoder(nn.Module):
    def __init__(self, n: int, m: int, l: int, negative_slope: float):
        super().__init__()
        self.n = n
        self.m = m
        self.l = l
        self.negative_slope = negative_slope

        encoder_layers = []
        for i in range(l):
            encoder_layers.append(nn.Linear(n, n))
            encoder_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        encoder_layers.append(nn.Linear(n, m))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        decoder_layers.append(nn.Linear(m, n))
        decoder_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        for i in range(l - 1):
            decoder_layers.append(nn.Linear(n, n))
            decoder_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
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


def measure_encoding_linearity(model, n_samples=500, sparsity=0.1):
    """Measure linearity metrics for the encoder."""
    model.eval()

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)
        z = model.encode(x)

        # Fit best linear approximation
        x_with_bias = torch.cat([x, torch.ones(n_samples, 1, device=device)], dim=1)
        W_linear = torch.linalg.lstsq(x_with_bias, z).solution
        z_linear = x_with_bias @ W_linear

        # Linearity score
        z_var = z.var(dim=0).sum()
        residual_var = (z - z_linear).var(dim=0).sum()
        linearity_score = 1 - (residual_var / (z_var + 1e-8)).item()

        # Reconstruction comparison
        x_recon_full, _ = model(x)
        x_recon_linear = model.decode(z_linear)

        mse_full = nn.functional.mse_loss(x_recon_full, x).item()
        mse_linear = nn.functional.mse_loss(x_recon_linear, x).item()
        nonlinear_gain = (mse_linear - mse_full) / (mse_linear + 1e-8)

        # Effective dimensionality of latent space
        z_centered = z - z.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(z_centered)
        sv_normalized = singular_values / singular_values.sum()
        effective_dim = (1.0 / (sv_normalized ** 2).sum()).item()
        effective_dim_ratio = effective_dim / model.m

        # Top-1 variance fraction
        sv_squared = singular_values ** 2
        top1_var = (sv_squared[0] / sv_squared.sum()).item()

    model.train()
    return {
        'linearity_score': linearity_score,
        'mse_full': mse_full,
        'mse_linear': mse_linear,
        'nonlinear_gain': nonlinear_gain,
        'effective_dim': effective_dim,
        'effective_dim_ratio': effective_dim_ratio,
        'top1_var': top1_var
    }


def train_model(model, n_steps, batch_size=256, sparsity=0.1, lr=1e-3):
    """Train model for n_steps."""
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

    return np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)


def run_experiment():
    """Main experiment: test extended depth range at n=128."""

    # Fixed parameters (using best config from Exp 13)
    n = 128
    m = 8  # CR=16
    alpha = 0.1  # Best for n=128 from Exp 13
    sparsity = 0.1

    # Extended depth range
    depths = [3, 4, 5, 6, 7, 8, 9, 10]

    # Generous training scaling
    base_steps = 100  # Per depth level
    n_seeds = 3

    all_results = []

    print("=" * 60)
    print("Experiment 14: Extended Depth Search at n=128")
    print("=" * 60)
    print(f"n={n}, m={m}, CR={n//m}, α={alpha}")
    print(f"Depths: {depths}")
    print(f"Base steps: {base_steps} × (l+1)")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(depths) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for l in depths:
        # Scale training linearly with depth
        n_steps = base_steps * (l + 1)

        for seed in range(n_seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            model = Autoencoder(n, m, l, negative_slope=alpha).to(device)
            n_params = sum(p.numel() for p in model.parameters())

            final_loss = train_model(model, n_steps, sparsity=sparsity)

            metrics = measure_encoding_linearity(model, sparsity=sparsity)

            result = {
                'n': n,
                'm': m,
                'l': l,
                'alpha': alpha,
                'compression_ratio': n // m,
                'seed': seed,
                'n_params': n_params,
                'n_steps': n_steps,
                'final_loss': final_loss,
                **metrics
            }
            all_results.append(result)

            pbar.set_postfix({'l': l, 'gain': f"{metrics['nonlinear_gain']:.5f}"})
            pbar.update(1)

    pbar.close()
    return all_results


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    depths = sorted(list(set(r['l'] for r in results)))

    # Aggregate by depth
    depth_stats = {}
    for l in depths:
        subset = [r for r in results if r['l'] == l]
        gains = [r['nonlinear_gain'] for r in subset]
        mses = [r['mse_full'] for r in subset]
        linearities = [r['linearity_score'] for r in subset]

        depth_stats[l] = {
            'mean_gain': np.mean(gains),
            'std_gain': np.std(gains),
            'max_gain': np.max(gains),
            'mean_mse': np.mean(mses),
            'mean_linearity': np.mean(linearities),
            'n_steps': subset[0]['n_steps'],
            'n_params': subset[0]['n_params']
        }

    # Overall correlations
    all_l = [r['l'] for r in results]
    all_gain = [r['nonlinear_gain'] for r in results]
    all_mse = [r['mse_full'] for r in results]
    all_linearity = [r['linearity_score'] for r in results]

    depth_gain_corr = np.corrcoef(all_l, all_gain)[0, 1]
    depth_mse_corr = np.corrcoef(all_l, all_mse)[0, 1]
    depth_linearity_corr = np.corrcoef(all_l, all_linearity)[0, 1]

    print(f"\n  CORRELATIONS:")
    print(f"    Depth vs Nonlinear Gain: r = {depth_gain_corr:.3f}")
    print(f"    Depth vs MSE: r = {depth_mse_corr:.3f}")
    print(f"    Depth vs Linearity: r = {depth_linearity_corr:.3f}")

    # Results table
    print(f"\n  DEPTH EFFECTS:")
    print(f"  {'l':>3} | {'Steps':>6} | {'Params':>8} | {'Gain (mean±std)':>18} | {'MSE':>8} | {'Linearity':>9}")
    print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*8}-+-{'-'*18}-+-{'-'*8}-+-{'-'*9}")

    for l in depths:
        s = depth_stats[l]
        print(f"  {l:>3} | {s['n_steps']:>6} | {s['n_params']:>8} | {s['mean_gain']:.5f} ± {s['std_gain']:.5f} | {s['mean_mse']:.5f} | {s['mean_linearity']:.5f}")

    # Find optimal depth
    mean_gains = [depth_stats[l]['mean_gain'] for l in depths]
    optimal_depth = depths[np.argmax(mean_gains)]
    max_gain = max(mean_gains)

    print(f"\n  OPTIMAL DEPTH: l={optimal_depth} with mean gain={max_gain:.5f}")

    # Check for plateau
    gains_array = np.array(mean_gains)
    if len(gains_array) >= 3:
        # Check if gains plateau (diminishing returns)
        first_half = gains_array[:len(gains_array)//2]
        second_half = gains_array[len(gains_array)//2:]
        first_half_growth = (first_half[-1] - first_half[0]) / (len(first_half) - 1) if len(first_half) > 1 else 0
        second_half_growth = (second_half[-1] - second_half[0]) / (len(second_half) - 1) if len(second_half) > 1 else 0

        print(f"\n  PLATEAU ANALYSIS:")
        print(f"    First half growth rate: {first_half_growth:.6f} per depth level")
        print(f"    Second half growth rate: {second_half_growth:.6f} per depth level")
        if second_half_growth < first_half_growth * 0.5:
            print(f"    ⚠️ Diminishing returns detected in second half")
        elif second_half_growth > first_half_growth:
            print(f"    📈 Growth accelerating - no plateau yet")
        else:
            print(f"    → Roughly linear growth - plateau not reached")

    # Check if we hit the ceiling
    if optimal_depth == max(depths):
        print(f"\n  ⚠️ Optimal depth is at maximum tested ({optimal_depth}). True optimum may be higher.")
    else:
        print(f"\n  ✓ Depth ceiling found at l={optimal_depth}. Gains decline beyond this depth.")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Nonlinear gain vs depth
    ax = axes[0, 0]
    means = [depth_stats[l]['mean_gain'] for l in depths]
    stds = [depth_stats[l]['std_gain'] for l in depths]
    ax.errorbar(depths, means, yerr=stds, marker='o', capsize=3, linewidth=2, markersize=8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title(f'Nonlinear Gain vs Depth (n=128, α=0.1)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Mark optimal
    ax.axvline(x=optimal_depth, color='green', linestyle=':', alpha=0.5, label=f'Optimal l={optimal_depth}')
    ax.legend()

    # Plot 2: MSE vs depth
    ax = axes[0, 1]
    mses = [depth_stats[l]['mean_mse'] for l in depths]
    ax.plot(depths, mses, marker='s', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error vs Depth')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 3: Linearity score vs depth
    ax = axes[1, 0]
    linearities = [depth_stats[l]['mean_linearity'] for l in depths]
    ax.plot(depths, linearities, marker='^', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Linearity Score')
    ax.set_title('Encoding Linearity vs Depth')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 4: Gain per parameter (efficiency)
    ax = axes[1, 1]
    params = [depth_stats[l]['n_params'] for l in depths]
    gains = [depth_stats[l]['mean_gain'] for l in depths]
    efficiency = [g / (p / 1e6) for g, p in zip(gains, params)]  # Gain per million parameters
    ax.plot(depths, efficiency, marker='d', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain / Million Params')
    ax.set_title('Parameter Efficiency vs Depth')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/extended_depth_search.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/extended_depth_search.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Depth range tested: l={min(depths)} to l={max(depths)}")
    print(f"  Depth-gain correlation: r = {depth_gain_corr:.3f}")
    print(f"  Optimal depth: l={optimal_depth}")
    print(f"  Maximum nonlinear gain: {max_gain:.5f}")
    print(f"  Best individual run: {max([r['nonlinear_gain'] for r in results]):.5f}")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    stats = {
        'depth_gain_corr': depth_gain_corr,
        'depth_mse_corr': depth_mse_corr,
        'depth_linearity_corr': depth_linearity_corr,
        'optimal_depth': optimal_depth,
        'max_mean_gain': max_gain,
        'max_individual_gain': max([r['nonlinear_gain'] for r in results]),
        'depth_stats': depth_stats
    }

    serializable_results = make_serializable(results)
    serializable_stats = make_serializable(stats)

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump({
            'results': serializable_results,
            'stats': serializable_stats
        }, f, indent=2)
    print(f"Saved results to {OUTPUT_DIR}/results.json")

    return serializable_stats


if __name__ == "__main__":
    results = run_experiment()
    stats = analyze_and_plot(results)
