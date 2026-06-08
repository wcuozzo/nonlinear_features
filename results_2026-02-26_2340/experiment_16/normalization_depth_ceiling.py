"""
Experiment 16: Layer Normalization to Push the Depth Ceiling

Key question: Can normalization layers (without skip connections) stabilize training
of very deep networks (l≥8) and maintain/improve nonlinear encoding?

Prior findings:
- Exp 14: Depth ceiling at l≈7-8 where networks collapse to linear encoding
- Exp 15: Residual connections HURT nonlinear encoding (bias toward linearity)
- Exp 15 suggested: "Test layer normalization or batch normalization WITHOUT skip
  connections—these may stabilize gradients without biasing toward linearity"

Hypothesis:
1. Layer normalization will stabilize gradient flow at depths l≥8
2. Unlike skip connections, normalization should NOT bias toward linearity
3. Normalized deep networks may achieve positive nonlinear gain beyond the ceiling
4. The optimal depth may increase beyond l=7 with normalization

Design:
- Compare: standard vs layer-normalized vs batch-normalized autoencoders
- Fixed n=128, m=8 (CR=16), α=0.1 (best from prior experiments)
- Test l ∈ {4, 6, 8, 10} to see if normalization helps beyond ceiling
- Moderate training (scaled by depth)
- 3 seeds per configuration
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_16"


class StandardAutoencoder(nn.Module):
    """Standard autoencoder without normalization."""
    def __init__(self, n: int, m: int, l: int, negative_slope: float):
        super().__init__()
        self.n = n
        self.m = m
        self.l = l
        self.negative_slope = negative_slope
        self.norm_type = 'none'

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


class LayerNormAutoencoder(nn.Module):
    """Autoencoder with LayerNorm after each linear layer (before activation)."""
    def __init__(self, n: int, m: int, l: int, negative_slope: float):
        super().__init__()
        self.n = n
        self.m = m
        self.l = l
        self.negative_slope = negative_slope
        self.norm_type = 'layernorm'

        # Encoder: Linear -> LayerNorm -> Activation
        encoder_layers = []
        for i in range(l):
            encoder_layers.append(nn.Linear(n, n))
            encoder_layers.append(nn.LayerNorm(n))
            encoder_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        encoder_layers.append(nn.Linear(n, m))
        # Optional: add LayerNorm at bottleneck
        encoder_layers.append(nn.LayerNorm(m))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: Linear -> LayerNorm -> Activation
        decoder_layers = []
        decoder_layers.append(nn.Linear(m, n))
        decoder_layers.append(nn.LayerNorm(n))
        decoder_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        for i in range(l - 1):
            decoder_layers.append(nn.Linear(n, n))
            decoder_layers.append(nn.LayerNorm(n))
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


class BatchNormAutoencoder(nn.Module):
    """Autoencoder with BatchNorm after each linear layer (before activation)."""
    def __init__(self, n: int, m: int, l: int, negative_slope: float):
        super().__init__()
        self.n = n
        self.m = m
        self.l = l
        self.negative_slope = negative_slope
        self.norm_type = 'batchnorm'

        # Encoder: Linear -> BatchNorm -> Activation
        encoder_layers = []
        for i in range(l):
            encoder_layers.append(nn.Linear(n, n))
            encoder_layers.append(nn.BatchNorm1d(n))
            encoder_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        encoder_layers.append(nn.Linear(n, m))
        encoder_layers.append(nn.BatchNorm1d(m))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: Linear -> BatchNorm -> Activation
        decoder_layers = []
        decoder_layers.append(nn.Linear(m, n))
        decoder_layers.append(nn.BatchNorm1d(n))
        decoder_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        for i in range(l - 1):
            decoder_layers.append(nn.Linear(n, n))
            decoder_layers.append(nn.BatchNorm1d(n))
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
    """Main experiment: compare standard vs normalized autoencoders across depths."""

    # Fixed parameters (using best config from prior experiments)
    n = 128
    m = 8  # CR=16
    alpha = 0.1  # Best for n=128
    sparsity = 0.1

    # Depths to test: include depths beyond the ceiling (l≥8)
    depths = [4, 6, 8, 10]

    # Architecture types
    norm_types = ['none', 'layernorm', 'batchnorm']

    # Training scaling
    base_steps = 80  # Per depth level
    n_seeds = 3

    all_results = []

    print("=" * 60)
    print("Experiment 16: Normalization to Push the Depth Ceiling")
    print("=" * 60)
    print(f"n={n}, m={m}, CR={n//m}, α={alpha}")
    print(f"Depths: {depths}")
    print(f"Normalization types: {norm_types}")
    print(f"Base steps: {base_steps} × (l+1)")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(depths) * len(norm_types) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for norm_type in norm_types:
        for l in depths:
            # Scale training with depth
            n_steps = base_steps * (l + 1)

            for seed in range(n_seeds):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                if norm_type == 'none':
                    model = StandardAutoencoder(n, m, l, negative_slope=alpha).to(device)
                elif norm_type == 'layernorm':
                    model = LayerNormAutoencoder(n, m, l, negative_slope=alpha).to(device)
                else:  # batchnorm
                    model = BatchNormAutoencoder(n, m, l, negative_slope=alpha).to(device)

                n_params = sum(p.numel() for p in model.parameters())

                final_loss = train_model(model, n_steps, sparsity=sparsity)

                metrics = measure_encoding_linearity(model, sparsity=sparsity)

                result = {
                    'norm_type': norm_type,
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

                pbar.set_postfix({'norm': norm_type, 'l': l, 'gain': f"{metrics['nonlinear_gain']:.5f}"})
                pbar.update(1)

    pbar.close()
    return all_results


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    norm_types = ['none', 'layernorm', 'batchnorm']
    depths = sorted(list(set(r['l'] for r in results)))

    # Aggregate by norm type and depth
    norm_depth_stats = {}
    for norm_type in norm_types:
        norm_depth_stats[norm_type] = {}
        for l in depths:
            subset = [r for r in results if r['norm_type'] == norm_type and r['l'] == l]
            if subset:
                gains = [r['nonlinear_gain'] for r in subset]
                mses = [r['mse_full'] for r in subset]
                linearities = [r['linearity_score'] for r in subset]

                norm_depth_stats[norm_type][l] = {
                    'mean_gain': np.mean(gains),
                    'std_gain': np.std(gains),
                    'max_gain': np.max(gains),
                    'min_gain': np.min(gains),
                    'mean_mse': np.mean(mses),
                    'mean_linearity': np.mean(linearities),
                    'n_steps': subset[0]['n_steps'],
                    'n_params': subset[0]['n_params'],
                    'positive_count': sum(1 for g in gains if g > 0)
                }

    # Correlations by norm type
    print(f"\n  DEPTH vs NONLINEAR GAIN CORRELATIONS:")
    for norm_type in norm_types:
        subset = [r for r in results if r['norm_type'] == norm_type]
        all_l = [r['l'] for r in subset]
        all_gain = [r['nonlinear_gain'] for r in subset]
        corr = np.corrcoef(all_l, all_gain)[0, 1]
        print(f"    {norm_type}: r = {corr:.3f}")

    # Results table
    print(f"\n  COMPARISON TABLE:")
    print(f"  {'Norm':>10} | {'l':>2} | {'Steps':>5} | {'Gain (mean±std)':>18} | {'MSE':>8} | {'Lin':>6} | {'Pos':>3}")
    print(f"  {'-'*10}-+-{'-'*2}-+-{'-'*5}-+-{'-'*18}-+-{'-'*8}-+-{'-'*6}-+-{'-'*3}")

    for l in depths:
        for norm_type in norm_types:
            if l in norm_depth_stats[norm_type]:
                s = norm_depth_stats[norm_type][l]
                print(f"  {norm_type:>10} | {l:>2} | {s['n_steps']:>5} | {s['mean_gain']:.5f} ± {s['std_gain']:.5f} | {s['mean_mse']:.5f} | {s['mean_linearity']:.3f} | {s['positive_count']}/3")

    # Check which depths show improvement over baseline
    print(f"\n  IMPROVEMENT OVER BASELINE (none) BY DEPTH:")
    print(f"  {'l':>2} | {'None':>10} | {'LayerNorm':>10} | {'BatchNorm':>10} | {'Best':>10}")
    print(f"  {'-'*2}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for l in depths:
        none_gain = norm_depth_stats['none'][l]['mean_gain']
        ln_gain = norm_depth_stats['layernorm'][l]['mean_gain']
        bn_gain = norm_depth_stats['batchnorm'][l]['mean_gain']
        best = max([(none_gain, 'none'), (ln_gain, 'layernorm'), (bn_gain, 'batchnorm')], key=lambda x: x[0])
        print(f"  {l:>2} | {none_gain:>10.5f} | {ln_gain:>10.5f} | {bn_gain:>10.5f} | {best[1]:>10}")

    # Check if normalization overcomes the depth ceiling
    print(f"\n  DEPTH CEILING ANALYSIS (l≥8):")
    for norm_type in norm_types:
        gains_at_depth8plus = [r['nonlinear_gain'] for r in results if r['norm_type'] == norm_type and r['l'] >= 8]
        mean_gain_8plus = np.mean(gains_at_depth8plus)
        positive_count = sum(1 for g in gains_at_depth8plus if g > 0)
        print(f"    {norm_type}: Mean gain = {mean_gain_8plus:.5f}, positive in {positive_count}/{len(gains_at_depth8plus)} runs")

    # Variance comparison
    print(f"\n  VARIANCE COMPARISON (stability):")
    for norm_type in norm_types:
        all_stds = [norm_depth_stats[norm_type][l]['std_gain'] for l in depths]
        mean_std = np.mean(all_stds)
        print(f"    {norm_type}: Mean std across depths = {mean_std:.5f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'none': 'blue', 'layernorm': 'green', 'batchnorm': 'orange'}
    markers = {'none': 'o', 'layernorm': 's', 'batchnorm': '^'}

    # Plot 1: Nonlinear gain vs depth (all norm types)
    ax = axes[0, 0]
    for norm_type in norm_types:
        means = [norm_depth_stats[norm_type][l]['mean_gain'] for l in depths]
        stds = [norm_depth_stats[norm_type][l]['std_gain'] for l in depths]
        ax.errorbar(depths, means, yerr=stds, marker=markers[norm_type], capsize=3,
                   linewidth=2, markersize=8, label=norm_type, color=colors[norm_type])
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5, label='Ceiling (l=8)')
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth by Normalization Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 2: MSE vs depth (all norm types)
    ax = axes[0, 1]
    for norm_type in norm_types:
        mses = [norm_depth_stats[norm_type][l]['mean_mse'] for l in depths]
        ax.plot(depths, mses, marker=markers[norm_type], linewidth=2, markersize=8,
               label=norm_type, color=colors[norm_type])
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 3: Gain difference from baseline
    ax = axes[1, 0]
    width = 0.35
    x = np.arange(len(depths))
    none_gains = [norm_depth_stats['none'][l]['mean_gain'] for l in depths]
    ln_diffs = [norm_depth_stats['layernorm'][l]['mean_gain'] - none_gains[i] for i, l in enumerate(depths)]
    bn_diffs = [norm_depth_stats['batchnorm'][l]['mean_gain'] - none_gains[i] for i, l in enumerate(depths)]
    ax.bar(x - width/2, ln_diffs, width, label='LayerNorm - None', color='green', alpha=0.7)
    ax.bar(x + width/2, bn_diffs, width, label='BatchNorm - None', color='orange', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Gain Difference from Baseline')
    ax.set_title('Normalization Advantage by Depth')
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Linearity vs depth (all norm types)
    ax = axes[1, 1]
    for norm_type in norm_types:
        linearities = [norm_depth_stats[norm_type][l]['mean_linearity'] for l in depths]
        ax.plot(depths, linearities, marker=markers[norm_type], linewidth=2, markersize=8,
               label=norm_type, color=colors[norm_type])
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Linearity Score')
    ax.set_title('Encoding Linearity vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/normalization_comparison.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/normalization_comparison.png")

    # Find best configurations
    best_overall = max(results, key=lambda x: x['nonlinear_gain'])
    best_by_norm = {}
    for norm_type in norm_types:
        subset = [r for r in results if r['norm_type'] == norm_type]
        best_by_norm[norm_type] = max(subset, key=lambda x: x['nonlinear_gain'])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for norm_type in norm_types:
        b = best_by_norm[norm_type]
        print(f"  Best {norm_type}: l={b['l']}, gain={b['nonlinear_gain']:.5f}")
    print(f"  Overall best: {best_overall['norm_type']} at l={best_overall['l']}, gain={best_overall['nonlinear_gain']:.5f}")

    # Calculate norm type wins per depth
    norm_wins = {nt: 0 for nt in norm_types}
    for l in depths:
        gains = [(norm_depth_stats[nt][l]['mean_gain'], nt) for nt in norm_types]
        winner = max(gains, key=lambda x: x[0])[1]
        norm_wins[winner] += 1
    print(f"\n  Wins by depth: none={norm_wins['none']}, layernorm={norm_wins['layernorm']}, batchnorm={norm_wins['batchnorm']}")

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
        'norm_depth_stats': norm_depth_stats,
        'best_by_norm': {nt: {'l': b['l'], 'gain': b['nonlinear_gain']} for nt, b in best_by_norm.items()},
        'best_overall': {'norm_type': best_overall['norm_type'], 'l': best_overall['l'], 'gain': best_overall['nonlinear_gain']},
        'norm_wins': norm_wins
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
