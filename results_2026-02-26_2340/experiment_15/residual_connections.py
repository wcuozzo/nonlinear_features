"""
Experiment 15: Residual Connections to Overcome Depth Ceiling

Key question: Can skip connections enable stable training of very deep networks (l≥8),
overcoming the depth ceiling discovered in Experiment 14?

Prior findings:
- Exp 14: Networks with l≥8 collapse to linear encoding (gain=0) due to optimization issues
- Exp 14: l=7 achieves highest gain but with high variance (unstable)
- The depth ceiling appears to be an optimization problem, not architectural limitation

Hypothesis:
1. Residual connections will stabilize training at depths l≥8
2. Skip connections will allow networks to achieve positive nonlinear gain beyond the ceiling
3. The optimal depth with residual connections may be higher than l=7
4. Residual networks may show lower variance (more stable) at all depths

Design:
- Compare: standard autoencoder vs residual autoencoder
- Fixed n=128, m=8 (CR=16), α=0.1 (best from prior experiments)
- Test l ∈ {4, 6, 8, 10, 12} to see if residual helps beyond ceiling
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_15"


class StandardAutoencoder(nn.Module):
    """Standard autoencoder without skip connections."""
    def __init__(self, n: int, m: int, l: int, negative_slope: float):
        super().__init__()
        self.n = n
        self.m = m
        self.l = l
        self.negative_slope = negative_slope
        self.has_residual = False

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


class ResidualBlock(nn.Module):
    """A simple residual block: x + f(x)"""
    def __init__(self, n: int, negative_slope: float):
        super().__init__()
        self.linear = nn.Linear(n, n)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        return x + self.activation(self.linear(x))


class ResidualAutoencoder(nn.Module):
    """Autoencoder with residual/skip connections in encoder and decoder."""
    def __init__(self, n: int, m: int, l: int, negative_slope: float):
        super().__init__()
        self.n = n
        self.m = m
        self.l = l
        self.negative_slope = negative_slope
        self.has_residual = True

        # Encoder: l residual blocks followed by projection to bottleneck
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(n, negative_slope) for _ in range(l)
        ])
        self.encoder_proj = nn.Linear(n, m)

        # Decoder: projection from bottleneck followed by l residual blocks
        self.decoder_proj = nn.Linear(m, n)
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock(n, negative_slope) for _ in range(l)
        ])
        # Final linear layer without activation
        self.decoder_out = nn.Linear(n, n)

    def encode(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_proj(x)

    def decode(self, z):
        x = self.decoder_proj(z)
        x = nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        for block in self.decoder_blocks:
            x = block(x)
        return self.decoder_out(x)

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
    """Main experiment: compare standard vs residual autoencoders across depths."""

    # Fixed parameters (using best config from prior experiments)
    n = 128
    m = 8  # CR=16
    alpha = 0.1  # Best for n=128
    sparsity = 0.1

    # Depths to test: include depths beyond the ceiling (l≥8)
    depths = [4, 6, 8, 10, 12]

    # Architecture types
    architectures = ['standard', 'residual']

    # Training scaling
    base_steps = 80  # Per depth level
    n_seeds = 3

    all_results = []

    print("=" * 60)
    print("Experiment 15: Residual Connections to Overcome Depth Ceiling")
    print("=" * 60)
    print(f"n={n}, m={m}, CR={n//m}, α={alpha}")
    print(f"Depths: {depths}")
    print(f"Architectures: {architectures}")
    print(f"Base steps: {base_steps} × (l+1)")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(depths) * len(architectures) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for arch in architectures:
        for l in depths:
            # Scale training with depth
            n_steps = base_steps * (l + 1)

            for seed in range(n_seeds):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                if arch == 'standard':
                    model = StandardAutoencoder(n, m, l, negative_slope=alpha).to(device)
                else:
                    model = ResidualAutoencoder(n, m, l, negative_slope=alpha).to(device)

                n_params = sum(p.numel() for p in model.parameters())

                final_loss = train_model(model, n_steps, sparsity=sparsity)

                metrics = measure_encoding_linearity(model, sparsity=sparsity)

                result = {
                    'architecture': arch,
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

                pbar.set_postfix({'arch': arch, 'l': l, 'gain': f"{metrics['nonlinear_gain']:.5f}"})
                pbar.update(1)

    pbar.close()
    return all_results


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    architectures = ['standard', 'residual']
    depths = sorted(list(set(r['l'] for r in results)))

    # Aggregate by architecture and depth
    arch_depth_stats = {}
    for arch in architectures:
        arch_depth_stats[arch] = {}
        for l in depths:
            subset = [r for r in results if r['architecture'] == arch and r['l'] == l]
            if subset:
                gains = [r['nonlinear_gain'] for r in subset]
                mses = [r['mse_full'] for r in subset]
                linearities = [r['linearity_score'] for r in subset]

                arch_depth_stats[arch][l] = {
                    'mean_gain': np.mean(gains),
                    'std_gain': np.std(gains),
                    'max_gain': np.max(gains),
                    'mean_mse': np.mean(mses),
                    'mean_linearity': np.mean(linearities),
                    'n_steps': subset[0]['n_steps'],
                    'n_params': subset[0]['n_params']
                }

    # Correlations by architecture
    print(f"\n  DEPTH vs NONLINEAR GAIN CORRELATIONS:")
    for arch in architectures:
        subset = [r for r in results if r['architecture'] == arch]
        all_l = [r['l'] for r in subset]
        all_gain = [r['nonlinear_gain'] for r in subset]
        corr = np.corrcoef(all_l, all_gain)[0, 1]
        print(f"    {arch}: r = {corr:.3f}")

    # Results table
    print(f"\n  COMPARISON TABLE:")
    print(f"  {'Arch':>10} | {'l':>2} | {'Steps':>5} | {'Gain (mean±std)':>18} | {'MSE':>8} | {'Lin':>6}")
    print(f"  {'-'*10}-+-{'-'*2}-+-{'-'*5}-+-{'-'*18}-+-{'-'*8}-+-{'-'*6}")

    for l in depths:
        for arch in architectures:
            if l in arch_depth_stats[arch]:
                s = arch_depth_stats[arch][l]
                print(f"  {arch:>10} | {l:>2} | {s['n_steps']:>5} | {s['mean_gain']:.5f} ± {s['std_gain']:.5f} | {s['mean_mse']:.5f} | {s['mean_linearity']:.3f}")

    # Check which depths show improvement with residual
    print(f"\n  RESIDUAL vs STANDARD IMPROVEMENT:")
    print(f"  {'l':>2} | {'Std Gain':>10} | {'Res Gain':>10} | {'Improvement':>12} | {'Winner':>10}")
    print(f"  {'-'*2}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

    residual_wins = 0
    for l in depths:
        std_gain = arch_depth_stats['standard'][l]['mean_gain']
        res_gain = arch_depth_stats['residual'][l]['mean_gain']

        if std_gain > 0:
            improvement = (res_gain - std_gain) / std_gain * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            improvement = res_gain - std_gain
            imp_str = f"{improvement:+.5f}"

        winner = 'residual' if res_gain > std_gain else 'standard'
        if res_gain > std_gain:
            residual_wins += 1

        print(f"  {l:>2} | {std_gain:>10.5f} | {res_gain:>10.5f} | {imp_str:>12} | {winner:>10}")

    print(f"\n  Residual wins {residual_wins}/{len(depths)} depth configurations")

    # Check if residual overcomes the depth ceiling
    print(f"\n  DEPTH CEILING ANALYSIS:")
    for arch in architectures:
        gains_at_depth8plus = [r['nonlinear_gain'] for r in results if r['architecture'] == arch and r['l'] >= 8]
        mean_gain_8plus = np.mean(gains_at_depth8plus)
        positive_count = sum(1 for g in gains_at_depth8plus if g > 0)
        print(f"    {arch}: Mean gain at l≥8 = {mean_gain_8plus:.5f}, positive in {positive_count}/{len(gains_at_depth8plus)} runs")

    # Variance comparison
    print(f"\n  VARIANCE COMPARISON (stability):")
    for arch in architectures:
        all_stds = [arch_depth_stats[arch][l]['std_gain'] for l in depths]
        mean_std = np.mean(all_stds)
        print(f"    {arch}: Mean std across depths = {mean_std:.5f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {'standard': 'blue', 'residual': 'red'}
    markers = {'standard': 'o', 'residual': 's'}

    # Plot 1: Nonlinear gain vs depth (both architectures)
    ax = axes[0, 0]
    for arch in architectures:
        means = [arch_depth_stats[arch][l]['mean_gain'] for l in depths]
        stds = [arch_depth_stats[arch][l]['std_gain'] for l in depths]
        ax.errorbar(depths, means, yerr=stds, marker=markers[arch], capsize=3,
                   linewidth=2, markersize=8, label=arch, color=colors[arch])
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5, label='Ceiling (l=8)')
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth by Architecture')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 2: MSE vs depth (both architectures)
    ax = axes[0, 1]
    for arch in architectures:
        mses = [arch_depth_stats[arch][l]['mean_mse'] for l in depths]
        ax.plot(depths, mses, marker=markers[arch], linewidth=2, markersize=8,
               label=arch, color=colors[arch])
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 3: Improvement ratio (residual / standard)
    ax = axes[1, 0]
    std_gains = [arch_depth_stats['standard'][l]['mean_gain'] for l in depths]
    res_gains = [arch_depth_stats['residual'][l]['mean_gain'] for l in depths]
    ratios = [r - s for r, s in zip(res_gains, std_gains)]  # Difference since some may be negative
    ax.bar(depths, ratios, color='green', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Gain Difference (Residual - Standard)')
    ax.set_title('Residual Advantage by Depth')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 4: Linearity vs depth (both architectures)
    ax = axes[1, 1]
    for arch in architectures:
        linearities = [arch_depth_stats[arch][l]['mean_linearity'] for l in depths]
        ax.plot(depths, linearities, marker=markers[arch], linewidth=2, markersize=8,
               label=arch, color=colors[arch])
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Linearity Score')
    ax.set_title('Encoding Linearity vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/residual_comparison.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/residual_comparison.png")

    # Find best configurations
    best_standard = max([r for r in results if r['architecture'] == 'standard'],
                       key=lambda x: x['nonlinear_gain'])
    best_residual = max([r for r in results if r['architecture'] == 'residual'],
                       key=lambda x: x['nonlinear_gain'])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Best standard: l={best_standard['l']}, gain={best_standard['nonlinear_gain']:.5f}")
    print(f"  Best residual: l={best_residual['l']}, gain={best_residual['nonlinear_gain']:.5f}")
    print(f"  Overall best: {'residual' if best_residual['nonlinear_gain'] > best_standard['nonlinear_gain'] else 'standard'}")

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

    # Compute final statistics
    stats = {
        'arch_depth_stats': arch_depth_stats,
        'best_standard': {'l': best_standard['l'], 'gain': best_standard['nonlinear_gain']},
        'best_residual': {'l': best_residual['l'], 'gain': best_residual['nonlinear_gain']},
        'residual_wins': residual_wins,
        'total_depths': len(depths)
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
