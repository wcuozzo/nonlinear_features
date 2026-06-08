"""
Experiment 12: LeakyReLU Negative Slope Optimization

Key question: Is there an optimal negative slope (alpha) for LeakyReLU that maximizes nonlinear gain?

Prior findings from Experiment 11:
- LeakyReLU achieved highest max nonlinear gain (0.00336) vs ReLU (0.00280)
- LeakyReLU's optimal depth is l=3 (non-monotonic), while ReLU peaks at l=5 (monotonic)
- LeakyReLU allows gradient flow through negative regions
- Default LeakyReLU uses alpha=0.01

Hypothesis:
1. There exists an optimal negative slope (alpha) that maximizes nonlinear gain
2. Very small alpha (~0) should behave like ReLU (lose LeakyReLU's advantage)
3. Very large alpha (~1) should behave like linear (lose all nonlinearity)
4. The optimal alpha may vary with depth

Design:
- Test alpha ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}
- Test l ∈ {2, 3, 4} (around LeakyReLU's optimal depth from Exp 11)
- n=64, CR=16 (same as Exp 11)
- 3 seeds per config
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_12"


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

    model.train()
    return {
        'linearity_score': linearity_score,
        'mse_full': mse_full,
        'mse_linear': mse_linear,
        'nonlinear_gain': nonlinear_gain,
        'effective_dim': effective_dim,
        'effective_dim_ratio': effective_dim_ratio
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
    """Main experiment: test LeakyReLU negative slope effects on nonlinear encoding."""

    n = 64
    compression_ratio = 16
    m = n // compression_ratio  # m = 4

    # Test various negative slopes
    # 0 = ReLU, 1 = linear scaling for negatives
    alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

    # Focus on depths around LeakyReLU's optimal from Exp 11
    depths = [2, 3, 4]

    sparsity = 0.1
    base_steps = 100  # Scale with depth
    n_seeds = 3

    all_results = []

    print("=" * 60)
    print("Experiment 12: LeakyReLU Negative Slope Optimization")
    print("=" * 60)
    print(f"n={n}, m={m} (CR={compression_ratio})")
    print(f"Depths: {depths}")
    print(f"Negative slopes (alpha): {alphas}")
    print(f"Base steps: {base_steps} (scaled by (l+1)/2)")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(depths) * len(alphas) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for alpha in alphas:
        for l in depths:
            # Scale training with depth
            n_steps = int(base_steps * (l + 1) / 2)

            for seed in range(n_seeds):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                model = Autoencoder(n, m, l, negative_slope=alpha).to(device)
                n_params = sum(p.numel() for p in model.parameters())

                final_loss = train_model(model, n_steps, sparsity=sparsity)

                metrics = measure_encoding_linearity(model, sparsity=sparsity)

                result = {
                    'alpha': alpha,
                    'n': n,
                    'm': m,
                    'l': l,
                    'compression_ratio': compression_ratio,
                    'seed': seed,
                    'n_params': n_params,
                    'n_steps': n_steps,
                    'final_loss': final_loss,
                    **metrics
                }
                all_results.append(result)
                pbar.update(1)

    pbar.close()
    return all_results


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    alphas = sorted(list(set(r['alpha'] for r in results)))
    depths = sorted(list(set(r['l'] for r in results)))

    # Compute statistics per alpha
    alpha_stats = {}
    for alpha in alphas:
        subset = [r for r in results if r['alpha'] == alpha]

        # Average over all depths and seeds
        mean_gain = np.mean([r['nonlinear_gain'] for r in subset])
        max_gain = np.max([r['nonlinear_gain'] for r in subset])
        std_gain = np.std([r['nonlinear_gain'] for r in subset])
        mean_linearity = np.mean([r['linearity_score'] for r in subset])

        # Depth-gain correlation for this alpha
        ls = [r['l'] for r in subset]
        gs = [r['nonlinear_gain'] for r in subset]
        depth_corr = np.corrcoef(ls, gs)[0, 1] if len(set(ls)) > 1 else 0

        # Find optimal depth
        depth_gains = {}
        for l in depths:
            l_subset = [r for r in subset if r['l'] == l]
            if l_subset:
                depth_gains[l] = np.mean([r['nonlinear_gain'] for r in l_subset])

        optimal_l = max(depth_gains, key=depth_gains.get)

        alpha_stats[alpha] = {
            'mean_gain': mean_gain,
            'max_gain': max_gain,
            'std_gain': std_gain,
            'mean_linearity': mean_linearity,
            'depth_corr': depth_corr,
            'optimal_l': optimal_l,
            'depth_gains': depth_gains
        }

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Nonlinear gain vs alpha by depth
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
    for i, l in enumerate(depths):
        subset = [r for r in results if r['l'] == l]
        alpha_gains = {}
        alpha_stds = {}
        for alpha in alphas:
            alpha_subset = [r for r in subset if r['alpha'] == alpha]
            if alpha_subset:
                gains = [r['nonlinear_gain'] for r in alpha_subset]
                alpha_gains[alpha] = np.mean(gains)
                alpha_stds[alpha] = np.std(gains)

        as_sorted = sorted(alpha_gains.keys())
        gs = [alpha_gains[a] for a in as_sorted]
        stds = [alpha_stds[a] for a in as_sorted]
        ax.errorbar(as_sorted, gs, yerr=stds, marker='o', capsize=3,
                   label=f'l={l}', color=colors[i])

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Negative Slope (α)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs LeakyReLU Slope by Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Max nonlinear gain by alpha (across all depths)
    ax = axes[0, 1]
    max_gains = [alpha_stats[alpha]['max_gain'] for alpha in alphas]
    mean_gains = [alpha_stats[alpha]['mean_gain'] for alpha in alphas]

    x = np.arange(len(alphas))
    width = 0.35
    ax.bar(x - width/2, max_gains, width, label='Max Gain', alpha=0.7)
    ax.bar(x + width/2, mean_gains, width, label='Mean Gain', alpha=0.7)
    ax.set_xlabel('Negative Slope (α)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Max and Mean Nonlinear Gain by Slope')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.2f}' for a in alphas], rotation=45)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot 3: Linearity score vs alpha
    ax = axes[1, 0]
    mean_linearities = [alpha_stats[alpha]['mean_linearity'] for alpha in alphas]
    ax.plot(alphas, mean_linearities, marker='o', linewidth=2)
    ax.set_xlabel('Negative Slope (α)')
    ax.set_ylabel('Linearity Score')
    ax.set_title('Encoding Linearity vs LeakyReLU Slope')
    ax.grid(True, alpha=0.3)

    # Plot 4: Heatmap of nonlinear gain (alpha vs depth)
    ax = axes[1, 1]
    gain_matrix = np.zeros((len(alphas), len(depths)))
    for i, alpha in enumerate(alphas):
        for j, l in enumerate(depths):
            subset = [r for r in results if r['alpha'] == alpha and r['l'] == l]
            if subset:
                gain_matrix[i, j] = np.mean([r['nonlinear_gain'] for r in subset])

    im = ax.imshow(gain_matrix, aspect='auto', cmap='RdYlGn', origin='lower')
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f'{a:.2f}' for a in alphas])
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Negative Slope (α)')
    ax.set_title('Nonlinear Gain Heatmap')
    plt.colorbar(im, ax=ax, label='Nonlinear Gain')

    # Add annotations
    for i in range(len(alphas)):
        for j in range(len(depths)):
            val = gain_matrix[i, j]
            color = 'white' if abs(val) > 0.001 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/leakyrelu_slope_effects.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/leakyrelu_slope_effects.png")

    # Compute key correlations
    all_alphas_list = [r['alpha'] for r in results]
    all_gains = [r['nonlinear_gain'] for r in results]
    all_linearity = [r['linearity_score'] for r in results]

    alpha_gain_corr = np.corrcoef(all_alphas_list, all_gains)[0, 1]
    alpha_linearity_corr = np.corrcoef(all_alphas_list, all_linearity)[0, 1]

    # Find optimal alpha
    best_alpha = max(alphas, key=lambda a: alpha_stats[a]['max_gain'])
    best_config = max(results, key=lambda r: r['nonlinear_gain'])

    # Print statistics
    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)

    print(f"\n  Alpha vs Nonlinear Gain correlation: r = {alpha_gain_corr:.3f}")
    print(f"  Alpha vs Linearity Score correlation: r = {alpha_linearity_corr:.3f}")
    print(f"\n  Optimal alpha: {best_alpha}")
    print(f"  Best config: alpha={best_config['alpha']}, l={best_config['l']}, gain={best_config['nonlinear_gain']:.5f}")

    # Rank alphas by max gain
    ranked = sorted(alphas, key=lambda a: alpha_stats[a]['max_gain'], reverse=True)

    print("\n  ALPHA RANKING (by max nonlinear gain):")
    for i, alpha in enumerate(ranked, 1):
        stats = alpha_stats[alpha]
        print(f"    {i}. α={alpha:0.2f}: max_gain={stats['max_gain']:.5f}, "
              f"depth_corr={stats['depth_corr']:.3f}, optimal_l={stats['optimal_l']}")

    # Summary table by alpha and depth
    print("\n" + "=" * 60)
    print("DETAILED RESULTS BY ALPHA AND DEPTH")
    print("=" * 60)

    print(f"\n  {'Alpha':>6} | {'l':>2} | {'Gain':>10} | {'MSE':>8} | {'Linearity':>9}")
    print("  " + "-" * 55)

    for alpha in alphas:
        for l in depths:
            subset = [r for r in results if r['alpha'] == alpha and r['l'] == l]
            if subset:
                avg_gain = np.mean([r['nonlinear_gain'] for r in subset])
                avg_mse = np.mean([r['mse_full'] for r in subset])
                avg_lin = np.mean([r['linearity_score'] for r in subset])
                marker = "*" if (alpha == best_config['alpha'] and l == best_config['l']) else " "
                print(f"  {alpha:>6.2f} | {l:>2} | {avg_gain:>+10.5f} | {avg_mse:>8.5f} | {avg_lin:>9.4f} {marker}")

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

    serializable_results = make_serializable(results)
    serializable_stats = make_serializable({
        'alpha_stats': alpha_stats,
        'ranked_alphas': ranked,
        'alpha_gain_corr': alpha_gain_corr,
        'alpha_linearity_corr': alpha_linearity_corr,
        'best_config': best_config,
    })

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump({
            'results': serializable_results,
            'stats': serializable_stats
        }, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR}/results.json")

    return alpha_stats, best_config


if __name__ == "__main__":
    results = run_experiment()
    stats, best_config = analyze_and_plot(results)
