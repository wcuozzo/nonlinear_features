"""
Experiment 13: Combined Optimal Configuration Test

Key question: Do the optimal findings from prior experiments combine synergistically?

Prior findings to combine:
- Exp 12: Optimal α=0.2 for LeakyReLU (44% improvement over default)
- Exp 10: Larger n benefits from more depth (optimal_l scales with log2(n))
- Exp 8: Depth × compression are substitutes (negative interaction r=-0.925)
- Exp 5: Larger n yields higher nonlinear gain with scaled training

Hypothesis:
1. Combining α=0.2 with the best depth/compression at each n should produce
   the highest nonlinear gains observed so far
2. The α=0.2 advantage should persist or increase at larger n
3. There may be an α × depth interaction (optimal α may shift with depth)

Design:
- Test n ∈ {64, 128, 256}
- Test α ∈ {0.0 (ReLU), 0.1, 0.2, 0.3} (around the optimal from Exp 12)
- Test l ∈ {3, 4, 5} (around optimal depths from Exp 10)
- Fixed CR=16 (high compression, known to work well)
- Scaled training: steps ∝ n × (l+1)/2
- 2 seeds per config (for speed)
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_13"


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
    """Main experiment: test combined optimal configuration."""

    # Test parameters
    ns = [64, 128, 256]
    alphas = [0.0, 0.1, 0.2, 0.3]  # Include ReLU (0.0) as baseline
    depths = [3, 4, 5]
    compression_ratio = 16

    sparsity = 0.1
    base_steps = 50  # Will be scaled by n and depth
    n_seeds = 2

    all_results = []

    print("=" * 60)
    print("Experiment 13: Combined Optimal Configuration Test")
    print("=" * 60)
    print(f"Input dimensions: {ns}")
    print(f"Negative slopes (alpha): {alphas}")
    print(f"Depths: {depths}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"Base steps: {base_steps} (scaled by n/32 × (l+1)/2)")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(ns) * len(alphas) * len(depths) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for n in ns:
        m = n // compression_ratio

        for alpha in alphas:
            for l in depths:
                # Scale training with both n and depth (per Exp 5 & 6 findings)
                n_steps = int(base_steps * (n / 32) * (l + 1) / 2)

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

    ns = sorted(list(set(r['n'] for r in results)))
    alphas = sorted(list(set(r['alpha'] for r in results)))
    depths = sorted(list(set(r['l'] for r in results)))

    # Overall correlations
    all_alpha = [r['alpha'] for r in results]
    all_n = [r['n'] for r in results]
    all_l = [r['l'] for r in results]
    all_gain = [r['nonlinear_gain'] for r in results]
    all_log_n = [np.log2(r['n']) for r in results]

    alpha_gain_corr = np.corrcoef(all_alpha, all_gain)[0, 1]
    n_gain_corr = np.corrcoef(all_log_n, all_gain)[0, 1]
    depth_gain_corr = np.corrcoef(all_l, all_gain)[0, 1]

    print(f"\n  OVERALL CORRELATIONS:")
    print(f"    Alpha vs Nonlinear Gain: r = {alpha_gain_corr:.3f}")
    print(f"    log2(n) vs Nonlinear Gain: r = {n_gain_corr:.3f}")
    print(f"    Depth vs Nonlinear Gain: r = {depth_gain_corr:.3f}")

    # Best config per n
    print(f"\n  BEST CONFIG PER INPUT DIMENSION:")
    best_per_n = {}
    for n in ns:
        subset = [r for r in results if r['n'] == n]
        best = max(subset, key=lambda x: x['nonlinear_gain'])
        best_per_n[n] = best
        print(f"    n={n}: α={best['alpha']}, l={best['l']}, gain={best['nonlinear_gain']:.5f}, MSE={best['mse_full']:.5f}")

    # Alpha effect by n
    print(f"\n  ALPHA EFFECT BY INPUT DIMENSION:")
    alpha_effect_by_n = {}
    for n in ns:
        subset = [r for r in results if r['n'] == n]
        alpha_gains = {}
        for alpha in alphas:
            a_subset = [r for r in subset if r['alpha'] == alpha]
            if a_subset:
                alpha_gains[alpha] = np.mean([r['nonlinear_gain'] for r in a_subset])

        best_alpha = max(alpha_gains, key=alpha_gains.get)
        relu_gain = alpha_gains.get(0.0, 0)
        best_gain = alpha_gains[best_alpha]
        improvement = (best_gain - relu_gain) / (abs(relu_gain) + 1e-8) * 100

        alpha_effect_by_n[n] = {
            'best_alpha': best_alpha,
            'best_gain': best_gain,
            'relu_gain': relu_gain,
            'improvement': improvement,
            'alpha_gains': alpha_gains
        }
        print(f"    n={n}: best α={best_alpha}, gain={best_gain:.5f} ({improvement:+.1f}% vs ReLU)")

    # Depth effect by n
    print(f"\n  DEPTH EFFECT BY INPUT DIMENSION:")
    for n in ns:
        subset = [r for r in results if r['n'] == n]
        ls = [r['l'] for r in subset]
        gs = [r['nonlinear_gain'] for r in subset]
        corr = np.corrcoef(ls, gs)[0, 1]
        print(f"    n={n}: depth-gain correlation r = {corr:.3f}")

    # Alpha × depth interaction
    print(f"\n  ALPHA × DEPTH INTERACTION:")
    for n in ns:
        subset = [r for r in results if r['n'] == n]
        # Check if optimal alpha varies with depth
        for l in depths:
            l_subset = [r for r in subset if r['l'] == l]
            alpha_gains = {}
            for alpha in alphas:
                a_subset = [r for r in l_subset if r['alpha'] == alpha]
                if a_subset:
                    alpha_gains[alpha] = np.mean([r['nonlinear_gain'] for r in a_subset])
            best_alpha = max(alpha_gains, key=alpha_gains.get)
            print(f"    n={n}, l={l}: best α={best_alpha}, gain={alpha_gains[best_alpha]:.5f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1-3: Nonlinear gain vs alpha by depth, one plot per n
    for idx, n in enumerate(ns):
        ax = axes[0, idx]
        subset = [r for r in results if r['n'] == n]
        colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))

        for i, l in enumerate(depths):
            l_subset = [r for r in subset if r['l'] == l]
            alpha_gains = {}
            alpha_stds = {}
            for alpha in alphas:
                a_subset = [r for r in l_subset if r['alpha'] == alpha]
                if a_subset:
                    gains = [r['nonlinear_gain'] for r in a_subset]
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
        ax.set_title(f'n={n}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Best nonlinear gain vs n (comparing ReLU vs optimal α)
    ax = axes[1, 0]
    relu_gains = []
    optimal_gains = []
    for n in ns:
        relu_gains.append(alpha_effect_by_n[n]['relu_gain'])
        optimal_gains.append(alpha_effect_by_n[n]['best_gain'])

    x = np.arange(len(ns))
    width = 0.35
    ax.bar(x - width/2, relu_gains, width, label='ReLU (α=0)', alpha=0.7)
    ax.bar(x + width/2, optimal_gains, width, label='Optimal α', alpha=0.7)
    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Best Nonlinear Gain')
    ax.set_title('ReLU vs Optimal LeakyReLU by n')
    ax.set_xticks(x)
    ax.set_xticklabels(ns)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot 5: Scaling of nonlinear gain with n
    ax = axes[1, 1]
    best_gains = [best_per_n[n]['nonlinear_gain'] for n in ns]
    ax.plot(ns, best_gains, 'o-', linewidth=2, markersize=10)
    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Best Nonlinear Gain')
    ax.set_title('Best Nonlinear Gain vs n')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Add scaling fit
    log_ns = np.log2(ns)
    log_gains = np.log(np.array(best_gains) + 1e-6)
    valid = ~np.isinf(log_gains)
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(log_ns[valid], log_gains[valid], 1)
        ax.text(0.05, 0.95, f'Scaling: gain ∝ n^{slope:.2f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    # Plot 6: Heatmap of optimal alpha by (n, l)
    ax = axes[1, 2]
    opt_alpha_matrix = np.zeros((len(ns), len(depths)))
    gain_matrix = np.zeros((len(ns), len(depths)))

    for i, n in enumerate(ns):
        for j, l in enumerate(depths):
            subset = [r for r in results if r['n'] == n and r['l'] == l]
            if subset:
                best = max(subset, key=lambda x: x['nonlinear_gain'])
                opt_alpha_matrix[i, j] = best['alpha']
                gain_matrix[i, j] = best['nonlinear_gain']

    im = ax.imshow(gain_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Input Dimension (n)')
    ax.set_title('Best Nonlinear Gain (α annotated)')
    plt.colorbar(im, ax=ax, label='Nonlinear Gain')

    # Annotate with optimal alpha
    for i in range(len(ns)):
        for j in range(len(depths)):
            alpha = opt_alpha_matrix[i, j]
            gain = gain_matrix[i, j]
            ax.text(j, i, f'α={alpha:.1f}\n{gain:.4f}', ha='center', va='center',
                   fontsize=8, color='white' if gain > 0.005 else 'black')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/combined_optimal_config.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/combined_optimal_config.png")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Compute improvement of optimal over ReLU
    all_improvements = []
    for n in ns:
        improvement = alpha_effect_by_n[n]['improvement']
        all_improvements.append(improvement)

    mean_improvement = np.mean(all_improvements)

    # Overall best config
    overall_best = max(results, key=lambda x: x['nonlinear_gain'])

    print(f"\n  Overall best config: n={overall_best['n']}, α={overall_best['alpha']}, l={overall_best['l']}")
    print(f"  Best nonlinear gain achieved: {overall_best['nonlinear_gain']:.5f}")
    print(f"  Mean improvement of optimal α over ReLU: {mean_improvement:.1f}%")

    # Does optimal alpha scale with n?
    opt_alphas = [alpha_effect_by_n[n]['best_alpha'] for n in ns]
    if len(set(opt_alphas)) > 1:
        corr = np.corrcoef(ns, opt_alphas)[0, 1]
        print(f"  Correlation of optimal α with n: r = {corr:.3f}")
    else:
        print(f"  Optimal α is consistent: {opt_alphas[0]} across all n")

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
        'alpha_gain_corr': alpha_gain_corr,
        'n_gain_corr': n_gain_corr,
        'depth_gain_corr': depth_gain_corr,
        'best_per_n': best_per_n,
        'alpha_effect_by_n': alpha_effect_by_n,
        'overall_best': overall_best,
        'mean_improvement_over_relu': mean_improvement,
    })

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump({
            'results': serializable_results,
            'stats': serializable_stats
        }, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR}/results.json")

    return serializable_stats


if __name__ == "__main__":
    results = run_experiment()
    stats = analyze_and_plot(results)
