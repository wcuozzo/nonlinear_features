"""
Experiment 18: Does the Depth Ceiling Scale with Input Dimension (n)?

Key question: Is the depth ceiling at l≈7-8 (found for n=128 in Exp 14, 17)
universal or does it depend on input dimension?

Prior findings:
- Exp 14: Depth ceiling at l≈7-8 for n=128, CR=16
- Exp 17: Ceiling invariant to compression ratio (still l≈6-7 for CR=8,16,32 at n=128)
- Exp 10: Optimal depth scales with log2(n), suggesting larger n may benefit from deeper networks
- Exp 5: Larger n achieves higher nonlinear gains when properly trained

Hypothesis:
Larger n may support deeper networks before collapse. Two competing mechanisms:
1. If ceiling is due to vanishing gradients / optimization difficulty → larger n may have
   LOWER ceiling (harder optimization)
2. If ceiling is due to limited expressiveness → larger n may have HIGHER ceiling
   (more capacity to exploit)

We predict the ceiling scales positively with n, based on Exp 10's finding that optimal
depth scales with log2(n).

Design:
- Test n ∈ {64, 128, 256} at depths l ∈ {4, 6, 8, 10}
- Fixed CR=16 (optimal from prior experiments)
- Fixed α=0.1 (best overall from Exp 13)
- Training scaled by n and l
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_18"


class StandardAutoencoder(nn.Module):
    """Standard autoencoder architecture (best from Exp 16)."""
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
    """Main experiment: test if depth ceiling depends on input dimension."""

    # Input dimensions to test
    input_dims = [64, 128, 256]

    # Fixed compression ratio
    cr = 16

    # Activation slope
    alpha = 0.1
    sparsity = 0.1

    # Depths to test: include depths around and beyond the ceiling
    depths = [4, 6, 8, 10]

    # Training scaling: base_steps × (n/64) × (l+1)/2
    # This ensures adequate training for larger models and deeper networks
    base_steps = 60
    n_seeds = 3

    all_results = []

    print("=" * 60)
    print("Experiment 18: Depth Ceiling vs Input Dimension")
    print("=" * 60)
    print(f"Input dimensions (n): {input_dims}")
    print(f"Bottleneck sizes (m=n/CR): {[n//cr for n in input_dims]}")
    print(f"Compression ratio (CR): {cr}")
    print(f"α (LeakyReLU slope): {alpha}")
    print(f"Sparsity: {sparsity}")
    print(f"Depths: {depths}")
    print(f"Base steps: {base_steps} × (n/64) × (l+1)/2")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(input_dims) * len(depths) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for n in input_dims:
        m = n // cr  # Bottleneck size

        for l in depths:
            # Scale training with both n and l
            n_steps = int(base_steps * (n / 64) * (l + 1) / 2)

            for seed in range(n_seeds):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                model = StandardAutoencoder(n, m, l, negative_slope=alpha).to(device)
                n_params = sum(p.numel() for p in model.parameters())

                final_loss = train_model(model, n_steps, sparsity=sparsity)

                metrics = measure_encoding_linearity(model, sparsity=sparsity)

                result = {
                    'n': n,
                    'm': m,
                    'l': l,
                    'compression_ratio': cr,
                    'alpha': alpha,
                    'seed': seed,
                    'n_params': n_params,
                    'n_steps': n_steps,
                    'final_loss': final_loss,
                    **metrics
                }
                all_results.append(result)

                pbar.set_postfix({'n': n, 'l': l, 'gain': f"{metrics['nonlinear_gain']:.5f}"})
                pbar.update(1)

    pbar.close()
    return all_results


def find_ceiling_depth(gains_by_depth, depths, threshold=0.0005):
    """
    Find the depth at which nonlinear gain drops below threshold.
    Returns the ceiling depth (last depth with gain > threshold).
    """
    ceiling = None
    for l in depths:
        if l in gains_by_depth and gains_by_depth[l] > threshold:
            ceiling = l
    return ceiling


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    input_dims = sorted(list(set(r['n'] for r in results)))
    depths = sorted(list(set(r['l'] for r in results)))

    # Aggregate by n and depth
    n_depth_stats = {}
    for n in input_dims:
        n_depth_stats[n] = {}
        for l in depths:
            subset = [r for r in results if r['n'] == n and r['l'] == l]
            if subset:
                gains = [r['nonlinear_gain'] for r in subset]
                mses = [r['mse_full'] for r in subset]
                linearities = [r['linearity_score'] for r in subset]

                n_depth_stats[n][l] = {
                    'mean_gain': np.mean(gains),
                    'std_gain': np.std(gains),
                    'max_gain': np.max(gains),
                    'min_gain': np.min(gains),
                    'mean_mse': np.mean(mses),
                    'mean_linearity': np.mean(linearities),
                    'n_steps': subset[0]['n_steps'],
                    'n_params': subset[0]['n_params'],
                    'm': subset[0]['m'],
                    'positive_count': sum(1 for g in gains if g > 0),
                    'n_runs': len(gains)
                }

    # Correlations by n
    print(f"\n  DEPTH vs NONLINEAR GAIN CORRELATIONS (by n):")
    n_depth_corrs = {}
    for n in input_dims:
        subset = [r for r in results if r['n'] == n]
        all_l = [r['l'] for r in subset]
        all_gain = [r['nonlinear_gain'] for r in subset]
        corr = np.corrcoef(all_l, all_gain)[0, 1]
        n_depth_corrs[n] = corr
        print(f"    n={n}: r = {corr:.3f}")

    # Overall correlations
    print(f"\n  OVERALL CORRELATIONS:")
    all_n = [r['n'] for r in results]
    all_l = [r['l'] for r in results]
    all_gain = [r['nonlinear_gain'] for r in results]
    all_log_n = [np.log2(r['n']) for r in results]

    corr_n_gain = np.corrcoef(all_n, all_gain)[0, 1]
    corr_l_gain = np.corrcoef(all_l, all_gain)[0, 1]
    corr_logn_gain = np.corrcoef(all_log_n, all_gain)[0, 1]

    # n × l interaction
    all_product = [r['l'] * np.log2(r['n']) for r in results]
    corr_product_gain = np.corrcoef(all_product, all_gain)[0, 1]

    print(f"    n vs Gain: r = {corr_n_gain:.3f}")
    print(f"    log2(n) vs Gain: r = {corr_logn_gain:.3f}")
    print(f"    Depth vs Gain: r = {corr_l_gain:.3f}")
    print(f"    l×log2(n) vs Gain: r = {corr_product_gain:.3f}")

    # Results table
    print(f"\n  RESULTS TABLE:")
    print(f"  {'n':>4} | {'m':>3} | {'l':>2} | {'Steps':>5} | {'Gain (mean±std)':>18} | {'MSE':>8} | {'Lin':>6} | {'Pos':>4}")
    print(f"  {'-'*4}-+-{'-'*3}-+-{'-'*2}-+-{'-'*5}-+-{'-'*18}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}")

    for n in input_dims:
        for l in depths:
            if l in n_depth_stats[n]:
                s = n_depth_stats[n][l]
                print(f"  {n:>4} | {s['m']:>3} | {l:>2} | {s['n_steps']:>5} | {s['mean_gain']:.5f} ± {s['std_gain']:.5f} | {s['mean_mse']:.5f} | {s['mean_linearity']:.3f} | {s['positive_count']}/{s['n_runs']}")

    # Depth ceiling analysis by n
    print(f"\n  DEPTH CEILING ANALYSIS:")
    print(f"  (Ceiling = last depth where mean gain > 0.0005)")

    ceiling_by_n = {}
    for n in input_dims:
        gains_by_depth = {l: n_depth_stats[n][l]['mean_gain'] for l in depths}
        ceiling = find_ceiling_depth(gains_by_depth, depths)
        ceiling_by_n[n] = ceiling

        # Find max gain depth
        max_gain_depth = max(depths, key=lambda l: n_depth_stats[n][l]['mean_gain'])
        max_gain = n_depth_stats[n][max_gain_depth]['mean_gain']

        print(f"    n={n}: ceiling ≈ l={ceiling}, max gain at l={max_gain_depth} ({max_gain:.5f})")

    # Test if ceiling correlates with log2(n)
    ceilings = [ceiling_by_n[n] for n in input_dims if ceiling_by_n[n] is not None]
    log_ns = [np.log2(n) for n in input_dims if ceiling_by_n[n] is not None]
    if len(ceilings) >= 2:
        ceiling_corr = np.corrcoef(log_ns, ceilings)[0, 1]
        print(f"\n    log2(n) vs Ceiling Depth correlation: r = {ceiling_corr:.3f}")

    # Fraction of runs with positive gain at l≥8
    print(f"\n  POSITIVE GAIN RATES AT DEEP LAYERS (l≥8):")
    for n in input_dims:
        deep_gains = [r['nonlinear_gain'] for r in results if r['n'] == n and r['l'] >= 8]
        positive_rate = sum(1 for g in deep_gains if g > 0) / len(deep_gains) if deep_gains else 0
        mean_gain = np.mean(deep_gains) if deep_gains else 0
        print(f"    n={n}: {sum(1 for g in deep_gains if g > 0)}/{len(deep_gains)} positive ({positive_rate:.1%}), mean={mean_gain:.5f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {64: 'blue', 128: 'green', 256: 'red'}
    markers = {64: 'o', 128: 's', 256: '^'}

    # Plot 1: Nonlinear gain vs depth (all n values)
    ax = axes[0, 0]
    for n in input_dims:
        means = [n_depth_stats[n][l]['mean_gain'] for l in depths]
        stds = [n_depth_stats[n][l]['std_gain'] for l in depths]
        ax.errorbar(depths, means, yerr=stds, marker=markers[n], capsize=3,
                   linewidth=2, markersize=8, label=f'n={n} (m={n//16})', color=colors[n])
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth by Input Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 2: MSE vs depth (all n values)
    ax = axes[0, 1]
    for n in input_dims:
        mses = [n_depth_stats[n][l]['mean_mse'] for l in depths]
        ax.plot(depths, mses, marker=markers[n], linewidth=2, markersize=8,
               label=f'n={n}', color=colors[n])
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 3: Heatmap of nonlinear gain
    ax = axes[1, 0]
    gain_matrix = np.zeros((len(input_dims), len(depths)))
    for i, n in enumerate(input_dims):
        for j, l in enumerate(depths):
            gain_matrix[i, j] = n_depth_stats[n][l]['mean_gain']

    im = ax.imshow(gain_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.001, vmax=0.01)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_yticks(range(len(input_dims)))
    ax.set_yticklabels([f'n={n}' for n in input_dims])
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Input Dimension')
    ax.set_title('Nonlinear Gain Heatmap')

    # Add text annotations
    for i in range(len(input_dims)):
        for j in range(len(depths)):
            val = gain_matrix[i, j]
            color = 'white' if abs(val) > 0.004 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax)

    # Plot 4: Depth ceiling vs n and depth-gain correlation by n
    ax = axes[1, 1]

    # Plot ceiling vs n
    valid_ns = [n for n in input_dims if ceiling_by_n[n] is not None]
    valid_ceilings = [ceiling_by_n[n] for n in valid_ns]
    ax.bar([f'n={n}' for n in input_dims], [ceiling_by_n.get(n, 0) for n in input_dims],
           color=[colors[n] for n in input_dims], alpha=0.7, label='Ceiling Depth')
    ax.set_xlabel('Input Dimension')
    ax.set_ylabel('Ceiling Depth')
    ax.set_title('Depth Ceiling by Input Dimension')
    ax.grid(True, alpha=0.3)

    # Add ceiling values as text
    for i, n in enumerate(input_dims):
        ceil = ceiling_by_n.get(n, None)
        if ceil is not None:
            ax.text(i, ceil + 0.3, f'l={ceil}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/depth_ceiling_vs_input_dim.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/depth_ceiling_vs_input_dim.png")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for n in input_dims:
        best_depth = max(depths, key=lambda l: n_depth_stats[n][l]['mean_gain'])
        best_gain = n_depth_stats[n][best_depth]['mean_gain']
        print(f"  n={n}: Best depth={best_depth}, max gain={best_gain:.5f}, ceiling≈{ceiling_by_n[n]}")

    # Key finding
    print(f"\n  KEY FINDINGS:")
    print(f"    - Depth-gain correlation varies with n: {dict(n_depth_corrs)}")
    print(f"    - Ceiling depths by n: {ceiling_by_n}")

    # Test ceiling scaling hypothesis
    if len(valid_ceilings) >= 2:
        ceiling_slope = np.polyfit(log_ns, valid_ceilings, 1)[0] if len(valid_ns) >= 2 else None
        if ceiling_slope is not None:
            print(f"    - Ceiling scaling: Δl / Δlog2(n) ≈ {ceiling_slope:.2f}")

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
        'n_depth_stats': n_depth_stats,
        'ceiling_by_n': ceiling_by_n,
        'n_depth_corrs': n_depth_corrs,
        'overall_correlations': {
            'n_gain': corr_n_gain,
            'logn_gain': corr_logn_gain,
            'depth_gain': corr_l_gain,
            'product_gain': corr_product_gain
        }
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
