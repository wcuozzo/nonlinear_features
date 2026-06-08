"""
Experiment 9: Phase Boundary Formula Verification Across Input Dimensions

Key question: Does the phase boundary formula l × log2(CR) ≈ 10 hold across different n?

Prior findings (Exp 8):
- Depth and compression are SUBSTITUTES (interaction r=-0.925)
- Phase boundary approximated as l × log2(CR) ≈ 10 at n=64
- Best config at n=64 was l=4, CR=32 (giving 4 × 5 = 20)

Hypothesis:
- The phase boundary formula should hold across input dimensions
- OR: The formula may scale with n (larger n needs different critical product)

Design:
- Test n ∈ {32, 64, 128}
- For each n, test the region around the hypothesized phase boundary
- Use scaled training: steps ∝ n × l (both dimension and depth scale training)
- Check where nonlinear gain crosses from negative to positive
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_9"


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

    model.train()
    return {
        'linearity_score': linearity_score,
        'mse_full': mse_full,
        'mse_linear': mse_linear,
        'nonlinear_gain': nonlinear_gain
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
    """Main experiment: test phase boundary across input dimensions."""

    # Test across input dimensions
    input_dims = [32, 64, 128]

    # Depths and compression ratios to test
    depths = [1, 2, 3, 4]
    compression_ratios = [4, 8, 16, 32]

    sparsity = 0.1
    base_steps = 50  # Keep short for speed, scale with n and l
    n_seeds = 2

    all_results = []

    print("=" * 60)
    print("Experiment 9: Phase Boundary Formula Verification")
    print("=" * 60)
    print(f"Input dimensions: {input_dims}")
    print(f"Depths: {depths}")
    print(f"Compression Ratios: {compression_ratios}")
    print(f"Base steps: {base_steps} (scaled by n/32 × (l+1)/2)")
    print(f"Seeds: {n_seeds}")
    print(f"Testing phase boundary hypothesis: l × log2(CR) ≈ 10")
    print("=" * 60)

    total_configs = len(input_dims) * len(depths) * len(compression_ratios) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for n in input_dims:
        for l in depths:
            for cr in compression_ratios:
                m = n // cr
                if m < 1:  # Skip invalid configs
                    pbar.update(n_seeds)
                    continue

                # Scale training with BOTH n and l
                n_steps = int(base_steps * (n / 32) * (l + 1) / 2)

                # Compute the phase boundary product
                boundary_product = l * np.log2(cr)

                for seed in range(n_seeds):
                    torch.manual_seed(42 + seed)
                    np.random.seed(42 + seed)

                    model = Autoencoder(n, m, l).to(device)
                    n_params = sum(p.numel() for p in model.parameters())

                    final_loss = train_model(model, n_steps, sparsity=sparsity)

                    metrics = measure_encoding_linearity(model, sparsity=sparsity)

                    result = {
                        'n': n,
                        'm': m,
                        'l': l,
                        'compression_ratio': cr,
                        'log2_cr': np.log2(cr),
                        'boundary_product': boundary_product,
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

    # Group by input dimension
    input_dims = sorted(list(set(r['n'] for r in results)))
    depths = sorted(list(set(r['l'] for r in results)))
    crs = sorted(list(set(r['compression_ratio'] for r in results)))

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1-3: Heatmaps for each n
    for idx, n in enumerate(input_dims):
        ax = axes[0, idx]

        subset_results = [r for r in results if r['n'] == n]

        gain_matrix = np.zeros((len(depths), len(crs)))
        for i, l in enumerate(depths):
            for j, cr in enumerate(crs):
                subset = [r for r in subset_results if r['l'] == l and r['compression_ratio'] == cr]
                if subset:
                    gain_matrix[i, j] = np.mean([r['nonlinear_gain'] for r in subset])
                else:
                    gain_matrix[i, j] = np.nan

        im = ax.imshow(gain_matrix, aspect='auto', cmap='RdYlBu_r', origin='lower',
                       vmin=-0.005, vmax=0.015)
        ax.set_xticks(range(len(crs)))
        ax.set_xticklabels(crs)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels(depths)
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Depth (l)')
        ax.set_title(f'Nonlinear Gain (n={n})')
        plt.colorbar(im, ax=ax)

        # Add values
        for i in range(len(depths)):
            for j in range(len(crs)):
                val = gain_matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.005 else 'black'
                    ax.text(j, i, f'{val:.4f}', ha='center', va='center', color=color, fontsize=7)

    # Plot 4: Nonlinear gain vs boundary product (l × log2(CR))
    ax = axes[1, 0]
    for n in input_dims:
        subset = [r for r in results if r['n'] == n]
        # Average over seeds
        bp_gain = {}
        for r in subset:
            bp = r['boundary_product']
            if bp not in bp_gain:
                bp_gain[bp] = []
            bp_gain[bp].append(r['nonlinear_gain'])

        bps = sorted(bp_gain.keys())
        gains = [np.mean(bp_gain[bp]) for bp in bps]
        ax.plot(bps, gains, marker='o', label=f'n={n}')

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=10, color='r', linestyle='--', alpha=0.7, label='Hypothesized boundary (10)')
    ax.set_xlabel('l × log2(CR)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Phase Boundary: Nonlinear Gain vs l × log2(CR)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Critical product by n (where gain crosses zero)
    ax = axes[1, 1]
    critical_products = []
    for n in input_dims:
        subset = [r for r in results if r['n'] == n]
        bp_gain = {}
        for r in subset:
            bp = r['boundary_product']
            if bp not in bp_gain:
                bp_gain[bp] = []
            bp_gain[bp].append(r['nonlinear_gain'])

        # Find critical product (interpolate where gain crosses zero)
        bps = sorted(bp_gain.keys())
        gains = [np.mean(bp_gain[bp]) for bp in bps]

        # Find first positive crossing
        critical = None
        for i in range(len(bps) - 1):
            if gains[i] < 0 and gains[i + 1] >= 0:
                # Linear interpolation
                critical = bps[i] + (bps[i+1] - bps[i]) * (-gains[i]) / (gains[i+1] - gains[i])
                break

        if critical is None and gains[-1] > 0:
            # All positive or crossing not found - estimate conservatively
            for i in range(len(bps)):
                if gains[i] >= 0:
                    critical = bps[i]
                    break

        critical_products.append((n, critical if critical else bps[-1]))

    ns = [cp[0] for cp in critical_products]
    crits = [cp[1] for cp in critical_products]
    ax.bar(range(len(ns)), crits, tick_label=[str(n) for n in ns], color='steelblue', alpha=0.7)
    ax.axhline(y=10, color='r', linestyle='--', alpha=0.7, label='Hypothesized boundary (10)')
    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Critical l × log2(CR)')
    ax.set_title('Critical Phase Boundary Product by n')
    ax.legend()

    # Store for stats
    critical_dict = dict(critical_products)

    # Plot 6: Correlation between boundary product and gain, by n
    ax = axes[1, 2]
    correlations = []
    for n in input_dims:
        subset = [r for r in results if r['n'] == n]
        bp = [r['boundary_product'] for r in subset]
        gain = [r['nonlinear_gain'] for r in subset]
        corr = np.corrcoef(bp, gain)[0, 1]
        correlations.append(corr)

    ax.bar(range(len(input_dims)), correlations, tick_label=[str(n) for n in input_dims],
           color='steelblue', alpha=0.7)
    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Correlation (l×log2(CR) vs Gain)')
    ax.set_title('Predictive Power of Phase Boundary Formula')
    ax.set_ylim([0, 1])
    for i, corr in enumerate(correlations):
        ax.text(i, corr + 0.02, f'{corr:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/phase_boundary_scaling.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/phase_boundary_scaling.png")

    # Compute statistics
    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)

    # Overall correlation
    all_bp = [r['boundary_product'] for r in results]
    all_gain = [r['nonlinear_gain'] for r in results]
    overall_corr = np.corrcoef(all_bp, all_gain)[0, 1]
    print(f"  Overall: l×log2(CR) vs Nonlinear Gain (r): {overall_corr:.3f}")

    # By n
    print("\n  BY INPUT DIMENSION:")
    for i, n in enumerate(input_dims):
        print(f"    n={n}: correlation = {correlations[i]:.3f}, critical product = {critical_dict.get(n, 'N/A'):.2f}")

    # Test if critical product scales with n
    print("\n  CRITICAL PRODUCT SCALING:")
    ns = list(critical_dict.keys())
    crits = list(critical_dict.values())
    if len(ns) >= 3:
        log_ns = [np.log2(n) for n in ns]
        # Check if critical product is constant (formula holds) or scales
        crit_std = np.std(crits)
        crit_mean = np.mean(crits)
        crit_cv = crit_std / crit_mean if crit_mean > 0 else float('inf')
        print(f"    Mean critical product: {crit_mean:.2f}")
        print(f"    Std critical product: {crit_std:.2f}")
        print(f"    Coefficient of variation: {crit_cv:.3f}")

        if crit_cv < 0.2:
            print("    → Critical product is STABLE across n (formula l×log2(CR)≈const HOLDS)")
        else:
            print("    → Critical product VARIES with n (formula needs n-dependent adjustment)")
            # Check correlation with log(n)
            corr_n_crit = np.corrcoef(log_ns, crits)[0, 1]
            print(f"    → Correlation of critical product with log2(n): {corr_n_crit:.3f}")

    # Best configuration per n
    print("\n  BEST CONFIGURATIONS:")
    for n in input_dims:
        subset = [r for r in results if r['n'] == n]
        if subset:
            best = max(subset, key=lambda r: r['nonlinear_gain'])
            print(f"    n={n}: l={best['l']}, CR={best['compression_ratio']}, m={best['m']}, "
                  f"l×log2(CR)={best['boundary_product']:.1f}, gain={best['nonlinear_gain']:.5f}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (averaged over seeds)")
    print("=" * 60)

    for n in input_dims:
        print(f"\n  n = {n}:")
        print(f"  {'l':>3} | {'CR':>3} | {'m':>3} | {'l×log2CR':>8} | {'Gain':>9}")
        print("  " + "-" * 45)

        for l in depths:
            for cr in crs:
                subset = [r for r in results if r['n'] == n and r['l'] == l and r['compression_ratio'] == cr]
                if subset:
                    m = subset[0]['m']
                    bp = subset[0]['boundary_product']
                    avg_gain = np.mean([r['nonlinear_gain'] for r in subset])
                    marker = "+" if avg_gain > 0 else " "
                    print(f"  {l:>3} | {cr:>3} | {m:>3} | {bp:>8.1f} | {avg_gain:>+9.5f} {marker}")

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
        'overall_correlation': overall_corr,
        'correlations_by_n': dict(zip([str(n) for n in input_dims], correlations)),
        'critical_products_by_n': {str(k): v for k, v in critical_dict.items()},
        'critical_product_mean': float(crit_mean),
        'critical_product_std': float(crit_std),
        'critical_product_cv': float(crit_cv)
    }

    serializable_results = make_serializable(results)
    serializable_stats = make_serializable(stats)

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump({
            'results': serializable_results,
            'stats': serializable_stats
        }, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR}/results.json")

    return stats


if __name__ == "__main__":
    results = run_experiment()
    stats = analyze_and_plot(results)
