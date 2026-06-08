"""
Experiment 10: Optimal Depth Scaling with Input Dimension

Key question: Does optimal depth scale with input dimension n?

Prior findings:
- Exp 6: Depth correlation is r=0.94 with proper training; deeper = better at n=128
- Exp 8: Depth and compression are SUBSTITUTES (r=-0.925 interaction)
- Exp 9: Phase boundary critical product scales inversely with n (r=-0.999)
       Larger n requires LOWER l×log2(CR) for nonlinearity

Open question:
- If larger n makes nonlinearity "easier" to achieve, does this mean:
  A) Larger n needs LESS depth (can rely more on compression alone)?
  B) Larger n benefits from MORE depth (more capacity to exploit)?
  C) Optimal depth is invariant to n?

Hypothesis:
- Optimal depth may INCREASE with n because:
  1. More features = more potential interactions to learn
  2. Larger models have more capacity = can utilize deeper computation
  BUT this is speculative - the substitution effect may dominate

Design:
- Test n ∈ {32, 64, 128, 256}
- For each n, test l ∈ {1, 2, 3, 4, 5, 6}
- Use fixed moderate compression CR=16 (to isolate depth effect)
- Scale training with both n and l: steps ∝ (n/32) × (l+1)/2
- Find optimal depth at each n
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_10"


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
    """Main experiment: test optimal depth scaling with n."""

    # Input dimensions
    input_dims = [32, 64, 128, 256]

    # Depths to test
    depths = [1, 2, 3, 4, 5, 6]

    # Fixed compression ratio to isolate depth effect
    compression_ratio = 16

    sparsity = 0.1
    base_steps = 50  # Scale with n and l
    n_seeds = 2

    all_results = []

    print("=" * 60)
    print("Experiment 10: Optimal Depth Scaling with Input Dimension")
    print("=" * 60)
    print(f"Input dimensions: {input_dims}")
    print(f"Depths: {depths}")
    print(f"Compression Ratio: {compression_ratio} (fixed)")
    print(f"Base steps: {base_steps} (scaled by n/32 × (l+1)/2)")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(input_dims) * len(depths) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for n in input_dims:
        m = n // compression_ratio
        if m < 1:
            pbar.update(len(depths) * n_seeds)
            continue

        for l in depths:
            # Scale training with both n and l
            n_steps = int(base_steps * (n / 32) * (l + 1) / 2)

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

    input_dims = sorted(list(set(r['n'] for r in results)))
    depths = sorted(list(set(r['l'] for r in results)))

    # Find optimal depth for each n
    optimal_depths = {}
    optimal_gains = {}

    for n in input_dims:
        subset = [r for r in results if r['n'] == n]
        # Average over seeds
        depth_gains = {}
        for l in depths:
            l_subset = [r for r in subset if r['l'] == l]
            if l_subset:
                depth_gains[l] = np.mean([r['nonlinear_gain'] for r in l_subset])

        if depth_gains:
            optimal_l = max(depth_gains, key=depth_gains.get)
            optimal_depths[n] = optimal_l
            optimal_gains[n] = depth_gains[optimal_l]

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Nonlinear gain vs depth for each n
    ax = axes[0, 0]
    for n in input_dims:
        subset = [r for r in results if r['n'] == n]
        depth_gains = {}
        depth_stds = {}
        for l in depths:
            l_subset = [r for r in subset if r['l'] == l]
            if l_subset:
                gains = [r['nonlinear_gain'] for r in l_subset]
                depth_gains[l] = np.mean(gains)
                depth_stds[l] = np.std(gains)

        ds = sorted(depth_gains.keys())
        gs = [depth_gains[d] for d in ds]
        stds = [depth_stds[d] for d in ds]
        ax.errorbar(ds, gs, yerr=stds, marker='o', capsize=3, label=f'n={n}')

        # Mark optimal
        opt_l = optimal_depths.get(n)
        if opt_l and opt_l in depth_gains:
            ax.scatter([opt_l], [depth_gains[opt_l]], s=150, marker='*', zorder=5)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth by Input Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Optimal depth vs n
    ax = axes[0, 1]
    ns = sorted(optimal_depths.keys())
    opt_ds = [optimal_depths[n] for n in ns]
    ax.bar(range(len(ns)), opt_ds, tick_label=[str(n) for n in ns], color='steelblue', alpha=0.7)
    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Optimal Depth')
    ax.set_title('Optimal Depth vs Input Dimension')

    # Add trend line
    if len(ns) >= 3:
        log_ns = np.array([np.log2(n) for n in ns])
        opt_ds_arr = np.array(opt_ds)
        z = np.polyfit(log_ns, opt_ds_arr, 1)
        p = np.poly1d(z)
        ax.plot(range(len(ns)), p(log_ns), 'r--', alpha=0.7, label=f'Trend: {z[0]:.2f}×log2(n)+{z[1]:.2f}')
        ax.legend()

    # Plot 3: Depth-gain correlation by n
    ax = axes[1, 0]
    depth_corrs = []
    for n in input_dims:
        subset = [r for r in results if r['n'] == n]
        ls = [r['l'] for r in subset]
        gs = [r['nonlinear_gain'] for r in subset]
        if len(set(ls)) > 1:
            corr = np.corrcoef(ls, gs)[0, 1]
        else:
            corr = 0
        depth_corrs.append(corr)

    ax.bar(range(len(input_dims)), depth_corrs, tick_label=[str(n) for n in input_dims],
           color='steelblue', alpha=0.7)
    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Depth-Gain Correlation')
    ax.set_title('Depth Effect Strength by Input Dimension')
    ax.set_ylim([0, 1])
    for i, corr in enumerate(depth_corrs):
        ax.text(i, corr + 0.02, f'{corr:.3f}', ha='center', fontsize=10)

    # Plot 4: Max nonlinear gain vs n
    ax = axes[1, 1]
    max_gains = [optimal_gains.get(n, 0) for n in input_dims]
    ax.bar(range(len(input_dims)), max_gains, tick_label=[str(n) for n in input_dims],
           color='steelblue', alpha=0.7)
    ax.set_xlabel('Input Dimension (n)')
    ax.set_ylabel('Max Nonlinear Gain (at optimal depth)')
    ax.set_title('Maximum Nonlinear Gain vs Input Dimension')
    for i, g in enumerate(max_gains):
        ax.text(i, g + 0.001, f'{g:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/optimal_depth_scaling.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/optimal_depth_scaling.png")

    # Compute statistics
    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)

    # Overall depth-gain correlation
    all_ls = [r['l'] for r in results]
    all_gs = [r['nonlinear_gain'] for r in results]
    overall_depth_corr = np.corrcoef(all_ls, all_gs)[0, 1]
    print(f"  Overall depth vs gain correlation: {overall_depth_corr:.3f}")

    # n vs optimal depth correlation
    ns_arr = np.array(list(optimal_depths.keys()))
    opt_arr = np.array(list(optimal_depths.values()))
    if len(ns_arr) >= 3:
        n_optd_corr = np.corrcoef(np.log2(ns_arr), opt_arr)[0, 1]
        print(f"  log2(n) vs optimal depth correlation: {n_optd_corr:.3f}")

        # Linear fit: optimal_depth = a * log2(n) + b
        z = np.polyfit(np.log2(ns_arr), opt_arr, 1)
        print(f"  Scaling law fit: optimal_l = {z[0]:.2f} × log2(n) + {z[1]:.2f}")

    print("\n  OPTIMAL DEPTHS BY INPUT DIMENSION:")
    for n in sorted(optimal_depths.keys()):
        print(f"    n={n}: optimal l={optimal_depths[n]}, max gain={optimal_gains.get(n, 0):.5f}")

    # n vs max gain correlation
    if len(input_dims) >= 3:
        max_g_arr = np.array([optimal_gains.get(n, 0) for n in input_dims])
        n_maxg_corr = np.corrcoef(np.log2(input_dims), max_g_arr)[0, 1]
        print(f"\n  log2(n) vs max gain correlation: {n_maxg_corr:.3f}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (averaged over seeds)")
    print("=" * 60)

    print(f"\n  {'n':>4} | {'l':>2} | {'m':>3} | {'steps':>5} | {'Gain':>9} | {'MSE':>7}")
    print("  " + "-" * 50)

    for n in input_dims:
        for l in depths:
            subset = [r for r in results if r['n'] == n and r['l'] == l]
            if subset:
                m = subset[0]['m']
                steps = subset[0]['n_steps']
                avg_gain = np.mean([r['nonlinear_gain'] for r in subset])
                avg_mse = np.mean([r['mse_full'] for r in subset])
                marker = "*" if l == optimal_depths.get(n) else " "
                print(f"  {n:>4} | {l:>2} | {m:>3} | {steps:>5} | {avg_gain:>+9.5f} | {avg_mse:>7.5f} {marker}")

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
        'overall_depth_gain_corr': overall_depth_corr,
        'depth_gain_corr_by_n': dict(zip([str(n) for n in input_dims], depth_corrs)),
        'optimal_depths': {str(k): v for k, v in optimal_depths.items()},
        'optimal_gains': {str(k): v for k, v in optimal_gains.items()},
    }

    if len(ns_arr) >= 3:
        stats['log2n_vs_optimal_depth_corr'] = float(n_optd_corr)
        stats['scaling_fit_slope'] = float(z[0])
        stats['scaling_fit_intercept'] = float(z[1])

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
