"""
Experiment 8: Depth × Compression Interaction

Key question: Does optimal depth depend on compression ratio?

Prior findings:
- Exp 2-3: Compression is the primary driver of nonlinear gain (r=0.833)
- Exp 6: Depth correlates strongly with nonlinear gain under scaled training (r=0.94)
- But: These were tested largely independently. Is there an interaction?

Hypotheses:
1. Higher compression may require MORE depth (more computation to achieve compression)
2. OR: Higher compression may need LESS depth (simpler transformation suffices)
3. There may be a "sweet spot" in the (depth, compression) plane

Design:
- Full factorial: l ∈ {1,2,3,4,5} × CR ∈ {4,8,16,32}
- Fixed n=64 for speed
- Scaled training: steps = base × l (deeper needs more training)
- Look for interaction effects in the depth-compression plane
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_8"


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


def measure_latent_stats(model, n_samples=500, sparsity=0.1):
    """Measure latent space statistics."""
    model.eval()

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)
        z = model.encode(x)

        variances = z.var(dim=0).cpu().numpy()
        total_var = variances.sum()
        top1_var = variances.max() / (total_var + 1e-8)

        eff_dim = (total_var ** 2) / ((variances ** 2).sum() + 1e-8)
        eff_dim_ratio = eff_dim / model.m

    model.train()
    return {
        'top1_var': top1_var,
        'eff_dim_ratio': eff_dim_ratio
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
    """Main experiment: full factorial depth × compression."""

    n = 64  # Fixed input dimension for speed
    depths = [1, 2, 3, 4, 5]
    compression_ratios = [4, 8, 16, 32]  # m = n / CR

    sparsity = 0.1
    base_steps = 100  # Scale with depth
    n_seeds = 2

    all_results = []

    print("=" * 60)
    print("Experiment 8: Depth × Compression Interaction")
    print("=" * 60)
    print(f"n = {n}")
    print(f"Depths: {depths}")
    print(f"Compression Ratios: {compression_ratios}")
    print(f"Base steps: {base_steps} (scaled by depth)")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(depths) * len(compression_ratios) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for l in depths:
        for cr in compression_ratios:
            m = n // cr
            # Scale training steps with depth (deeper networks need more training)
            n_steps = base_steps * (l + 1) // 2

            for seed in range(n_seeds):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                model = Autoencoder(n, m, l).to(device)
                n_params = sum(p.numel() for p in model.parameters())

                final_loss = train_model(model, n_steps, sparsity=sparsity)

                metrics = measure_encoding_linearity(model, sparsity=sparsity)
                latent_stats = measure_latent_stats(model, sparsity=sparsity)

                result = {
                    'n': n,
                    'm': m,
                    'l': l,
                    'compression_ratio': cr,
                    'seed': seed,
                    'n_params': n_params,
                    'n_steps': n_steps,
                    'final_loss': final_loss,
                    **metrics,
                    **latent_stats
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

    # Convert to arrays for analysis
    depths = sorted(list(set(r['l'] for r in results)))
    crs = sorted(list(set(r['compression_ratio'] for r in results)))

    # Build heatmap data (average over seeds)
    gain_matrix = np.zeros((len(depths), len(crs)))
    mse_matrix = np.zeros((len(depths), len(crs)))
    linearity_matrix = np.zeros((len(depths), len(crs)))

    for i, l in enumerate(depths):
        for j, cr in enumerate(crs):
            subset = [r for r in results if r['l'] == l and r['compression_ratio'] == cr]
            gain_matrix[i, j] = np.mean([r['nonlinear_gain'] for r in subset])
            mse_matrix[i, j] = np.mean([r['mse_full'] for r in subset])
            linearity_matrix[i, j] = np.mean([r['linearity_score'] for r in subset])

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Nonlinear gain heatmap
    ax = axes[0, 0]
    im = ax.imshow(gain_matrix, aspect='auto', cmap='RdYlBu_r', origin='lower')
    ax.set_xticks(range(len(crs)))
    ax.set_xticklabels(crs)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Depth (l)')
    ax.set_title('Nonlinear Gain (Depth × Compression)')
    plt.colorbar(im, ax=ax)

    # Add values to heatmap
    for i in range(len(depths)):
        for j in range(len(crs)):
            val = gain_matrix[i, j]
            color = 'white' if abs(val) > np.max(np.abs(gain_matrix)) * 0.5 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', color=color, fontsize=8)

    # Plot 2: MSE heatmap
    ax = axes[0, 1]
    im = ax.imshow(mse_matrix, aspect='auto', cmap='viridis_r', origin='lower')
    ax.set_xticks(range(len(crs)))
    ax.set_xticklabels(crs)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Depth (l)')
    ax.set_title('MSE (Depth × Compression)')
    plt.colorbar(im, ax=ax)

    # Plot 3: Linearity heatmap
    ax = axes[0, 2]
    im = ax.imshow(linearity_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(crs)))
    ax.set_xticklabels(crs)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Depth (l)')
    ax.set_title('Linearity Score (Depth × Compression)')
    plt.colorbar(im, ax=ax)

    # Plot 4: Nonlinear gain vs depth (by compression ratio)
    ax = axes[1, 0]
    for cr in crs:
        gains_by_depth = []
        for l in depths:
            subset = [r for r in results if r['l'] == l and r['compression_ratio'] == cr]
            gains_by_depth.append(np.mean([r['nonlinear_gain'] for r in subset]))
        ax.plot(depths, gains_by_depth, marker='o', label=f'CR={cr}')

    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Depth Effect by Compression Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot 5: Nonlinear gain vs compression (by depth)
    ax = axes[1, 1]
    for l in depths:
        gains_by_cr = []
        for cr in crs:
            subset = [r for r in results if r['l'] == l and r['compression_ratio'] == cr]
            gains_by_cr.append(np.mean([r['nonlinear_gain'] for r in subset]))
        ax.plot(crs, gains_by_cr, marker='o', label=f'l={l}')

    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Compression Effect by Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xscale('log', base=2)

    # Plot 6: Optimal depth for each compression ratio
    ax = axes[1, 2]
    optimal_depths = []
    max_gains = []
    for cr in crs:
        best_gain = -float('inf')
        best_depth = None
        for l in depths:
            subset = [r for r in results if r['l'] == l and r['compression_ratio'] == cr]
            avg_gain = np.mean([r['nonlinear_gain'] for r in subset])
            if avg_gain > best_gain:
                best_gain = avg_gain
                best_depth = l
        optimal_depths.append(best_depth)
        max_gains.append(best_gain)

    ax2 = ax.twinx()
    ax.bar(range(len(crs)), optimal_depths, alpha=0.7, color='steelblue', label='Optimal Depth')
    ax2.plot(range(len(crs)), max_gains, 'ro-', label='Max Gain')

    ax.set_xticks(range(len(crs)))
    ax.set_xticklabels(crs)
    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Optimal Depth', color='steelblue')
    ax2.set_ylabel('Max Nonlinear Gain', color='red')
    ax.set_title('Optimal Depth & Max Gain by Compression')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/depth_compression_interaction.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/depth_compression_interaction.png")

    # Compute statistics and interaction effects
    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)

    # Main effects correlations
    gains = [r['nonlinear_gain'] for r in results]
    depth_vals = [r['l'] for r in results]
    cr_vals = [r['compression_ratio'] for r in results]
    log_cr_vals = [np.log2(r['compression_ratio']) for r in results]

    corr_depth_gain = np.corrcoef(depth_vals, gains)[0, 1]
    corr_cr_gain = np.corrcoef(cr_vals, gains)[0, 1]
    corr_logcr_gain = np.corrcoef(log_cr_vals, gains)[0, 1]

    print(f"  Depth vs Nonlinear Gain (r): {corr_depth_gain:.3f}")
    print(f"  Compression Ratio vs Nonlinear Gain (r): {corr_cr_gain:.3f}")
    print(f"  log2(CR) vs Nonlinear Gain (r): {corr_logcr_gain:.3f}")

    # Interaction: how does depth effect vary by compression?
    print("\n  DEPTH EFFECT BY COMPRESSION RATIO:")
    depth_effects_by_cr = {}
    for cr in crs:
        subset = [r for r in results if r['compression_ratio'] == cr]
        d = [r['l'] for r in subset]
        g = [r['nonlinear_gain'] for r in subset]
        if len(d) >= 4:
            corr = np.corrcoef(d, g)[0, 1]
            depth_effects_by_cr[cr] = corr
            print(f"    CR={cr}: depth-gain correlation = {corr:.3f}")

    # Interaction: how does compression effect vary by depth?
    print("\n  COMPRESSION EFFECT BY DEPTH:")
    cr_effects_by_depth = {}
    for l in depths:
        subset = [r for r in results if r['l'] == l]
        c = [np.log2(r['compression_ratio']) for r in subset]
        g = [r['nonlinear_gain'] for r in subset]
        if len(c) >= 4:
            corr = np.corrcoef(c, g)[0, 1]
            cr_effects_by_depth[l] = corr
            print(f"    l={l}: logCR-gain correlation = {corr:.3f}")

    # Best configuration
    best_result = max(results, key=lambda r: r['nonlinear_gain'])
    print(f"\n  BEST CONFIGURATION:")
    print(f"    l={best_result['l']}, CR={best_result['compression_ratio']}, m={best_result['m']}")
    print(f"    Nonlinear Gain: {best_result['nonlinear_gain']:.5f}")
    print(f"    MSE: {best_result['mse_full']:.5f}")

    # Check for interaction (does depth effect increase with compression?)
    if len(depth_effects_by_cr) >= 3:
        cr_list = list(depth_effects_by_cr.keys())
        effect_list = list(depth_effects_by_cr.values())
        interaction_corr = np.corrcoef(cr_list, effect_list)[0, 1]
        print(f"\n  INTERACTION (CR vs depth effect strength): {interaction_corr:.3f}")
        if interaction_corr > 0.3:
            print("    → Higher compression STRENGTHENS depth benefit")
        elif interaction_corr < -0.3:
            print("    → Higher compression WEAKENS depth benefit")
        else:
            print("    → Weak interaction (depth and compression effects largely independent)")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (averaged over seeds)")
    print("=" * 60)
    print(f"{'l':>3} | {'CR':>3} | {'m':>3} | {'Gain':>9} | {'MSE':>7} | {'Linearity':>9}")
    print("-" * 50)

    for l in depths:
        for cr in crs:
            subset = [r for r in results if r['l'] == l and r['compression_ratio'] == cr]
            m = subset[0]['m']
            avg_gain = np.mean([r['nonlinear_gain'] for r in subset])
            avg_mse = np.mean([r['mse_full'] for r in subset])
            avg_lin = np.mean([r['linearity_score'] for r in subset])
            print(f"{l:>3} | {cr:>3} | {m:>3} | {avg_gain:>9.5f} | {avg_mse:>7.5f} | {avg_lin:>9.4f}")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    stats = {
        'corr_depth_gain': corr_depth_gain,
        'corr_cr_gain': corr_cr_gain,
        'corr_logcr_gain': corr_logcr_gain,
        'depth_effects_by_cr': depth_effects_by_cr,
        'cr_effects_by_depth': cr_effects_by_depth,
        'best_config': {
            'l': best_result['l'],
            'cr': best_result['compression_ratio'],
            'm': best_result['m'],
            'nonlinear_gain': best_result['nonlinear_gain']
        },
        'optimal_depths_by_cr': dict(zip(crs, optimal_depths)),
        'max_gains_by_cr': dict(zip(crs, max_gains))
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
