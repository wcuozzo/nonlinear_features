"""
Experiment 11: Activation Function Effects on Nonlinear Encoding

Key question: How do different activation functions affect the nonlinearity phase boundary?

Prior findings:
- All prior experiments used ReLU exclusively
- Exp 8: Phase boundary follows l × log2(CR) ≈ 10 (for n=64)
- Exp 9: Phase boundary shifts with n: l × log2(CR) + 2×log2(n) ≈ 20
- Exp 10: Deeper networks benefit larger n (r=0.992 for n-gain scaling)

Hypothesis:
Different activations should affect both:
1. The LOCATION of the phase boundary (easier or harder to achieve nonlinearity)
2. The MAGNITUDE of nonlinear gain (more or less expressive)

Key activations to test:
- ReLU: baseline, sharp nonlinearity, sparse gradients
- LeakyReLU: allows negative gradients, may enable smoother learning
- GELU: smooth, used in modern transformers, approximates stochastic ReLU
- Tanh: bounded, symmetric, classical choice
- SiLU/Swish: self-gated, smooth, popular in recent architectures

Prediction:
- Smooth activations (GELU, SiLU) may achieve higher nonlinear gain because they
  allow gradient flow through the entire range
- Sharp activations (ReLU) may require more depth to achieve same nonlinearity

Design:
- Test n=64, CR=16 (moderate settings from prior work)
- Test l ∈ {1, 2, 3, 4, 5} to see depth scaling per activation
- Compare 5 activation functions
- Scaled training (proportional to depth)
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_11"


class Autoencoder(nn.Module):
    def __init__(self, n: int, m: int, l: int, activation_fn):
        super().__init__()
        self.n = n
        self.m = m
        self.l = l
        self.activation_name = activation_fn.__class__.__name__

        encoder_layers = []
        for i in range(l):
            encoder_layers.append(nn.Linear(n, n))
            encoder_layers.append(activation_fn())
        encoder_layers.append(nn.Linear(n, m))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        decoder_layers.append(nn.Linear(m, n))
        decoder_layers.append(activation_fn())
        for i in range(l - 1):
            decoder_layers.append(nn.Linear(n, n))
            decoder_layers.append(activation_fn())
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


# Define activation functions to test
ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'GELU': nn.GELU,
    'Tanh': nn.Tanh,
    'SiLU': nn.SiLU,  # Also known as Swish
}


def run_experiment():
    """Main experiment: test activation function effects on nonlinear encoding."""

    n = 64
    compression_ratio = 16
    m = n // compression_ratio  # m = 4

    depths = [1, 2, 3, 4, 5]
    sparsity = 0.1
    base_steps = 100  # Scale with depth
    n_seeds = 3

    all_results = []

    print("=" * 60)
    print("Experiment 11: Activation Function Effects on Nonlinear Encoding")
    print("=" * 60)
    print(f"n={n}, m={m} (CR={compression_ratio})")
    print(f"Depths: {depths}")
    print(f"Activations: {list(ACTIVATIONS.keys())}")
    print(f"Base steps: {base_steps} (scaled by (l+1)/2)")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(depths) * len(ACTIVATIONS) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for act_name, act_fn in ACTIVATIONS.items():
        for l in depths:
            # Scale training with depth
            n_steps = int(base_steps * (l + 1) / 2)

            for seed in range(n_seeds):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                model = Autoencoder(n, m, l, act_fn).to(device)
                n_params = sum(p.numel() for p in model.parameters())

                final_loss = train_model(model, n_steps, sparsity=sparsity)

                metrics = measure_encoding_linearity(model, sparsity=sparsity)

                result = {
                    'activation': act_name,
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

    activations = list(ACTIVATIONS.keys())
    depths = sorted(list(set(r['l'] for r in results)))

    # Compute statistics per activation
    activation_stats = {}
    for act in activations:
        subset = [r for r in results if r['activation'] == act]

        # Average over all depths and seeds
        mean_gain = np.mean([r['nonlinear_gain'] for r in subset])
        max_gain = np.max([r['nonlinear_gain'] for r in subset])

        # Depth-gain correlation for this activation
        ls = [r['l'] for r in subset]
        gs = [r['nonlinear_gain'] for r in subset]
        depth_corr = np.corrcoef(ls, gs)[0, 1]

        # Find optimal depth
        depth_gains = {}
        for l in depths:
            l_subset = [r for r in subset if r['l'] == l]
            if l_subset:
                depth_gains[l] = np.mean([r['nonlinear_gain'] for r in l_subset])

        optimal_l = max(depth_gains, key=depth_gains.get)

        # Critical depth (where gain crosses zero)
        critical_l = None
        for l in depths:
            if depth_gains.get(l, -1) > 0:
                critical_l = l
                break

        activation_stats[act] = {
            'mean_gain': mean_gain,
            'max_gain': max_gain,
            'depth_corr': depth_corr,
            'optimal_l': optimal_l,
            'critical_l': critical_l,
            'depth_gains': depth_gains
        }

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Nonlinear gain vs depth by activation
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(activations)))
    for i, act in enumerate(activations):
        subset = [r for r in results if r['activation'] == act]
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
        ax.errorbar(ds, gs, yerr=stds, marker='o', capsize=3,
                   label=act, color=colors[i])

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth by Activation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Max nonlinear gain by activation
    ax = axes[0, 1]
    max_gains = [activation_stats[act]['max_gain'] for act in activations]
    bars = ax.bar(range(len(activations)), max_gains, tick_label=activations,
                  color=[colors[i] for i in range(len(activations))], alpha=0.7)
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Max Nonlinear Gain')
    ax.set_title('Maximum Nonlinear Gain by Activation')
    ax.tick_params(axis='x', rotation=30)
    for i, g in enumerate(max_gains):
        ax.text(i, g + 0.001, f'{g:.4f}', ha='center', fontsize=9)

    # Plot 3: Depth-gain correlation by activation
    ax = axes[1, 0]
    depth_corrs = [activation_stats[act]['depth_corr'] for act in activations]
    ax.bar(range(len(activations)), depth_corrs, tick_label=activations,
           color=[colors[i] for i in range(len(activations))], alpha=0.7)
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Depth-Gain Correlation')
    ax.set_title('Depth Effect Strength by Activation')
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', rotation=30)
    for i, corr in enumerate(depth_corrs):
        ax.text(i, corr + 0.02, f'{corr:.3f}', ha='center', fontsize=9)

    # Plot 4: Critical depth (where gain becomes positive)
    ax = axes[1, 1]
    critical_depths = [activation_stats[act]['critical_l'] if activation_stats[act]['critical_l'] else 6
                       for act in activations]
    optimal_depths = [activation_stats[act]['optimal_l'] for act in activations]

    x = np.arange(len(activations))
    width = 0.35
    ax.bar(x - width/2, critical_depths, width, label='Critical depth (gain > 0)', alpha=0.7)
    ax.bar(x + width/2, optimal_depths, width, label='Optimal depth', alpha=0.7)
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Depth')
    ax.set_title('Critical and Optimal Depth by Activation')
    ax.set_xticks(x)
    ax.set_xticklabels(activations, rotation=30)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/activation_effects.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/activation_effects.png")

    # Print statistics
    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)

    # Rank activations by max gain
    ranked = sorted(activations, key=lambda a: activation_stats[a]['max_gain'], reverse=True)

    print("\n  ACTIVATION RANKING (by max nonlinear gain):")
    for i, act in enumerate(ranked, 1):
        stats = activation_stats[act]
        critical_str = f"l={stats['critical_l']}" if stats['critical_l'] else "never"
        print(f"    {i}. {act:12s}: max_gain={stats['max_gain']:.5f}, "
              f"depth_corr={stats['depth_corr']:.3f}, "
              f"optimal_l={stats['optimal_l']}, "
              f"critical: {critical_str}")

    # Compute correlations
    # Mean gain vs linearity score
    all_gains = [r['nonlinear_gain'] for r in results]
    all_linearity = [r['linearity_score'] for r in results]
    gain_linearity_corr = np.corrcoef(all_gains, all_linearity)[0, 1]
    print(f"\n  Nonlinear gain vs linearity score: r={gain_linearity_corr:.3f}")

    # Summary table
    print("\n" + "=" * 60)
    print("DETAILED RESULTS BY ACTIVATION AND DEPTH")
    print("=" * 60)

    print(f"\n  {'Activation':>12} | {'l':>2} | {'Gain':>10} | {'MSE':>8} | {'Linearity':>9}")
    print("  " + "-" * 60)

    for act in activations:
        for l in depths:
            subset = [r for r in results if r['activation'] == act and r['l'] == l]
            if subset:
                avg_gain = np.mean([r['nonlinear_gain'] for r in subset])
                avg_mse = np.mean([r['mse_full'] for r in subset])
                avg_lin = np.mean([r['linearity_score'] for r in subset])
                marker = "*" if l == activation_stats[act]['optimal_l'] else " "
                print(f"  {act:>12} | {l:>2} | {avg_gain:>+10.5f} | {avg_mse:>8.5f} | {avg_lin:>9.4f} {marker}")

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
        'activation_stats': activation_stats,
        'ranked_activations': ranked,
        'gain_linearity_corr': gain_linearity_corr,
    })

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump({
            'results': serializable_results,
            'stats': serializable_stats
        }, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR}/results.json")

    return activation_stats


if __name__ == "__main__":
    results = run_experiment()
    stats = analyze_and_plot(results)
