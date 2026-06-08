"""
Experiment 19: Do Different Optimizers Break the Depth Ceiling?

Key question: Is the depth ceiling at l≈6-7 an Adam-specific optimization problem?

Prior findings:
- Exp 14-18: Depth ceiling at l≈6-7 is UNIVERSAL, invariant to:
  - Compression ratio (Exp 17)
  - Input dimension (Exp 18)
  - Normalization (Exp 16) - helps slightly but at cost of peak performance
  - Residual connections (Exp 15) - actively hurt nonlinear encoding
- The ceiling appears to be an optimization failure, not an architecture limitation
  (MSE increases at l≥8, linearity score approaches 1.0)

Hypothesis:
The Adam optimizer may be failing at high depth due to:
1. Vanishing/exploding gradients - SGD+momentum might be more stable
2. Adaptive learning rate collapse - AdamW with weight decay might help
3. Second-moment estimation issues - RAdam might help

We test: Adam, AdamW, SGD+momentum, and a learning rate schedule (Adam+warmup).
If ANY optimizer achieves positive nonlinear gain at l≥8, the ceiling is optimizer-specific.

Design:
- Fixed n=128, CR=16 (m=8), α=0.1, sparsity=0.1
- Depths l ∈ {4, 6, 8, 10}
- Optimizers: Adam, AdamW (wd=0.01), SGD+momentum (0.9), Adam+warmup
- Steps scaled by depth: 80×(l+1)
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_19"


class StandardAutoencoder(nn.Module):
    """Standard autoencoder architecture."""
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


def create_optimizer(model, opt_type, lr=1e-3):
    """Create optimizer based on type."""
    if opt_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif opt_type == 'sgd_momentum':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_type == 'adam_warmup':
        # Will use scheduler separately
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def train_model(model, n_steps, opt_type, batch_size=256, sparsity=0.1, lr=1e-3):
    """Train model for n_steps with specified optimizer."""
    optimizer = create_optimizer(model, opt_type, lr)

    # Warmup scheduler for adam_warmup
    scheduler = None
    if opt_type == 'adam_warmup':
        warmup_steps = min(100, n_steps // 4)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    losses = []

    for step in range(n_steps):
        x = generate_sparse_data(batch_size, model.n, sparsity)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    return np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)


def run_experiment():
    """Main experiment: test if different optimizers break the depth ceiling."""

    # Fixed configuration
    n = 128
    cr = 16
    m = n // cr  # = 8
    alpha = 0.1
    sparsity = 0.1

    # Optimizers to test
    optimizers = ['adam', 'adamw', 'sgd_momentum', 'adam_warmup']

    # Depths: include depths around and beyond the ceiling
    depths = [4, 6, 8, 10]

    # Training scaling: base_steps × (l+1)
    base_steps = 80
    n_seeds = 3

    all_results = []

    print("=" * 60)
    print("Experiment 19: Optimizer Effects on Depth Ceiling")
    print("=" * 60)
    print(f"Input dimension (n): {n}")
    print(f"Bottleneck size (m): {m}")
    print(f"Compression ratio (CR): {cr}")
    print(f"α (LeakyReLU slope): {alpha}")
    print(f"Sparsity: {sparsity}")
    print(f"Depths: {depths}")
    print(f"Optimizers: {optimizers}")
    print(f"Base steps: {base_steps} × (l+1)")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    total_configs = len(optimizers) * len(depths) * n_seeds
    pbar = tqdm(total=total_configs, desc="Running experiments")

    for opt_type in optimizers:
        for l in depths:
            # Scale training with depth
            n_steps = int(base_steps * (l + 1))

            for seed in range(n_seeds):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                model = StandardAutoencoder(n, m, l, negative_slope=alpha).to(device)
                n_params = sum(p.numel() for p in model.parameters())

                final_loss = train_model(model, n_steps, opt_type, sparsity=sparsity)

                metrics = measure_encoding_linearity(model, sparsity=sparsity)

                result = {
                    'n': n,
                    'm': m,
                    'l': l,
                    'compression_ratio': cr,
                    'alpha': alpha,
                    'optimizer': opt_type,
                    'seed': seed,
                    'n_params': n_params,
                    'n_steps': n_steps,
                    'final_loss': final_loss,
                    **metrics
                }
                all_results.append(result)

                pbar.set_postfix({'opt': opt_type, 'l': l, 'gain': f"{metrics['nonlinear_gain']:.5f}"})
                pbar.update(1)

    pbar.close()
    return all_results


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    optimizers = sorted(list(set(r['optimizer'] for r in results)))
    depths = sorted(list(set(r['l'] for r in results)))

    # Aggregate by optimizer and depth
    opt_depth_stats = {}
    for opt in optimizers:
        opt_depth_stats[opt] = {}
        for l in depths:
            subset = [r for r in results if r['optimizer'] == opt and r['l'] == l]
            if subset:
                gains = [r['nonlinear_gain'] for r in subset]
                mses = [r['mse_full'] for r in subset]
                linearities = [r['linearity_score'] for r in subset]

                opt_depth_stats[opt][l] = {
                    'mean_gain': np.mean(gains),
                    'std_gain': np.std(gains),
                    'max_gain': np.max(gains),
                    'min_gain': np.min(gains),
                    'mean_mse': np.mean(mses),
                    'mean_linearity': np.mean(linearities),
                    'n_steps': subset[0]['n_steps'],
                    'positive_count': sum(1 for g in gains if g > 0),
                    'n_runs': len(gains)
                }

    # Correlations by optimizer
    print(f"\n  DEPTH vs NONLINEAR GAIN CORRELATIONS (by optimizer):")
    opt_depth_corrs = {}
    for opt in optimizers:
        subset = [r for r in results if r['optimizer'] == opt]
        all_l = [r['l'] for r in subset]
        all_gain = [r['nonlinear_gain'] for r in subset]
        corr = np.corrcoef(all_l, all_gain)[0, 1]
        opt_depth_corrs[opt] = corr
        print(f"    {opt}: r = {corr:.3f}")

    # Results table
    print(f"\n  RESULTS TABLE:")
    print(f"  {'Optimizer':>12} | {'l':>2} | {'Steps':>5} | {'Gain (mean±std)':>18} | {'MSE':>8} | {'Lin':>6} | {'Pos':>4}")
    print(f"  {'-'*12}-+-{'-'*2}-+-{'-'*5}-+-{'-'*18}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}")

    for opt in optimizers:
        for l in depths:
            if l in opt_depth_stats[opt]:
                s = opt_depth_stats[opt][l]
                print(f"  {opt:>12} | {l:>2} | {s['n_steps']:>5} | {s['mean_gain']:.5f} ± {s['std_gain']:.5f} | {s['mean_mse']:.5f} | {s['mean_linearity']:.3f} | {s['positive_count']}/{s['n_runs']}")

    # Key comparison: positive gain at l≥8
    print(f"\n  POSITIVE GAIN RATES AT DEEP LAYERS (l≥8):")
    for opt in optimizers:
        deep_gains = [r['nonlinear_gain'] for r in results if r['optimizer'] == opt and r['l'] >= 8]
        positive_count = sum(1 for g in deep_gains if g > 0)
        total = len(deep_gains)
        positive_rate = positive_count / total if total > 0 else 0
        mean_gain = np.mean(deep_gains) if deep_gains else 0
        print(f"    {opt:>12}: {positive_count}/{total} positive ({positive_rate:.1%}), mean={mean_gain:.5f}")

    # Best optimizer at each depth
    print(f"\n  BEST OPTIMIZER BY DEPTH:")
    for l in depths:
        best_opt = max(optimizers, key=lambda o: opt_depth_stats[o][l]['mean_gain'])
        best_gain = opt_depth_stats[best_opt][l]['mean_gain']
        print(f"    l={l}: {best_opt} (gain={best_gain:.5f})")

    # Did any optimizer break the ceiling?
    print(f"\n  CEILING BREAKTHROUGH TEST:")
    any_breakthrough = False
    for opt in optimizers:
        for l in [8, 10]:
            if l in opt_depth_stats[opt]:
                mean_gain = opt_depth_stats[opt][l]['mean_gain']
                if mean_gain > 0.0005:  # Meaningful positive gain
                    print(f"    *** {opt} achieved meaningful gain at l={l}: {mean_gain:.5f} ***")
                    any_breakthrough = True
    if not any_breakthrough:
        print(f"    No optimizer broke the depth ceiling (all gains at l≥8 < 0.0005)")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'adam': 'blue', 'adamw': 'orange', 'sgd_momentum': 'green', 'adam_warmup': 'red'}
    markers = {'adam': 'o', 'adamw': 's', 'sgd_momentum': '^', 'adam_warmup': 'D'}

    # Plot 1: Nonlinear gain vs depth (all optimizers)
    ax = axes[0, 0]
    for opt in optimizers:
        means = [opt_depth_stats[opt][l]['mean_gain'] for l in depths]
        stds = [opt_depth_stats[opt][l]['std_gain'] for l in depths]
        ax.errorbar(depths, means, yerr=stds, marker=markers[opt], capsize=3,
                   linewidth=2, markersize=8, label=opt, color=colors[opt])
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=7, color='gray', linestyle=':', alpha=0.5, label='Ceiling (l≈7)')
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Depth by Optimizer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 2: MSE vs depth (all optimizers)
    ax = axes[0, 1]
    for opt in optimizers:
        mses = [opt_depth_stats[opt][l]['mean_mse'] for l in depths]
        ax.plot(depths, mses, marker=markers[opt], linewidth=2, markersize=8,
               label=opt, color=colors[opt])
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)

    # Plot 3: Heatmap of nonlinear gain
    ax = axes[1, 0]
    gain_matrix = np.zeros((len(optimizers), len(depths)))
    for i, opt in enumerate(optimizers):
        for j, l in enumerate(depths):
            gain_matrix[i, j] = opt_depth_stats[opt][l]['mean_gain']

    im = ax.imshow(gain_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.002, vmax=0.01)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_yticks(range(len(optimizers)))
    ax.set_yticklabels(optimizers)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Optimizer')
    ax.set_title('Nonlinear Gain Heatmap')

    # Add text annotations
    for i in range(len(optimizers)):
        for j in range(len(depths)):
            val = gain_matrix[i, j]
            color = 'white' if abs(val) > 0.003 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax)

    # Plot 4: Comparison bar chart at l=8 (the ceiling)
    ax = axes[1, 1]
    l_test = 8
    gains_at_8 = [opt_depth_stats[opt][l_test]['mean_gain'] for opt in optimizers]
    stds_at_8 = [opt_depth_stats[opt][l_test]['std_gain'] for opt in optimizers]

    bars = ax.bar(optimizers, gains_at_8, yerr=stds_at_8, capsize=5,
                  color=[colors[opt] for opt in optimizers], alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=0.0005, color='green', linestyle=':', alpha=0.5, label='Breakthrough threshold')
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Nonlinear Gain at l=8')
    ax.set_title(f'Nonlinear Gain at Depth l={l_test} (Beyond Ceiling)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, gains_at_8):
        ypos = val + 0.0002 if val >= 0 else val - 0.0005
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.5f}',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/optimizer_depth_ceiling.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/optimizer_depth_ceiling.png")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for opt in optimizers:
        best_depth = max(depths, key=lambda l: opt_depth_stats[opt][l]['mean_gain'])
        best_gain = opt_depth_stats[opt][best_depth]['mean_gain']
        gain_at_8 = opt_depth_stats[opt][8]['mean_gain']
        print(f"  {opt:>12}: Best l={best_depth} (gain={best_gain:.5f}), l=8 gain={gain_at_8:.5f}")

    print(f"\n  KEY FINDINGS:")
    print(f"    - Depth-gain correlations by optimizer: {dict(opt_depth_corrs)}")

    # Overall winner
    best_opt_overall = max(optimizers, key=lambda o: max(opt_depth_stats[o][l]['mean_gain'] for l in depths))
    best_gain_overall = max(opt_depth_stats[best_opt_overall][l]['mean_gain'] for l in depths)
    print(f"    - Best optimizer overall: {best_opt_overall} (max gain={best_gain_overall:.5f})")

    # Winner at deep layers
    best_opt_deep = max(optimizers, key=lambda o: opt_depth_stats[o][8]['mean_gain'])
    best_gain_deep = opt_depth_stats[best_opt_deep][8]['mean_gain']
    print(f"    - Best optimizer at l=8: {best_opt_deep} (gain={best_gain_deep:.5f})")

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
        'opt_depth_stats': opt_depth_stats,
        'opt_depth_corrs': opt_depth_corrs,
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
