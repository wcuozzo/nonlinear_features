"""
Experiment 20: Progressive Depth Training to Break the Depth Ceiling

Key question: Can we train autoencoders past l=7-8 by progressively growing depth?

Prior findings:
- Exp 14-19: Depth ceiling at l≈6-7 is UNIVERSAL
  - Invariant to compression ratio (Exp 17)
  - Invariant to input dimension (Exp 18)
  - Not broken by different optimizers (Exp 19)
  - Not helped by residual connections (Exp 15, actually hurt)
  - LayerNorm helps marginally at deep layers but at cost of peak performance (Exp 16)
- The ceiling appears to be an initialization/early-training problem, not a capacity issue

Hypothesis:
Progressive depth training may break the ceiling by:
1. Starting with a well-trained shallow network (which DOES achieve nonlinear encoding)
2. Gradually adding layers one at a time, initializing new layers to approximate identity
3. The well-trained encoder provides a good initialization for deeper networks

This is inspired by "Net2Net" and progressive GAN training approaches.

Design:
- n=128, m=8 (CR=16), α=0.1, sparsity=0.1
- Two conditions:
  A) Progressive: Train l=2 fully → add layer → train → add layer → ... up to l=10
  B) Standard: Train l=4,6,8,10 from scratch (baseline)
- Steps per depth: 200 per layer (so l=10 total gets ~1600 steps after all additions)
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
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_20"


class GrowableAutoencoder(nn.Module):
    """Autoencoder that can grow deeper progressively."""
    def __init__(self, n: int, m: int, initial_depth: int, negative_slope: float):
        super().__init__()
        self.n = n
        self.m = m
        self.negative_slope = negative_slope

        # Start with initial_depth
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # Build initial encoder: n -> [n]*initial_depth -> m
        for i in range(initial_depth):
            self.encoder_layers.append(nn.Linear(n, n))
        self.encoder_final = nn.Linear(n, m)

        # Build initial decoder: m -> [n]*initial_depth -> n
        self.decoder_first = nn.Linear(m, n)
        for i in range(initial_depth - 1):
            self.decoder_layers.append(nn.Linear(n, n))
        if initial_depth > 0:
            self.decoder_final = nn.Linear(n, n)
        else:
            self.decoder_final = None

        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self._current_depth = initial_depth

    @property
    def current_depth(self):
        return self._current_depth

    def add_layer(self):
        """Add one layer to both encoder and decoder."""
        # Add new encoder layer (initialized to small random weights)
        new_enc_layer = nn.Linear(self.n, self.n).to(device)
        # Initialize close to identity for smooth transition
        nn.init.eye_(new_enc_layer.weight)
        new_enc_layer.weight.data *= 0.1
        new_enc_layer.weight.data += torch.eye(self.n, device=device) * 0.9
        nn.init.zeros_(new_enc_layer.bias)
        self.encoder_layers.append(new_enc_layer)

        # Add new decoder layer
        new_dec_layer = nn.Linear(self.n, self.n).to(device)
        nn.init.eye_(new_dec_layer.weight)
        new_dec_layer.weight.data *= 0.1
        new_dec_layer.weight.data += torch.eye(self.n, device=device) * 0.9
        nn.init.zeros_(new_dec_layer.bias)
        self.decoder_layers.append(new_dec_layer)

        self._current_depth += 1

    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        return self.encoder_final(x)

    def decode(self, z):
        x = self.activation(self.decoder_first(z))
        for layer in self.decoder_layers:
            x = self.activation(layer(x))
        if self.decoder_final is not None:
            x = self.decoder_final(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class StandardAutoencoder(nn.Module):
    """Standard autoencoder for baseline comparison."""
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


def train_model_steps(model, optimizer, n_steps, batch_size=256, sparsity=0.1):
    """Train model for n_steps."""
    losses = []

    for step in range(n_steps):
        x = generate_sparse_data(batch_size, model.n, sparsity)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)


def run_progressive_training(n, m, alpha, sparsity, initial_depth, final_depth,
                             steps_per_layer, lr, seed):
    """Run progressive depth training."""
    torch.manual_seed(42 + seed)
    np.random.seed(42 + seed)

    model = GrowableAutoencoder(n, m, initial_depth, negative_slope=alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    depth_metrics = []
    total_steps = 0

    # Train initial depth
    final_loss = train_model_steps(model, optimizer, steps_per_layer, sparsity=sparsity)
    total_steps += steps_per_layer
    metrics = measure_encoding_linearity(model, sparsity=sparsity)
    depth_metrics.append({
        'depth': model.current_depth,
        'total_steps': total_steps,
        **metrics
    })

    # Progressively add layers
    while model.current_depth < final_depth:
        model.add_layer()
        # Reset optimizer to include new parameters
        optimizer = optim.Adam(model.parameters(), lr=lr)
        final_loss = train_model_steps(model, optimizer, steps_per_layer, sparsity=sparsity)
        total_steps += steps_per_layer
        metrics = measure_encoding_linearity(model, sparsity=sparsity)
        depth_metrics.append({
            'depth': model.current_depth,
            'total_steps': total_steps,
            **metrics
        })

    return depth_metrics, model


def run_standard_training(n, m, l, alpha, sparsity, n_steps, lr, seed):
    """Run standard training from scratch."""
    torch.manual_seed(42 + seed)
    np.random.seed(42 + seed)

    model = StandardAutoencoder(n, m, l, negative_slope=alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    final_loss = train_model_steps(model, optimizer, n_steps, sparsity=sparsity)
    metrics = measure_encoding_linearity(model, sparsity=sparsity)

    return {
        'depth': l,
        'total_steps': n_steps,
        **metrics
    }, model


def run_experiment():
    """Main experiment: progressive vs standard depth training."""

    # Fixed configuration
    n = 128
    cr = 16
    m = n // cr  # = 8
    alpha = 0.1
    sparsity = 0.1
    lr = 1e-3

    # Progressive training config
    initial_depth = 2
    final_depth = 10  # Go well past the ceiling
    steps_per_layer = 150  # Steps to train each new layer

    # Standard training config
    standard_depths = [2, 4, 6, 8, 10]

    n_seeds = 3

    all_results = {
        'progressive': [],
        'standard': []
    }

    print("=" * 60)
    print("Experiment 20: Progressive Depth Training")
    print("=" * 60)
    print(f"Input dimension (n): {n}")
    print(f"Bottleneck size (m): {m}")
    print(f"Compression ratio (CR): {cr}")
    print(f"α (LeakyReLU slope): {alpha}")
    print(f"Sparsity: {sparsity}")
    print(f"Learning rate: {lr}")
    print(f"\nProgressive: l={initial_depth} → l={final_depth}, {steps_per_layer} steps/layer")
    print(f"Standard: depths {standard_depths}, scaled steps")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    # Run progressive training
    print("\n--- Progressive Training ---")
    for seed in tqdm(range(n_seeds), desc="Progressive seeds"):
        depth_metrics, _ = run_progressive_training(
            n, m, alpha, sparsity, initial_depth, final_depth,
            steps_per_layer, lr, seed
        )
        all_results['progressive'].append({
            'seed': seed,
            'depth_progression': depth_metrics
        })

    # Run standard training
    print("\n--- Standard Training ---")
    pbar = tqdm(total=len(standard_depths) * n_seeds, desc="Standard training")
    for l in standard_depths:
        # Match total steps to progressive at this depth
        # Progressive spends steps_per_layer * (l - initial_depth + 1) getting to depth l
        # For fair comparison, use similar total steps
        total_steps_at_l = steps_per_layer * (l - initial_depth + 1)

        for seed in range(n_seeds):
            metrics, _ = run_standard_training(
                n, m, l, alpha, sparsity, total_steps_at_l, lr, seed
            )
            all_results['standard'].append({
                'seed': seed,
                **metrics
            })
            pbar.update(1)
    pbar.close()

    return all_results


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Aggregate progressive results
    progressive_by_depth = {}
    for run in results['progressive']:
        for dm in run['depth_progression']:
            d = dm['depth']
            if d not in progressive_by_depth:
                progressive_by_depth[d] = []
            progressive_by_depth[d].append(dm)

    # Aggregate standard results
    standard_by_depth = {}
    for run in results['standard']:
        d = run['depth']
        if d not in standard_by_depth:
            standard_by_depth[d] = []
        standard_by_depth[d].append(run)

    prog_depths = sorted(progressive_by_depth.keys())
    std_depths = sorted(standard_by_depth.keys())

    # Print comparison table
    print("\n  PROGRESSIVE vs STANDARD COMPARISON:")
    print(f"  {'Depth':>5} | {'Prog Gain (mean±std)':>20} | {'Std Gain (mean±std)':>20} | {'Winner':>10}")
    print(f"  {'-'*5}-+-{'-'*20}-+-{'-'*20}-+-{'-'*10}")

    comparison_data = []
    for d in prog_depths:
        prog_gains = [m['nonlinear_gain'] for m in progressive_by_depth[d]]
        prog_mean = np.mean(prog_gains)
        prog_std = np.std(prog_gains)

        if d in standard_by_depth:
            std_gains = [m['nonlinear_gain'] for m in standard_by_depth[d]]
            std_mean = np.mean(std_gains)
            std_std = np.std(std_gains)
            winner = 'prog' if prog_mean > std_mean else 'std'
            print(f"  {d:>5} | {prog_mean:.5f} ± {std_std:.5f} | {std_mean:.5f} ± {std_std:.5f} | {winner:>10}")
            comparison_data.append({
                'depth': d,
                'prog_mean': prog_mean,
                'prog_std': prog_std,
                'std_mean': std_mean,
                'std_std': std_std,
                'winner': winner
            })
        else:
            print(f"  {d:>5} | {prog_mean:.5f} ± {prog_std:.5f} | {'N/A':>20} | {'prog':>10}")
            comparison_data.append({
                'depth': d,
                'prog_mean': prog_mean,
                'prog_std': prog_std,
                'std_mean': None,
                'std_std': None,
                'winner': 'prog'
            })

    # Key metric: positive gain at l≥8
    print("\n  POSITIVE GAIN AT DEEP LAYERS (l≥8):")

    prog_deep_gains = []
    for d in prog_depths:
        if d >= 8:
            prog_deep_gains.extend([m['nonlinear_gain'] for m in progressive_by_depth[d]])

    std_deep_gains = []
    for d in std_depths:
        if d >= 8:
            std_deep_gains.extend([m['nonlinear_gain'] for m in standard_by_depth[d]])

    prog_positive = sum(1 for g in prog_deep_gains if g > 0) if prog_deep_gains else 0
    std_positive = sum(1 for g in std_deep_gains if g > 0) if std_deep_gains else 0

    print(f"    Progressive: {prog_positive}/{len(prog_deep_gains)} positive, mean={np.mean(prog_deep_gains):.5f}")
    print(f"    Standard: {std_positive}/{len(std_deep_gains)} positive, mean={np.mean(std_deep_gains):.5f}")

    # Did progressive break the ceiling?
    print("\n  CEILING BREAKTHROUGH TEST:")
    breakthrough = False
    for d in [8, 9, 10]:
        if d in progressive_by_depth:
            gains = [m['nonlinear_gain'] for m in progressive_by_depth[d]]
            mean_gain = np.mean(gains)
            positive_count = sum(1 for g in gains if g > 0.0005)
            if mean_gain > 0.001:
                print(f"    *** BREAKTHROUGH at l={d}: prog mean gain={mean_gain:.5f} ***")
                breakthrough = True
            else:
                print(f"    l={d}: prog mean gain={mean_gain:.5f}, positive count={positive_count}/{len(gains)}")

    if not breakthrough:
        print("    No breakthrough detected (all deep layer gains < 0.001)")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Progressive vs Standard nonlinear gain
    ax = axes[0, 0]
    prog_means = [np.mean([m['nonlinear_gain'] for m in progressive_by_depth[d]]) for d in prog_depths]
    prog_stds = [np.std([m['nonlinear_gain'] for m in progressive_by_depth[d]]) for d in prog_depths]

    ax.errorbar(prog_depths, prog_means, yerr=prog_stds, marker='o', capsize=3,
               linewidth=2, markersize=8, label='Progressive', color='blue')

    if std_depths:
        std_means = [np.mean([m['nonlinear_gain'] for m in standard_by_depth[d]]) for d in std_depths]
        std_stds = [np.std([m['nonlinear_gain'] for m in standard_by_depth[d]]) for d in std_depths]
        ax.errorbar(std_depths, std_means, yerr=std_stds, marker='s', capsize=3,
                   linewidth=2, markersize=8, label='Standard', color='orange')

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=7, color='gray', linestyle=':', alpha=0.5, label='Ceiling (l≈7)')
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Progressive vs Standard: Nonlinear Gain')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: MSE comparison
    ax = axes[0, 1]
    prog_mses = [np.mean([m['mse_full'] for m in progressive_by_depth[d]]) for d in prog_depths]
    ax.plot(prog_depths, prog_mses, marker='o', linewidth=2, markersize=8,
           label='Progressive', color='blue')

    if std_depths:
        std_mses = [np.mean([m['mse_full'] for m in standard_by_depth[d]]) for d in std_depths]
        ax.plot(std_depths, std_mses, marker='s', linewidth=2, markersize=8,
               label='Standard', color='orange')

    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Progressive vs Standard: Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Linearity score comparison
    ax = axes[1, 0]
    prog_lins = [np.mean([m['linearity_score'] for m in progressive_by_depth[d]]) for d in prog_depths]
    ax.plot(prog_depths, prog_lins, marker='o', linewidth=2, markersize=8,
           label='Progressive', color='blue')

    if std_depths:
        std_lins = [np.mean([m['linearity_score'] for m in standard_by_depth[d]]) for d in std_depths]
        ax.plot(std_depths, std_lins, marker='s', linewidth=2, markersize=8,
               label='Standard', color='orange')

    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Linearity Score')
    ax.set_title('Progressive vs Standard: Linearity Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Progressive training trajectory (single seed visualization)
    ax = axes[1, 1]
    if results['progressive']:
        for seed_idx, run in enumerate(results['progressive']):
            depths = [m['depth'] for m in run['depth_progression']]
            gains = [m['nonlinear_gain'] for m in run['depth_progression']]
            ax.plot(depths, gains, marker='o', linewidth=1.5, markersize=6,
                   alpha=0.7, label=f'Seed {seed_idx}')

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=7, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Progressive Training Trajectories (All Seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/progressive_depth_training.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/progressive_depth_training.png")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Best depth for each method
    best_prog_depth = max(prog_depths, key=lambda d: np.mean([m['nonlinear_gain'] for m in progressive_by_depth[d]]))
    best_prog_gain = np.mean([m['nonlinear_gain'] for m in progressive_by_depth[best_prog_depth]])
    print(f"  Progressive best: l={best_prog_depth}, gain={best_prog_gain:.5f}")

    if std_depths:
        best_std_depth = max(std_depths, key=lambda d: np.mean([m['nonlinear_gain'] for m in standard_by_depth[d]]))
        best_std_gain = np.mean([m['nonlinear_gain'] for m in standard_by_depth[best_std_depth]])
        print(f"  Standard best: l={best_std_depth}, gain={best_std_gain:.5f}")

    # Overall winner at l≥8
    prog_gain_8plus = np.mean(prog_deep_gains) if prog_deep_gains else 0
    std_gain_8plus = np.mean(std_deep_gains) if std_deep_gains else 0
    print(f"\n  Mean gain at l≥8:")
    print(f"    Progressive: {prog_gain_8plus:.5f}")
    print(f"    Standard: {std_gain_8plus:.5f}")
    print(f"    Winner: {'PROGRESSIVE' if prog_gain_8plus > std_gain_8plus else 'STANDARD'}")

    # Calculate improvement
    if std_deep_gains and prog_deep_gains:
        improvement = prog_gain_8plus - std_gain_8plus
        print(f"    Progressive improvement: {improvement:+.5f}")

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

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR}/results.json")

    return {
        'comparison': comparison_data,
        'prog_gain_8plus': prog_gain_8plus,
        'std_gain_8plus': std_gain_8plus,
        'breakthrough': breakthrough
    }


if __name__ == "__main__":
    results = run_experiment()
    stats = analyze_and_plot(results)
