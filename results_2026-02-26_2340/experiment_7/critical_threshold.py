"""
Experiment 7: Critical Training Threshold for Nonlinear Encoding

Key question: When during training does nonlinear encoding emerge?

Prior findings:
- Exp 5-6: Underfitting causes zero/negative nonlinear gain; proper training reveals strong benefits
- But at what point does the network "switch" from linear to nonlinear encoding?
- Is it a gradual transition or a sharp phase transition?

Design:
- Track nonlinear gain throughout training (checkpoints every N steps)
- Test multiple configurations (depth, compression) to see if thresholds vary
- Look for "switching points" where nonlinear gain becomes positive
- Hypothesis: There's a critical training step where nonlinear encoding emerges
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

OUTPUT_DIR = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_7"


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


def train_with_checkpoints(model, total_steps, checkpoint_interval, batch_size=256, sparsity=0.1, lr=1e-3):
    """Train model and measure metrics at regular checkpoints."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    checkpoints = []
    losses = []

    for step in range(total_steps):
        x = generate_sparse_data(batch_size, model.n, sparsity)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Checkpoint at regular intervals
        if (step + 1) % checkpoint_interval == 0 or step == 0:
            metrics = measure_encoding_linearity(model, sparsity=sparsity)
            latent_stats = measure_latent_stats(model, sparsity=sparsity)

            checkpoint = {
                'step': step + 1,
                'loss': np.mean(losses[-min(10, len(losses)):]),
                **metrics,
                **latent_stats
            }
            checkpoints.append(checkpoint)

    return checkpoints


def find_critical_threshold(checkpoints):
    """Find the step where nonlinear gain becomes consistently positive."""
    gains = [c['nonlinear_gain'] for c in checkpoints]
    steps = [c['step'] for c in checkpoints]

    # Find first point where gain > 0 and stays positive
    critical_step = None
    for i, (step, gain) in enumerate(zip(steps, gains)):
        if gain > 0:
            # Check if it stays positive (at least for a few more steps)
            remaining = gains[i:]
            if len(remaining) >= 3 and np.mean(remaining[:3]) > 0:
                critical_step = step
                break
            elif len(remaining) < 3 and np.mean(remaining) > 0:
                critical_step = step
                break

    return critical_step


def run_experiment():
    """Main experiment: track nonlinear gain evolution during training."""

    # Test configurations: vary depth and compression
    configs = [
        {'n': 64, 'm': 4, 'l': 2},   # High compression, moderate depth
        {'n': 64, 'm': 8, 'l': 2},   # Moderate compression, moderate depth
        {'n': 64, 'm': 4, 'l': 4},   # High compression, deeper
        {'n': 64, 'm': 8, 'l': 4},   # Moderate compression, deeper
        {'n': 128, 'm': 8, 'l': 2},  # Larger n, high compression
        {'n': 128, 'm': 8, 'l': 4},  # Larger n, deeper
    ]

    sparsity = 0.1
    total_steps = 400
    checkpoint_interval = 20  # Measure every 20 steps
    n_seeds = 2

    all_results = []

    print("=" * 60)
    print("Experiment 7: Critical Training Threshold")
    print("=" * 60)
    print(f"Configs: {len(configs)}")
    print(f"Total steps: {total_steps}, Checkpoint interval: {checkpoint_interval}")
    print(f"Seeds: {n_seeds}")
    print("=" * 60)

    for config in configs:
        n, m, l = config['n'], config['m'], config['l']
        compression_ratio = n // m

        print(f"\n--- Config: n={n}, m={m}, l={l}, CR={compression_ratio} ---")

        for seed in range(n_seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            model = Autoencoder(n, m, l).to(device)
            n_params = sum(p.numel() for p in model.parameters())

            checkpoints = train_with_checkpoints(
                model, total_steps, checkpoint_interval, sparsity=sparsity
            )

            critical_step = find_critical_threshold(checkpoints)
            final_gain = checkpoints[-1]['nonlinear_gain']

            result = {
                'n': n,
                'm': m,
                'l': l,
                'compression_ratio': compression_ratio,
                'seed': seed,
                'n_params': n_params,
                'critical_step': critical_step,
                'final_nonlinear_gain': final_gain,
                'checkpoints': checkpoints
            }
            all_results.append(result)

            status = f"step {critical_step}" if critical_step else "never"
            print(f"  Seed {seed}: critical threshold @ {status}, "
                  f"final gain={final_gain:.6f}")

    return all_results


def analyze_and_plot(results):
    """Analyze results and create visualizations."""

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Nonlinear gain trajectories by config
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results) // 2))
    config_idx = {}

    for i, result in enumerate(results):
        config_key = f"n={result['n']},m={result['m']},l={result['l']}"
        if config_key not in config_idx:
            config_idx[config_key] = len(config_idx)

        color = colors[config_idx[config_key]]
        steps = [c['step'] for c in result['checkpoints']]
        gains = [c['nonlinear_gain'] for c in result['checkpoints']]

        alpha = 0.7 if result['seed'] == 0 else 0.4
        ax.plot(steps, gains, color=color, alpha=alpha,
                label=config_key if result['seed'] == 0 else None)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain Evolution During Training')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 2: MSE trajectories
    ax = axes[0, 1]
    for i, result in enumerate(results):
        config_key = f"n={result['n']},m={result['m']},l={result['l']}"
        color = colors[config_idx[config_key]]
        steps = [c['step'] for c in result['checkpoints']]
        mse = [c['mse_full'] for c in result['checkpoints']]

        alpha = 0.7 if result['seed'] == 0 else 0.4
        ax.plot(steps, mse, color=color, alpha=alpha,
                label=config_key if result['seed'] == 0 else None)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('MSE')
    ax.set_title('Reconstruction Error During Training')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 3: Linearity score evolution
    ax = axes[0, 2]
    for i, result in enumerate(results):
        config_key = f"n={result['n']},m={result['m']},l={result['l']}"
        color = colors[config_idx[config_key]]
        steps = [c['step'] for c in result['checkpoints']]
        linearity = [c['linearity_score'] for c in result['checkpoints']]

        alpha = 0.7 if result['seed'] == 0 else 0.4
        ax.plot(steps, linearity, color=color, alpha=alpha,
                label=config_key if result['seed'] == 0 else None)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Linearity Score')
    ax.set_title('Encoder Linearity During Training')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 4: Critical threshold vs compression
    ax = axes[1, 0]
    critical_data = {}
    for result in results:
        key = (result['compression_ratio'], result['l'])
        if key not in critical_data:
            critical_data[key] = []
        if result['critical_step'] is not None:
            critical_data[key].append(result['critical_step'])

    for (cr, l), steps in critical_data.items():
        if steps:
            ax.scatter([cr] * len(steps), steps, s=80, alpha=0.7,
                      label=f"l={l}", marker='o' if l == 2 else 's')

    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Critical Step (first positive gain)')
    ax.set_title('Critical Threshold vs Compression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Critical threshold vs depth
    ax = axes[1, 1]
    for result in results:
        if result['critical_step'] is not None:
            marker = 'o' if result['n'] == 64 else 's'
            ax.scatter(result['l'], result['critical_step'],
                      s=80, alpha=0.7, marker=marker,
                      label=f"n={result['n']}" if result['seed'] == 0 else None)

    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Critical Step')
    ax.set_title('Critical Threshold vs Depth')
    ax.grid(True, alpha=0.3)
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Plot 6: Final gain vs critical step
    ax = axes[1, 2]
    for result in results:
        if result['critical_step'] is not None:
            cr = result['compression_ratio']
            color = 'blue' if cr >= 16 else 'orange'
            ax.scatter(result['critical_step'], result['final_nonlinear_gain'],
                      s=80, alpha=0.7, color=color,
                      marker='o' if result['l'] == 2 else 's')

    # Add legend for colors
    ax.scatter([], [], color='blue', s=80, label='CR≥16')
    ax.scatter([], [], color='orange', s=80, label='CR<16')
    ax.scatter([], [], marker='o', color='gray', s=80, label='l=2')
    ax.scatter([], [], marker='s', color='gray', s=80, label='l=4')

    ax.set_xlabel('Critical Step')
    ax.set_ylabel('Final Nonlinear Gain')
    ax.set_title('Earlier Threshold → Higher Final Gain?')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/critical_threshold.png', dpi=150)
    plt.close()
    print(f"\nSaved plot to {OUTPUT_DIR}/critical_threshold.png")

    # Compute statistics
    stats = {}

    # Critical thresholds by config type
    thresholds = [r['critical_step'] for r in results if r['critical_step'] is not None]
    stats['mean_critical_step'] = np.mean(thresholds) if thresholds else None
    stats['std_critical_step'] = np.std(thresholds) if thresholds else None
    stats['min_critical_step'] = np.min(thresholds) if thresholds else None
    stats['max_critical_step'] = np.max(thresholds) if thresholds else None

    # Configs that never reached positive gain
    never_positive = [r for r in results if r['critical_step'] is None]
    stats['configs_never_positive'] = len(never_positive)

    # Correlation between critical step and final gain
    valid = [(r['critical_step'], r['final_nonlinear_gain'])
             for r in results if r['critical_step'] is not None]
    if len(valid) >= 3:
        steps, gains = zip(*valid)
        stats['corr_threshold_vs_final'] = np.corrcoef(steps, gains)[0, 1]

    # By depth
    for l in [2, 4]:
        subset = [r for r in results if r['l'] == l and r['critical_step'] is not None]
        if subset:
            stats[f'l={l}_mean_threshold'] = np.mean([r['critical_step'] for r in subset])
            stats[f'l={l}_mean_final_gain'] = np.mean([r['final_nonlinear_gain'] for r in subset])

    print("\n" + "=" * 60)
    print("KEY STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        if v is not None and isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Summary by config
    print("\n" + "=" * 60)
    print("SUMMARY BY CONFIG")
    print("=" * 60)

    config_summary = {}
    for result in results:
        key = f"n={result['n']}, m={result['m']}, l={result['l']}"
        if key not in config_summary:
            config_summary[key] = {
                'critical_steps': [],
                'final_gains': [],
                'compression_ratio': result['compression_ratio']
            }
        if result['critical_step'] is not None:
            config_summary[key]['critical_steps'].append(result['critical_step'])
        config_summary[key]['final_gains'].append(result['final_nonlinear_gain'])

    for key, data in sorted(config_summary.items()):
        cr = data['compression_ratio']
        avg_thresh = np.mean(data['critical_steps']) if data['critical_steps'] else "N/A"
        avg_gain = np.mean(data['final_gains'])
        thresh_str = f"{avg_thresh:.0f}" if isinstance(avg_thresh, float) else avg_thresh
        print(f"  {key} (CR={cr}): threshold={thresh_str}, final_gain={avg_gain:.6f}")

    # Save results
    def make_serializable(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    serializable_results = make_serializable(results)

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump({
            'results': serializable_results,
            'stats': {k: float(v) if isinstance(v, (np.floating, float)) else v
                     for k, v in stats.items() if v is not None}
        }, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR}/results.json")

    return stats, config_summary


if __name__ == "__main__":
    results = run_experiment()
    stats, config_summary = analyze_and_plot(results)
