"""
Experiment 1: Manifold Learning and Latent Rank Analysis

Hypothesis: Deeper networks achieve nonlinear encoding by learning lower-rank
representations in latent space. This follows from Experiment 3 (v0) findings that:
- Depth decreases feature separation (r=-0.76)
- Depth benefit negatively correlates with effective dimensionality (r=-0.56)
- This suggests depth enables "manifold learning" rather than feature separation

This experiment directly tests whether deeper networks produce lower effective rank
in the latent encodings, and how this relates to nonlinear gain.

Key measurements:
1. Effective rank of latent representations across depth
2. Covariance structure of latent space
3. Relationship between latent rank and nonlinear gain
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from tqdm import tqdm
from typing import Dict, List, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# Model and Data (from notebook)
# ============================================================

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


def train_autoencoder(model, n_steps=3000, batch_size=256, sparsity=0.1, lr=1e-3, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    iterator = tqdm(range(n_steps), disable=not verbose)
    for step in iterator:
        x = generate_sparse_data(batch_size, model.n, sparsity)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def measure_encoding_linearity(model, n_samples=1000, sparsity=0.1):
    model.eval()
    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)
        z = model.encode(x)

        x_with_bias = torch.cat([x, torch.ones(n_samples, 1, device=device)], dim=1)
        W_linear = torch.linalg.lstsq(x_with_bias, z).solution
        z_linear = x_with_bias @ W_linear

        z_var = z.var(dim=0).sum()
        residual_var = (z - z_linear).var(dim=0).sum()
        linearity_score = 1 - (residual_var / z_var).item()

        x_recon_full, _ = model(x)
        x_recon_linear = model.decode(z_linear)

        mse_full = nn.functional.mse_loss(x_recon_full, x).item()
        mse_linear = nn.functional.mse_loss(x_recon_linear, x).item()

    return {
        'linearity_score': linearity_score,
        'mse_full': mse_full,
        'mse_linear': mse_linear,
        'nonlinear_gain': (mse_linear - mse_full) / (mse_linear + 1e-8)
    }


# ============================================================
# NEW METRICS: Latent Space Rank Analysis
# ============================================================

def compute_latent_rank_metrics(model, n_samples=1000, sparsity=0.1) -> Dict:
    """
    Compute effective rank and covariance structure of latent representations.

    Metrics:
    - effective_rank: exp(entropy of normalized singular values)
    - rank_ratio: effective_rank / m (how much of the bottleneck is used)
    - top_1_var_fraction: fraction of variance in first singular value
    - top_3_var_fraction: fraction in top 3 singular values
    """
    model.eval()
    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)
        z = model.encode(x)

        # Center the latent codes
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Compute SVD of latent codes
        U, S, Vh = torch.linalg.svd(z_centered, full_matrices=False)

        # Normalized singular values (for entropy calculation)
        S_norm = S / S.sum()

        # Effective rank (exponential of entropy)
        entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
        effective_rank = torch.exp(entropy).item()

        # Variance fractions
        var_total = (S ** 2).sum()
        top_1_var = (S[0] ** 2) / var_total if len(S) > 0 else 0
        top_3_var = (S[:3] ** 2).sum() / var_total if len(S) >= 3 else top_1_var

        # Covariance eigenspectrum (for visualization)
        cov = z_centered.T @ z_centered / n_samples
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals.flip(0)  # descending order

    return {
        'effective_rank': effective_rank,
        'rank_ratio': effective_rank / model.m,
        'top_1_var_fraction': top_1_var.item(),
        'top_3_var_fraction': top_3_var.item(),
        'singular_values': S.cpu().numpy(),
        'eigenvalues': eigvals.cpu().numpy()
    }


def compute_layer_ranks(model, n_samples=500, sparsity=0.1) -> Dict:
    """
    For deeper networks, compute effective rank at each layer of the encoder.
    This helps understand where rank reduction happens.
    """
    model.eval()
    layer_ranks = []

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, sparsity)

        # Forward through encoder layer by layer
        h = x
        for i, layer in enumerate(model.encoder):
            h = layer(h)
            if isinstance(layer, nn.Linear):
                # Compute effective rank at this layer
                h_centered = h - h.mean(dim=0, keepdim=True)
                S = torch.linalg.svdvals(h_centered)
                S_norm = S / S.sum()
                entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
                eff_rank = torch.exp(entropy).item()
                layer_ranks.append({
                    'layer': i,
                    'dim': h.shape[1],
                    'effective_rank': eff_rank,
                    'rank_ratio': eff_rank / h.shape[1]
                })

    return layer_ranks


# ============================================================
# EXPERIMENT: Depth vs Latent Rank
# ============================================================

def run_depth_rank_experiment(n=64, m=8, l_values=[1, 2, 3, 4, 5, 6],
                               sparsity=0.1, n_steps=100, n_seeds=3):
    """
    Main experiment: How does depth affect latent space rank?
    """
    results = []

    for l in l_values:
        for seed in range(n_seeds):
            torch.manual_seed(seed * 1000 + l)

            model = Autoencoder(n, m, l).to(device)
            train_autoencoder(model, n_steps=n_steps, sparsity=sparsity, verbose=False)

            lin_metrics = measure_encoding_linearity(model, sparsity=sparsity)
            rank_metrics = compute_latent_rank_metrics(model, sparsity=sparsity)
            layer_ranks = compute_layer_ranks(model, sparsity=sparsity)

            result = {
                'n': n,
                'm': m,
                'l': l,
                'seed': seed,
                **lin_metrics,
                'effective_rank': rank_metrics['effective_rank'],
                'rank_ratio': rank_metrics['rank_ratio'],
                'top_1_var_fraction': rank_metrics['top_1_var_fraction'],
                'top_3_var_fraction': rank_metrics['top_3_var_fraction'],
                'final_layer_rank_ratio': layer_ranks[-1]['rank_ratio'] if layer_ranks else None
            }
            results.append(result)
            print(f"l={l}, seed={seed}: rank_ratio={rank_metrics['rank_ratio']:.3f}, "
                  f"nonlinear_gain={lin_metrics['nonlinear_gain']:.4f}")

    return pd.DataFrame(results)


def run_compression_rank_experiment(n=64, m_values=[4, 8, 16, 32], l_values=[1, 3],
                                     sparsity=0.1, n_steps=100, n_seeds=2):
    """
    How does compression interact with depth for latent rank?
    """
    results = []

    for m in m_values:
        for l in l_values:
            for seed in range(n_seeds):
                torch.manual_seed(seed * 1000 + m * 100 + l)

                model = Autoencoder(n, m, l).to(device)
                train_autoencoder(model, n_steps=n_steps, sparsity=sparsity, verbose=False)

                lin_metrics = measure_encoding_linearity(model, sparsity=sparsity)
                rank_metrics = compute_latent_rank_metrics(model, sparsity=sparsity)

                result = {
                    'n': n,
                    'm': m,
                    'l': l,
                    'seed': seed,
                    'compression_ratio': n / m,
                    **lin_metrics,
                    'effective_rank': rank_metrics['effective_rank'],
                    'rank_ratio': rank_metrics['rank_ratio'],
                    'top_1_var_fraction': rank_metrics['top_1_var_fraction'],
                    'top_3_var_fraction': rank_metrics['top_3_var_fraction'],
                }
                results.append(result)

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    output_dir = "/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/results_2026-02-26_2340/experiment_1"

    print("=" * 60)
    print("Experiment 1: Manifold Learning and Latent Rank Analysis")
    print("=" * 60)

    # Exp A: Depth sweep for rank analysis
    print("\n--- Experiment A: Depth vs Latent Rank ---")
    df_depth = run_depth_rank_experiment(
        n=64, m=8,
        l_values=[1, 2, 3, 4],
        sparsity=0.1,
        n_steps=500,  # More steps for meaningful nonlinear gain
        n_seeds=3
    )
    df_depth.to_csv(f"{output_dir}/exp_a_depth_rank.csv", index=False)

    # Exp B: Compression x depth interaction
    print("\n--- Experiment B: Compression x Depth Interaction ---")
    df_comp = run_compression_rank_experiment(
        n=64,
        m_values=[4, 8, 16, 32],
        l_values=[1, 3],
        sparsity=0.1,
        n_steps=500,
        n_seeds=2
    )
    df_comp.to_csv(f"{output_dir}/exp_b_compression_depth.csv", index=False)

    # Compute correlations
    print("\n--- Key Correlations ---")
    df_all = pd.concat([df_depth, df_comp]).drop_duplicates()

    correlations = {
        'rank_ratio_vs_nonlinear_gain': df_all['rank_ratio'].corr(df_all['nonlinear_gain']),
        'depth_vs_rank_ratio': df_all['l'].corr(df_all['rank_ratio']),
        'depth_vs_nonlinear_gain': df_all['l'].corr(df_all['nonlinear_gain']),
        'top1_var_vs_nonlinear_gain': df_all['top_1_var_fraction'].corr(df_all['nonlinear_gain']),
    }

    for k, v in correlations.items():
        print(f"  {k}: {v:.3f}")

    with open(f"{output_dir}/correlations.json", 'w') as f:
        json.dump(correlations, f, indent=2)

    # ============================================================
    # PLOTTING
    # ============================================================

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # A1: Depth vs rank ratio
    ax = axes[0, 0]
    depth_means = df_depth.groupby('l').agg({'rank_ratio': 'mean', 'nonlinear_gain': 'mean'}).reset_index()
    ax.bar(depth_means['l'] - 0.15, depth_means['rank_ratio'], width=0.3, label='Rank Ratio', color='steelblue')
    ax2 = ax.twinx()
    ax2.bar(depth_means['l'] + 0.15, depth_means['nonlinear_gain'], width=0.3, label='Nonlinear Gain', color='coral')
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Rank Ratio', color='steelblue')
    ax2.set_ylabel('Nonlinear Gain', color='coral')
    ax.set_title('Depth vs Latent Rank & Nonlinear Gain')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # A2: Scatter of rank ratio vs nonlinear gain
    ax = axes[0, 1]
    scatter = ax.scatter(df_all['rank_ratio'], df_all['nonlinear_gain'],
                         c=df_all['l'], cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax, label='Depth (l)')
    ax.set_xlabel('Rank Ratio (effective_rank / m)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title(f"Rank Ratio vs Nonlinear Gain\n(r={correlations['rank_ratio_vs_nonlinear_gain']:.3f})")

    # A3: Top-1 variance fraction vs nonlinear gain
    ax = axes[0, 2]
    scatter = ax.scatter(df_all['top_1_var_fraction'], df_all['nonlinear_gain'],
                         c=df_all['l'], cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax, label='Depth (l)')
    ax.set_xlabel('Top-1 Variance Fraction')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title(f"Variance Concentration vs Nonlinear Gain\n(r={correlations['top1_var_vs_nonlinear_gain']:.3f})")

    # B1: Compression x depth heatmap for rank ratio
    ax = axes[1, 0]
    pivot_rank = df_comp.groupby(['m', 'l'])['rank_ratio'].mean().unstack()
    im = ax.imshow(pivot_rank.values, aspect='auto', cmap='YlOrRd', origin='lower')
    ax.set_xticks(range(len(pivot_rank.columns)))
    ax.set_xticklabels(pivot_rank.columns)
    ax.set_yticks(range(len(pivot_rank.index)))
    ax.set_yticklabels(pivot_rank.index)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Bottleneck (m)')
    ax.set_title('Rank Ratio by Compression & Depth')
    plt.colorbar(im, ax=ax, label='Rank Ratio')

    # B2: Compression x depth heatmap for nonlinear gain
    ax = axes[1, 1]
    pivot_gain = df_comp.groupby(['m', 'l'])['nonlinear_gain'].mean().unstack()
    im = ax.imshow(pivot_gain.values, aspect='auto', cmap='RdYlGn', origin='lower')
    ax.set_xticks(range(len(pivot_gain.columns)))
    ax.set_xticklabels(pivot_gain.columns)
    ax.set_yticks(range(len(pivot_gain.index)))
    ax.set_yticklabels(pivot_gain.index)
    ax.set_xlabel('Depth (l)')
    ax.set_ylabel('Bottleneck (m)')
    ax.set_title('Nonlinear Gain by Compression & Depth')
    plt.colorbar(im, ax=ax, label='Nonlinear Gain')

    # B3: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    # Build rank summary string dynamically
    rank_summary_lines = []
    for l_val in sorted(depth_means['l'].unique()):
        rr = depth_means[depth_means['l']==l_val]['rank_ratio'].values[0]
        rank_summary_lines.append(f"    • l={l_val}: {rr:.3f}")
    rank_summary_str = "\n".join(rank_summary_lines)

    summary_text = f"""
    Key Findings:

    Correlations:
    • Rank ratio vs Nonlinear gain: {correlations['rank_ratio_vs_nonlinear_gain']:.3f}
    • Depth vs Rank ratio: {correlations['depth_vs_rank_ratio']:.3f}
    • Depth vs Nonlinear gain: {correlations['depth_vs_nonlinear_gain']:.3f}
    • Top-1 var vs Nonlinear gain: {correlations['top1_var_vs_nonlinear_gain']:.3f}

    Rank Ratio by Depth:
{rank_summary_str}

    Interpretation:
    {'Deeper networks → lower rank (manifold learning)' if correlations['depth_vs_rank_ratio'] < 0 else 'Deeper networks → higher rank (spreading)'}
    {'Lower rank → more nonlinear gain' if correlations['rank_ratio_vs_nonlinear_gain'] < 0 else 'Higher rank → more nonlinear gain'}
    """
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/main_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save summary
    summary = {
        'correlations': correlations,
        'depth_rank_means': depth_means.to_dict('records'),
        'total_samples': len(df_all),
        'parameters': {
            'n': 64,
            'm_values': [4, 8, 16, 32],
            'l_values': [1, 2, 3, 4, 5, 6],
            'n_steps': 100,
            'sparsity': 0.1
        }
    }

    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Experiment 1 Complete!")
    print(f"Results saved to {output_dir}")
    print("=" * 60)
