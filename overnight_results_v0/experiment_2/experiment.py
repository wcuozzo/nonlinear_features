"""
Experiment 2: Feature Co-occurrence Drives Nonlinear Encoding

Hypothesis: Nonlinear encodings develop specifically for features that frequently
co-occur in training data. The network learns nonlinear interactions because it
needs to disentangle commonly co-activated features.

This builds on Experiment 1's key finding:
- Pairwise nonlinearity (r=0.70) strongly predicts nonlinear gain
- Interference reduction (r=0.09) does not

Key questions:
1. Do features with correlated activation patterns develop more nonlinear encodings?
2. Can we induce stronger nonlinearity by increasing feature co-occurrence?
3. Is there a critical co-occurrence threshold that triggers nonlinear encoding?

Approach:
- Generate data with controlled feature co-occurrence structure
- Train autoencoders and measure which feature pairs become nonlinearly encoded
- Test if co-occurrence statistics predict pairwise encoding nonlinearity
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from pathlib import Path
import pandas as pd
from scipy import stats

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

RESULTS_DIR = Path(__file__).parent
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Model (from main notebook)
# =============================================================================

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


# =============================================================================
# Data Generation with Controlled Co-occurrence
# =============================================================================

def generate_independent_sparse_data(n_samples: int, n_features: int, sparsity: float = 0.1) -> torch.Tensor:
    """Generate sparse data where features are independently activated."""
    mask = (torch.rand(n_samples, n_features) < sparsity).float()
    values = torch.rand(n_samples, n_features)
    return (mask * values).to(device)


def generate_cooccurrence_data(
    n_samples: int,
    n_features: int,
    base_sparsity: float = 0.1,
    cooccurrence_matrix: torch.Tensor = None,
    cooccurrence_strength: float = 0.5
) -> torch.Tensor:
    """
    Generate data with controlled feature co-occurrence.

    If cooccurrence_matrix[i,j] > 0, then features i and j tend to activate together.
    cooccurrence_strength controls how strong the correlation is (0 = independent, 1 = fully coupled).
    """
    n = n_features

    # Start with independent activations
    base_mask = (torch.rand(n_samples, n) < base_sparsity).float()

    if cooccurrence_matrix is not None:
        # For each feature pair with positive co-occurrence, propagate activations
        for i in range(n):
            for j in range(i+1, n):
                if cooccurrence_matrix[i, j] > 0:
                    # When feature i is active, sometimes also activate feature j
                    i_active = base_mask[:, i] > 0
                    propagate_mask = torch.rand(n_samples) < (cooccurrence_strength * cooccurrence_matrix[i, j])
                    base_mask[:, j] = torch.where(
                        i_active & propagate_mask,
                        torch.ones_like(base_mask[:, j]),
                        base_mask[:, j]
                    )
                    # Symmetric: j -> i
                    j_active = base_mask[:, j] > 0
                    propagate_mask = torch.rand(n_samples) < (cooccurrence_strength * cooccurrence_matrix[i, j])
                    base_mask[:, i] = torch.where(
                        j_active & propagate_mask,
                        torch.ones_like(base_mask[:, i]),
                        base_mask[:, i]
                    )

    values = torch.rand(n_samples, n)
    return (base_mask * values).to(device)


def create_clustered_cooccurrence(n_features: int, n_clusters: int, within_cluster_prob: float = 0.8):
    """
    Create a co-occurrence matrix where features are grouped into clusters.
    Features within the same cluster have high co-occurrence.
    """
    cooc = torch.zeros(n_features, n_features)
    cluster_size = n_features // n_clusters

    for c in range(n_clusters):
        start = c * cluster_size
        end = start + cluster_size if c < n_clusters - 1 else n_features
        for i in range(start, end):
            for j in range(i+1, end):
                cooc[i, j] = within_cluster_prob
                cooc[j, i] = within_cluster_prob

    return cooc


def create_hierarchical_cooccurrence(n_features: int, n_levels: int = 3):
    """
    Create hierarchical co-occurrence: some pairs very correlated, some moderately, some independent.
    """
    cooc = torch.zeros(n_features, n_features)

    for level in range(n_levels):
        group_size = n_features // (2 ** level)
        prob = 0.9 - level * 0.3  # Decreasing probability with hierarchy level

        for g in range(2 ** level):
            start = g * group_size
            end = start + group_size
            for i in range(start, min(end, n_features)):
                for j in range(i+1, min(end, n_features)):
                    if cooc[i, j] == 0:  # Don't overwrite stronger relationships
                        cooc[i, j] = prob
                        cooc[j, i] = prob

    return cooc


def measure_empirical_cooccurrence(data: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """Measure empirical co-occurrence from generated data."""
    n_samples, n_features = data.shape
    active = (data > threshold).float()

    # Co-occurrence = P(both active) / sqrt(P(i active) * P(j active))
    marginals = active.mean(dim=0)
    joint = (active.T @ active) / n_samples

    # Normalized co-occurrence (like correlation)
    denom = torch.sqrt(torch.outer(marginals, marginals)) + 1e-8
    cooc = joint / denom

    return cooc


# =============================================================================
# Training
# =============================================================================

def train_autoencoder_with_data_fn(model, data_fn, n_steps=5000, batch_size=256, lr=1e-3):
    """Train autoencoder with custom data generation function."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(n_steps):
        x = data_fn(batch_size)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# =============================================================================
# Metrics
# =============================================================================

def measure_encoding_linearity(model, data_fn, n_samples=1000):
    """Measure how linear the learned encoding is."""
    model.eval()

    with torch.no_grad():
        x = data_fn(n_samples)
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


def compute_pairwise_nonlinearity_detailed(model, n_pairs=None, n_samples=100):
    """
    Compute nonlinearity for all feature pairs (or a sample).
    Returns a matrix of pairwise nonlinearities.
    """
    model.eval()
    n = model.n

    if n_pairs is None:
        # Compute for all pairs if n is small enough
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    else:
        # Sample random pairs
        pairs = []
        for _ in range(n_pairs):
            i, j = np.random.choice(n, 2, replace=False)
            if i > j:
                i, j = j, i
            if (i, j) not in pairs:
                pairs.append((i, j))

    nonlinearity_matrix = torch.zeros(n, n, device=device)

    with torch.no_grad():
        for i, j in pairs:
            mags = torch.rand(n_samples, device=device) * 0.8 + 0.1

            # x_i: only feature i active
            x_i = torch.zeros(n_samples, n, device=device)
            x_i[:, i] = mags
            z_i = model.encode(x_i)

            # x_j: only feature j active
            x_j = torch.zeros(n_samples, n, device=device)
            x_j[:, j] = mags
            z_j = model.encode(x_j)

            # x_both: both features active
            x_both = torch.zeros(n_samples, n, device=device)
            x_both[:, i] = mags
            x_both[:, j] = mags
            z_both = model.encode(x_both)

            # Encoding of zero
            x_zero = torch.zeros(1, n, device=device)
            z_zero = model.encode(x_zero)

            # Additivity check
            z_expected = z_i + z_j - z_zero

            deviation = (z_both - z_expected).norm(dim=1)
            expected_norm = z_expected.norm(dim=1)
            relative_deviation = (deviation / (expected_norm + 1e-8)).mean().item()

            nonlinearity_matrix[i, j] = relative_deviation
            nonlinearity_matrix[j, i] = relative_deviation

    return nonlinearity_matrix


def compute_pairwise_correlation_with_cooccurrence(nonlinearity_matrix, cooccurrence_matrix):
    """
    Compute correlation between pairwise nonlinearity and co-occurrence.
    """
    n = nonlinearity_matrix.shape[0]

    # Extract upper triangular (excluding diagonal)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()

    nonlin_values = nonlinearity_matrix[mask].cpu().numpy()
    cooc_values = cooccurrence_matrix[mask].cpu().numpy()

    correlation, p_value = stats.pearsonr(nonlin_values, cooc_values)

    return correlation, p_value, nonlin_values, cooc_values


# =============================================================================
# Main Experiments
# =============================================================================

def experiment_cooccurrence_strength():
    """
    Experiment A: How does co-occurrence strength affect nonlinear encoding?
    """
    print("\n" + "="*60)
    print("Experiment A: Co-occurrence Strength")
    print("="*60)

    n, m, l = 32, 8, 2
    base_sparsity = 0.1
    n_steps = 4000

    # Create fixed cluster structure
    cooc_matrix = create_clustered_cooccurrence(n, n_clusters=4, within_cluster_prob=1.0)

    strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for strength in tqdm(strengths, desc="Co-occurrence strength"):
        # Data generation function for this strength
        def data_fn(batch_size, s=strength, cm=cooc_matrix):
            return generate_cooccurrence_data(
                batch_size, n, base_sparsity, cm, s
            )

        model = Autoencoder(n, m, l).to(device)
        losses = train_autoencoder_with_data_fn(model, data_fn, n_steps=n_steps)

        linearity = measure_encoding_linearity(model, data_fn)
        nonlin_matrix = compute_pairwise_nonlinearity_detailed(model, n_pairs=200)

        # Measure correlation between nonlinearity and co-occurrence
        empirical_cooc = measure_empirical_cooccurrence(data_fn(5000).cpu())
        corr, p_val, _, _ = compute_pairwise_correlation_with_cooccurrence(
            nonlin_matrix, empirical_cooc.to(device)
        )

        results.append({
            'cooccurrence_strength': strength,
            **linearity,
            'avg_pairwise_nonlinearity': nonlin_matrix[nonlin_matrix > 0].mean().item(),
            'nonlinearity_cooccurrence_correlation': corr,
            'correlation_p_value': p_val,
            'final_loss': float(np.mean(losses[-100:]))
        })

        print(f"  strength={strength:.1f}: nonlin_gain={linearity['nonlinear_gain']:.4f}, "
              f"corr(nonlin, cooc)={corr:.3f}")

    return pd.DataFrame(results)


def experiment_cluster_structure():
    """
    Experiment B: Do clustered features develop cluster-specific nonlinear encodings?
    """
    print("\n" + "="*60)
    print("Experiment B: Cluster Structure")
    print("="*60)

    n, m, l = 32, 8, 2
    base_sparsity = 0.1
    n_steps = 4000
    n_clusters = 4

    # Train with clustered co-occurrence
    cooc_matrix = create_clustered_cooccurrence(n, n_clusters, within_cluster_prob=0.8)

    def data_fn(batch_size):
        return generate_cooccurrence_data(batch_size, n, base_sparsity, cooc_matrix, 0.7)

    # Train model
    model = Autoencoder(n, m, l).to(device)
    losses = train_autoencoder_with_data_fn(model, data_fn, n_steps=n_steps)

    # Get detailed pairwise nonlinearity
    nonlin_matrix = compute_pairwise_nonlinearity_detailed(model, n_pairs=None)  # All pairs

    # Compute within-cluster vs between-cluster nonlinearity
    cluster_size = n // n_clusters
    within_cluster_nonlin = []
    between_cluster_nonlin = []

    for i in range(n):
        for j in range(i+1, n):
            cluster_i = i // cluster_size
            cluster_j = j // cluster_size

            nonlin_val = nonlin_matrix[i, j].item()

            if cluster_i == cluster_j:
                within_cluster_nonlin.append(nonlin_val)
            else:
                between_cluster_nonlin.append(nonlin_val)

    results = {
        'within_cluster_nonlinearity': np.mean(within_cluster_nonlin),
        'between_cluster_nonlinearity': np.mean(between_cluster_nonlin),
        'within_std': np.std(within_cluster_nonlin),
        'between_std': np.std(between_cluster_nonlin),
        'n_within_pairs': len(within_cluster_nonlin),
        'n_between_pairs': len(between_cluster_nonlin)
    }

    # Statistical test
    t_stat, p_value = stats.ttest_ind(within_cluster_nonlin, between_cluster_nonlin)
    results['ttest_statistic'] = t_stat
    results['ttest_p_value'] = p_value

    print(f"\nWithin-cluster nonlinearity: {results['within_cluster_nonlinearity']:.4f} +/- {results['within_std']:.4f}")
    print(f"Between-cluster nonlinearity: {results['between_cluster_nonlinearity']:.4f} +/- {results['between_std']:.4f}")
    print(f"T-test p-value: {p_value:.6f}")

    return results, nonlin_matrix.cpu().numpy(), cooc_matrix.numpy()


def experiment_compression_cooccurrence_interaction():
    """
    Experiment C: How do compression ratio and co-occurrence interact?
    """
    print("\n" + "="*60)
    print("Experiment C: Compression x Co-occurrence Interaction")
    print("="*60)

    n = 32
    l = 2
    base_sparsity = 0.1
    n_steps = 4000

    m_values = [4, 8, 16, 24]  # Different compression levels
    cooc_strengths = [0.0, 0.3, 0.6, 0.9]

    cooc_matrix = create_clustered_cooccurrence(n, n_clusters=4, within_cluster_prob=1.0)

    results = []

    for m in tqdm(m_values, desc="Compression levels"):
        for strength in cooc_strengths:
            def data_fn(batch_size, s=strength, cm=cooc_matrix):
                return generate_cooccurrence_data(batch_size, n, base_sparsity, cm, s)

            model = Autoencoder(n, m, l).to(device)
            losses = train_autoencoder_with_data_fn(model, data_fn, n_steps=n_steps)

            linearity = measure_encoding_linearity(model, data_fn)

            results.append({
                'n': n, 'm': m, 'l': l,
                'compression_ratio': n / m,
                'cooccurrence_strength': strength,
                **linearity,
                'final_loss': float(np.mean(losses[-100:]))
            })

    return pd.DataFrame(results)


def experiment_independent_vs_structured():
    """
    Experiment D: Compare independent features vs structured co-occurrence
    at different parameter settings.
    """
    print("\n" + "="*60)
    print("Experiment D: Independent vs Structured Features")
    print("="*60)

    configs = [
        {'n': 32, 'm': 8, 'l': 2},
        {'n': 64, 'm': 16, 'l': 2},
        {'n': 64, 'm': 8, 'l': 2},
        {'n': 128, 'm': 16, 'l': 2},
    ]

    base_sparsity = 0.1
    n_steps = 4000

    results = []

    for config in tqdm(configs, desc="Configurations"):
        n, m, l = config['n'], config['m'], config['l']

        # Independent features
        def data_fn_indep(batch_size):
            return generate_independent_sparse_data(batch_size, n, base_sparsity)

        model_indep = Autoencoder(n, m, l).to(device)
        losses_indep = train_autoencoder_with_data_fn(model_indep, data_fn_indep, n_steps=n_steps)
        linearity_indep = measure_encoding_linearity(model_indep, data_fn_indep)

        # Structured co-occurrence
        cooc_matrix = create_hierarchical_cooccurrence(n, n_levels=3)

        def data_fn_struct(batch_size, cm=cooc_matrix):
            return generate_cooccurrence_data(batch_size, n, base_sparsity, cm, 0.7)

        model_struct = Autoencoder(n, m, l).to(device)
        losses_struct = train_autoencoder_with_data_fn(model_struct, data_fn_struct, n_steps=n_steps)
        linearity_struct = measure_encoding_linearity(model_struct, data_fn_struct)

        results.append({
            **config,
            'compression_ratio': n / m,
            'independent_nonlinear_gain': linearity_indep['nonlinear_gain'],
            'independent_mse': linearity_indep['mse_full'],
            'structured_nonlinear_gain': linearity_struct['nonlinear_gain'],
            'structured_mse': linearity_struct['mse_full'],
            'nonlinear_gain_increase': linearity_struct['nonlinear_gain'] - linearity_indep['nonlinear_gain'],
        })

        print(f"  n={n}, m={m}: indep={linearity_indep['nonlinear_gain']:.4f}, "
              f"struct={linearity_struct['nonlinear_gain']:.4f}, "
              f"delta={linearity_struct['nonlinear_gain'] - linearity_indep['nonlinear_gain']:.4f}")

    return pd.DataFrame(results)


# =============================================================================
# Visualization
# =============================================================================

def plot_results(exp_a_results, exp_b_results, exp_c_results, exp_d_results,
                 nonlin_matrix, cooc_matrix):
    """Create comprehensive visualization of all experiments."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot A: Co-occurrence strength effect
    ax = axes[0, 0]
    ax.plot(exp_a_results['cooccurrence_strength'], exp_a_results['nonlinear_gain'],
            'o-', linewidth=2, markersize=8, label='Nonlinear Gain')
    ax.set_xlabel('Co-occurrence Strength')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('A: Co-occurrence Strength Effect')
    ax.grid(True, alpha=0.3)

    # Secondary y-axis for correlation
    ax2 = ax.twinx()
    ax2.plot(exp_a_results['cooccurrence_strength'],
             exp_a_results['nonlinearity_cooccurrence_correlation'],
             's--', color='orange', linewidth=2, markersize=8, label='Corr(nonlin, cooc)')
    ax2.set_ylabel('Correlation', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Plot B: Nonlinearity matrix vs co-occurrence
    ax = axes[0, 1]
    im = ax.imshow(nonlin_matrix, cmap='hot', aspect='auto')
    ax.set_title('B: Pairwise Nonlinearity Matrix')
    ax.set_xlabel('Feature j')
    ax.set_ylabel('Feature i')
    plt.colorbar(im, ax=ax, label='Nonlinearity')

    ax = axes[0, 2]
    im = ax.imshow(cooc_matrix, cmap='Blues', aspect='auto')
    ax.set_title('B: Co-occurrence Structure')
    ax.set_xlabel('Feature j')
    ax.set_ylabel('Feature i')
    plt.colorbar(im, ax=ax, label='Co-occurrence')

    # Plot C: Compression x Co-occurrence heatmap
    ax = axes[1, 0]
    pivot = exp_c_results.pivot(index='m', columns='cooccurrence_strength', values='nonlinear_gain')
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{c:.1f}' for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Co-occurrence Strength')
    ax.set_ylabel('Bottleneck (m)')
    ax.set_title('C: Compression x Co-occurrence')
    plt.colorbar(im, ax=ax, label='Nonlinear Gain')

    # Plot D: Independent vs Structured comparison
    ax = axes[1, 1]
    x = np.arange(len(exp_d_results))
    width = 0.35
    bars1 = ax.bar(x - width/2, exp_d_results['independent_nonlinear_gain'], width,
                   label='Independent', color='steelblue')
    bars2 = ax.bar(x + width/2, exp_d_results['structured_nonlinear_gain'], width,
                   label='Structured', color='coral')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('D: Independent vs Structured Features')
    ax.set_xticks(x)
    labels = [f"n={r['n']},m={r['m']}" for _, r in exp_d_results.iterrows()]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot: Within vs Between cluster nonlinearity (from exp B)
    ax = axes[1, 2]
    categories = ['Within\nCluster', 'Between\nCluster']
    values = [exp_b_results['within_cluster_nonlinearity'],
              exp_b_results['between_cluster_nonlinearity']]
    errors = [exp_b_results['within_std'], exp_b_results['between_std']]
    bars = ax.bar(categories, values, yerr=errors, capsize=5,
                  color=['coral', 'steelblue'], alpha=0.8)
    ax.set_ylabel('Average Pairwise Nonlinearity')
    ax.set_title(f'B: Cluster Structure Effect\n(p={exp_b_results["ttest_p_value"]:.4f})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'main_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved main_results.png")


def plot_detailed_cooccurrence_analysis(nonlin_matrix, cooc_matrix):
    """Scatter plot of pairwise nonlinearity vs co-occurrence."""
    fig, ax = plt.subplots(figsize=(8, 6))

    n = nonlin_matrix.shape[0]
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)

    nonlin_values = nonlin_matrix[mask]
    cooc_values = cooc_matrix[mask]

    ax.scatter(cooc_values, nonlin_values, alpha=0.5, s=20)

    # Fit line
    slope, intercept, r, p, se = stats.linregress(cooc_values, nonlin_values)
    x_line = np.array([cooc_values.min(), cooc_values.max()])
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
            label=f'r = {r:.3f}, p = {p:.4f}')

    ax.set_xlabel('Feature Co-occurrence')
    ax.set_ylabel('Pairwise Encoding Nonlinearity')
    ax.set_title('Does Co-occurrence Predict Nonlinear Encoding?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'cooccurrence_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved cooccurrence_scatter.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("Experiment 2: Feature Co-occurrence Drives Nonlinear Encoding")
    print("="*60)

    # Run all experiments
    exp_a_results = experiment_cooccurrence_strength()
    exp_b_results, nonlin_matrix, cooc_matrix = experiment_cluster_structure()
    exp_c_results = experiment_compression_cooccurrence_interaction()
    exp_d_results = experiment_independent_vs_structured()

    # Save results
    exp_a_results.to_csv(RESULTS_DIR / 'exp_a_cooccurrence_strength.csv', index=False)
    exp_c_results.to_csv(RESULTS_DIR / 'exp_c_compression_interaction.csv', index=False)
    exp_d_results.to_csv(RESULTS_DIR / 'exp_d_independent_vs_structured.csv', index=False)

    with open(RESULTS_DIR / 'exp_b_cluster_results.json', 'w') as f:
        json.dump(exp_b_results, f, indent=2)

    np.save(RESULTS_DIR / 'nonlinearity_matrix.npy', nonlin_matrix)
    np.save(RESULTS_DIR / 'cooccurrence_matrix.npy', cooc_matrix)

    # Visualizations
    plot_results(exp_a_results, exp_b_results, exp_c_results, exp_d_results,
                 nonlin_matrix, cooc_matrix)
    plot_detailed_cooccurrence_analysis(nonlin_matrix, cooc_matrix)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nExperiment A: Co-occurrence Strength Effect")
    print(f"  Correlation between strength and nonlinear gain: "
          f"{exp_a_results['cooccurrence_strength'].corr(exp_a_results['nonlinear_gain']):.3f}")

    print("\nExperiment B: Cluster Structure")
    print(f"  Within-cluster nonlinearity: {exp_b_results['within_cluster_nonlinearity']:.4f}")
    print(f"  Between-cluster nonlinearity: {exp_b_results['between_cluster_nonlinearity']:.4f}")
    print(f"  Difference significant: p = {exp_b_results['ttest_p_value']:.6f}")

    print("\nExperiment C: Compression x Co-occurrence")
    # Find optimal combination
    best_row = exp_c_results.loc[exp_c_results['nonlinear_gain'].idxmax()]
    print(f"  Highest nonlinear gain at: m={best_row['m']}, cooc_strength={best_row['cooccurrence_strength']}")

    print("\nExperiment D: Independent vs Structured")
    avg_increase = exp_d_results['nonlinear_gain_increase'].mean()
    print(f"  Average nonlinear gain increase with structure: {avg_increase:.4f}")

    # Compute key finding: correlation between co-occurrence and nonlinearity
    mask = np.triu(np.ones_like(nonlin_matrix), k=1).astype(bool)
    corr, p = stats.pearsonr(nonlin_matrix[mask].flatten(), cooc_matrix[mask].flatten())

    print(f"\nKEY FINDING:")
    print(f"  Correlation between co-occurrence and pairwise nonlinearity: {corr:.3f} (p={p:.6f})")

    # Save summary
    summary = {
        'exp_a_strength_gain_correlation': float(exp_a_results['cooccurrence_strength'].corr(exp_a_results['nonlinear_gain'])),
        'exp_b_within_cluster_nonlinearity': exp_b_results['within_cluster_nonlinearity'],
        'exp_b_between_cluster_nonlinearity': exp_b_results['between_cluster_nonlinearity'],
        'exp_b_ttest_p_value': exp_b_results['ttest_p_value'],
        'exp_d_avg_nonlinear_gain_increase': float(avg_increase),
        'cooccurrence_nonlinearity_correlation': float(corr),
        'cooccurrence_nonlinearity_p_value': float(p),
    }

    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    summary = main()
    print("\nExperiment complete! Results saved to:", RESULTS_DIR)
