"""
Experiment 1: Feature Interference and Superposition Analysis

Hypothesis: Nonlinear encodings emerge when the network needs to resolve
interference between features that would collide in a purely linear encoding.

Key questions:
1. Do features that would be nearly parallel in a linear encoding show more
   nonlinear encoding behavior?
2. Does the network use nonlinearity to "route around" feature interference?
3. Can we measure superposition quality and relate it to nonlinear gain?

Approach:
- Train autoencoders with varying (n, m, l)
- Measure feature interference in the linear vs learned encoding
- Compute superposition metrics (how many features are distinguishable)
- Correlate interference/superposition with nonlinear gain
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

RESULTS_DIR = Path(__file__).parent
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Model and Data (from main notebook)
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


def generate_sparse_data(n_samples: int, n_features: int, sparsity: float = 0.1) -> torch.Tensor:
    mask = (torch.rand(n_samples, n_features) < sparsity).float()
    values = torch.rand(n_samples, n_features)
    return (mask * values).to(device)


def train_autoencoder(model, n_steps=5000, batch_size=256, sparsity=0.1, lr=1e-3):
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

    return losses


# =============================================================================
# Linearity Metrics (from main notebook)
# =============================================================================

def measure_encoding_linearity(model, n_samples=1000, sparsity=0.1):
    """Measure how linear the learned encoding is."""
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


# =============================================================================
# NEW: Feature Interference Metrics
# =============================================================================

def compute_feature_directions(model, n_samples=500):
    """
    Compute the "direction" each input feature takes in latent space.
    For a linear encoder, this would be the i-th column of the weight matrix.
    For a nonlinear encoder, we estimate it from data.

    Returns:
    - feature_directions: (n, m) tensor where row i is the average direction
      feature i takes when active
    """
    model.eval()
    n, m = model.n, model.m

    with torch.no_grad():
        # Method: For each feature, generate samples where only that feature varies
        # and measure the resulting change in latent space
        feature_directions = torch.zeros(n, m, device=device)

        for feat_idx in range(n):
            # Create samples with only feature i active at varying magnitudes
            magnitudes = torch.linspace(0.1, 1.0, n_samples, device=device)
            x = torch.zeros(n_samples, n, device=device)
            x[:, feat_idx] = magnitudes

            z = model.encode(x)

            # Linear regression to find direction: z = c + direction * magnitude
            # direction = cov(z, mag) / var(mag)
            mag_centered = magnitudes - magnitudes.mean()
            z_centered = z - z.mean(dim=0, keepdim=True)

            direction = (z_centered.T @ mag_centered) / (mag_centered @ mag_centered)
            feature_directions[feat_idx] = direction

    return feature_directions


def compute_linear_feature_directions(model):
    """
    Compute what feature directions WOULD be if the encoder were linear.
    This is the product of all encoder layer weights.
    """
    # Extract the linear transformation by composing weight matrices
    # For encoder: W1 * act * W2 * act * ... * Wfinal
    # Linear approximation: just multiply weights (ignoring activations)

    linear_weights = None
    for layer in model.encoder:
        if isinstance(layer, nn.Linear):
            W = layer.weight.data  # (out, in)
            if linear_weights is None:
                linear_weights = W
            else:
                linear_weights = W @ linear_weights

    # linear_weights is (m, n), transpose to get (n, m) feature directions
    return linear_weights.T


def compute_interference_matrix(directions):
    """
    Compute pairwise interference between feature directions.
    Interference = |cos(theta)| between direction vectors.
    High interference means features point in similar directions.

    Returns:
    - interference_matrix: (n, n) matrix of pairwise interferences
    """
    # Normalize directions
    norms = directions.norm(dim=1, keepdim=True)
    normalized = directions / (norms + 1e-8)

    # Compute cosine similarities
    cos_sim = normalized @ normalized.T

    # Interference is absolute cosine similarity (parallel or anti-parallel both bad)
    interference = cos_sim.abs()

    return interference


def compute_superposition_metrics(model, n_samples=1000, sparsity=0.1):
    """
    Measure how well the network achieves superposition.

    Superposition = storing more than m features in m dimensions by exploiting
    sparsity. Metrics:
    - effective_features: How many features can be independently decoded?
    - interference_score: Average pairwise interference
    - linear_interference: What interference WOULD be with linear encoding
    """
    model.eval()
    n, m = model.n, model.m

    # Get learned and linear feature directions
    learned_directions = compute_feature_directions(model, n_samples=200)
    linear_directions = compute_linear_feature_directions(model)

    # Compute interference matrices
    learned_interference = compute_interference_matrix(learned_directions)
    linear_interference = compute_interference_matrix(linear_directions)

    # Average off-diagonal interference (exclude self-interference)
    mask = ~torch.eye(n, dtype=bool, device=device)
    avg_learned_interference = learned_interference[mask].mean().item()
    avg_linear_interference = linear_interference[mask].mean().item()

    # Effective features: count features with low interference with others
    # A feature is "effective" if its max interference with any other feature is below threshold
    threshold = 0.5
    max_interference_per_feature = learned_interference.clone()
    max_interference_per_feature.fill_diagonal_(0)
    max_interference_per_feature = max_interference_per_feature.max(dim=1).values
    effective_features = (max_interference_per_feature < threshold).sum().item()

    # Interference reduction: how much did nonlinearity help reduce interference?
    interference_reduction = (avg_linear_interference - avg_learned_interference) / (avg_linear_interference + 1e-8)

    # Feature recovery: For each feature, how well can we decode it?
    with torch.no_grad():
        feature_recovery_scores = []
        for feat_idx in range(n):
            # Generate samples with only this feature active
            x = torch.zeros(100, n, device=device)
            x[:, feat_idx] = torch.rand(100, device=device)

            x_recon, z = model(x)

            # Recovery score: correlation between input and reconstructed feature
            corr = torch.corrcoef(torch.stack([x[:, feat_idx], x_recon[:, feat_idx]]))[0, 1]
            feature_recovery_scores.append(corr.item() if not torch.isnan(corr) else 0)

        avg_feature_recovery = np.mean(feature_recovery_scores)

    return {
        'avg_learned_interference': avg_learned_interference,
        'avg_linear_interference': avg_linear_interference,
        'interference_reduction': interference_reduction,
        'effective_features': effective_features,
        'effective_features_ratio': effective_features / n,
        'avg_feature_recovery': avg_feature_recovery,
        'learned_interference_matrix': learned_interference.cpu().numpy(),
        'linear_interference_matrix': linear_interference.cpu().numpy(),
    }


def compute_pairwise_nonlinearity(model, n_pairs=50, n_samples=100):
    """
    Measure nonlinearity of encoding for pairs of features.

    For each pair (i, j), check if encoding(x_i + x_j) ≈ encoding(x_i) + encoding(x_j)
    Deviation from additivity = nonlinear interaction
    """
    model.eval()
    n = model.n

    with torch.no_grad():
        pair_nonlinearities = []

        # Sample random pairs
        pairs = []
        for _ in range(n_pairs):
            i, j = np.random.choice(n, 2, replace=False)
            pairs.append((i, j))

        for i, j in pairs:
            # Generate samples with features i and j independently
            mags = torch.rand(n_samples, device=device) * 0.8 + 0.1  # [0.1, 0.9]

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

            # Additivity check: z_both ≈ (z_i - z_zero) + (z_j - z_zero) + z_zero
            #                          = z_i + z_j - z_zero
            z_expected = z_i + z_j - z_zero

            # Nonlinearity = relative deviation from additivity
            deviation = (z_both - z_expected).norm(dim=1)
            expected_norm = z_expected.norm(dim=1)
            relative_deviation = (deviation / (expected_norm + 1e-8)).mean().item()

            pair_nonlinearities.append(relative_deviation)

    return {
        'avg_pairwise_nonlinearity': np.mean(pair_nonlinearities),
        'max_pairwise_nonlinearity': np.max(pair_nonlinearities),
        'std_pairwise_nonlinearity': np.std(pair_nonlinearities),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_single_experiment(n, m, l, sparsity=0.1, n_steps=5000):
    """Run a single experiment with all metrics."""
    print(f"  Training n={n}, m={m}, l={l}...")

    model = Autoencoder(n, m, l).to(device)
    losses = train_autoencoder(model, n_steps=n_steps, sparsity=sparsity)

    # Basic metrics
    linearity = measure_encoding_linearity(model, sparsity=sparsity)

    # New superposition/interference metrics
    superposition = compute_superposition_metrics(model, sparsity=sparsity)

    # Pairwise nonlinearity
    pairwise = compute_pairwise_nonlinearity(model)

    result = {
        'n': n, 'm': m, 'l': l, 'sparsity': sparsity,
        'final_loss': float(np.mean(losses[-100:])),
        'compression_ratio': n / m,
        **linearity,
        'avg_learned_interference': superposition['avg_learned_interference'],
        'avg_linear_interference': superposition['avg_linear_interference'],
        'interference_reduction': superposition['interference_reduction'],
        'effective_features': superposition['effective_features'],
        'effective_features_ratio': superposition['effective_features_ratio'],
        'avg_feature_recovery': superposition['avg_feature_recovery'],
        **pairwise,
    }

    return result, superposition


def run_experiment_sweep():
    """Run the main experiment sweep."""
    print("="*60)
    print("Experiment 1: Feature Interference and Superposition")
    print("="*60)

    # Configuration
    n_values = [16, 32, 64, 128]
    m_values = [4, 8, 16, 32]
    l_values = [1, 2, 3]
    sparsity = 0.1
    n_steps = 4000

    results = []
    interference_matrices = {}  # Store for visualization

    total = sum(1 for n in n_values for m in m_values for l in l_values if m < n)
    print(f"Running {total} configurations...")

    pbar = tqdm(total=total)
    for n in n_values:
        for m in m_values:
            if m >= n:
                continue
            for l in l_values:
                result, superposition = run_single_experiment(n, m, l, sparsity, n_steps)
                results.append(result)

                # Store interference matrices for select configs
                key = f"n{n}_m{m}_l{l}"
                interference_matrices[key] = {
                    'learned': superposition['learned_interference_matrix'].tolist(),
                    'linear': superposition['linear_interference_matrix'].tolist(),
                }

                pbar.update(1)
                pbar.set_postfix({
                    'nonlin_gain': f"{result['nonlinear_gain']:.3f}",
                    'interf_red': f"{result['interference_reduction']:.3f}"
                })

    pbar.close()
    return results, interference_matrices


def analyze_results(results):
    """Analyze and visualize results."""
    import pandas as pd

    df = pd.DataFrame(results)

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Nonlinear gain vs interference reduction
    ax = axes[0, 0]
    scatter = ax.scatter(df['interference_reduction'], df['nonlinear_gain'],
                        c=df['compression_ratio'], cmap='viridis', s=60, alpha=0.7)
    ax.set_xlabel('Interference Reduction')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Nonlinear Gain vs Interference Reduction')
    plt.colorbar(scatter, ax=ax, label='Compression (n/m)')

    # 2. Nonlinear gain vs pairwise nonlinearity
    ax = axes[0, 1]
    scatter = ax.scatter(df['avg_pairwise_nonlinearity'], df['nonlinear_gain'],
                        c=df['l'], cmap='plasma', s=60, alpha=0.7)
    ax.set_xlabel('Avg Pairwise Nonlinearity')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Encoding Nonlinearity Measures')
    plt.colorbar(scatter, ax=ax, label='Depth (l)')

    # 3. Effective features ratio vs compression
    ax = axes[0, 2]
    for l in df['l'].unique():
        subset = df[df['l'] == l]
        ax.scatter(subset['compression_ratio'], subset['effective_features_ratio'],
                  label=f'l={l}', s=60, alpha=0.7)
    ax.set_xlabel('Compression Ratio (n/m)')
    ax.set_ylabel('Effective Features Ratio')
    ax.set_title('Superposition: Effective Features')
    ax.legend()

    # 4. Learned vs Linear interference
    ax = axes[1, 0]
    ax.scatter(df['avg_linear_interference'], df['avg_learned_interference'],
              c=df['nonlinear_gain'], cmap='RdYlGn_r', s=60, alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Linear Interference')
    ax.set_ylabel('Learned Interference')
    ax.set_title('Interference: Linear vs Learned')
    ax.legend()

    # 5. Feature recovery vs nonlinear gain
    ax = axes[1, 1]
    scatter = ax.scatter(df['avg_feature_recovery'], df['nonlinear_gain'],
                        c=df['compression_ratio'], cmap='viridis', s=60, alpha=0.7)
    ax.set_xlabel('Avg Feature Recovery')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Feature Recovery vs Nonlinear Gain')
    plt.colorbar(scatter, ax=ax, label='Compression (n/m)')

    # 6. Compression ratio vs nonlinear gain (colored by interference reduction)
    ax = axes[1, 2]
    scatter = ax.scatter(df['compression_ratio'], df['nonlinear_gain'],
                        c=df['interference_reduction'], cmap='coolwarm', s=60, alpha=0.7)
    ax.set_xlabel('Compression Ratio (n/m)')
    ax.set_ylabel('Nonlinear Gain')
    ax.set_title('Phase Diagram (colored by interference reduction)')
    plt.colorbar(scatter, ax=ax, label='Interference Reduction')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'main_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Compute correlations
    correlations = {
        'nonlinear_gain_vs_interference_reduction': df['nonlinear_gain'].corr(df['interference_reduction']),
        'nonlinear_gain_vs_pairwise_nonlinearity': df['nonlinear_gain'].corr(df['avg_pairwise_nonlinearity']),
        'nonlinear_gain_vs_compression': df['nonlinear_gain'].corr(df['compression_ratio']),
        'interference_reduction_vs_compression': df['interference_reduction'].corr(df['compression_ratio']),
        'effective_features_vs_nonlinear_gain': df['effective_features_ratio'].corr(df['nonlinear_gain']),
    }

    return df, correlations


def plot_interference_comparison(interference_matrices, n_select=4):
    """Plot example interference matrices comparing linear vs learned."""
    keys = list(interference_matrices.keys())[:n_select]

    fig, axes = plt.subplots(n_select, 2, figsize=(10, 4*n_select))

    for i, key in enumerate(keys):
        data = interference_matrices[key]
        linear = np.array(data['linear'])
        learned = np.array(data['learned'])

        # Linear
        ax = axes[i, 0]
        im = ax.imshow(linear, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'{key}\nLinear Interference')
        ax.set_xlabel('Feature j')
        ax.set_ylabel('Feature i')
        plt.colorbar(im, ax=ax)

        # Learned
        ax = axes[i, 1]
        im = ax.imshow(learned, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'{key}\nLearned Interference')
        ax.set_xlabel('Feature j')
        ax.set_ylabel('Feature i')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'interference_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Run experiment
    results, interference_matrices = run_experiment_sweep()

    # Save raw results
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Analyze
    df, correlations = analyze_results(results)
    df.to_csv(RESULTS_DIR / 'results.csv', index=False)

    # Plot interference matrices
    plot_interference_comparison(interference_matrices)

    # Save correlations
    with open(RESULTS_DIR / 'correlations.json', 'w') as f:
        json.dump(correlations, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nKey Correlations:")
    for name, corr in correlations.items():
        print(f"  {name}: {corr:.3f}")

    print("\nKey Statistics:")
    print(f"  Avg nonlinear gain: {df['nonlinear_gain'].mean():.4f} +/- {df['nonlinear_gain'].std():.4f}")
    print(f"  Avg interference reduction: {df['interference_reduction'].mean():.4f} +/- {df['interference_reduction'].std():.4f}")
    print(f"  Avg pairwise nonlinearity: {df['avg_pairwise_nonlinearity'].mean():.4f}")

    # Find interesting regime
    high_nonlin = df[df['nonlinear_gain'] > df['nonlinear_gain'].quantile(0.75)]
    print(f"\nHigh nonlinear gain regime (top 25%):")
    print(f"  Avg compression ratio: {high_nonlin['compression_ratio'].mean():.1f}")
    print(f"  Avg depth: {high_nonlin['l'].mean():.1f}")
    print(f"  Avg interference reduction: {high_nonlin['interference_reduction'].mean():.3f}")

    return results, correlations


if __name__ == '__main__':
    results, correlations = main()
    print("\nExperiment complete! Results saved to:", RESULTS_DIR)
