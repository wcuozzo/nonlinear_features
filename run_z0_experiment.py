"""
z(0) Constraint Experiment

Tests whether encoder biases (z(0) location) matter for autoencoder performance.
Addresses TODO: "Are the bias terms (where z(0) ends up) doing anything?"

This script runs the key experiments from z0_constraint_exploration.ipynb
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from scipy import stats
import json
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class AutoencoderNoBias(nn.Module):
    """
    Autoencoder with configurable encoder biases.

    When encoder_bias=False, z(0) = 0 is guaranteed:
    - ReLU(W @ 0) = ReLU(0) = 0
    - So the zero input maps to the origin in z-space.

    When encoder_bias=True (default), z(0) can be anywhere.
    """
    def __init__(self, n: int, m: int, l: int, encoder_bias: bool = True, activation=nn.ReLU):
        super().__init__()
        self.n = n
        self.m = m
        self.l = l
        self.encoder_bias = encoder_bias

        # Build encoder: (l-1) layers of [Linear(n->n) + ReLU], then Linear(n->m)
        encoder_layers = []
        for i in range(l - 1):
            encoder_layers.append(nn.Linear(n, n, bias=encoder_bias))
            encoder_layers.append(activation())
        encoder_layers.append(nn.Linear(n, m, bias=encoder_bias))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder always has biases (doesn't affect z(0))
        decoder_layers = []
        decoder_layers.append(nn.Linear(m, n))
        for i in range(l - 1):
            decoder_layers.append(activation())
            decoder_layers.append(nn.Linear(n, n))
        decoder_layers.append(activation())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def generate_sparse_data(n_samples: int, n_features: int, S: float = 0.95) -> torch.Tensor:
    """Generate sparse data where each feature is active with probability (1-S)."""
    mask = (torch.rand(n_samples, n_features) > S).float()
    values = torch.rand(n_samples, n_features)
    return (mask * values).to(device)


def train_model(model, n_steps=20000, batch_size=1024, S=0.9, lr=1e-3, weight_decay=1e-2, verbose=False):
    """Train the autoencoder with cosine LR schedule."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    iterator = tqdm(range(n_steps), desc="Training", disable=not verbose)

    for step in iterator:
        x = generate_sparse_data(batch_size, model.n, S)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if verbose and step % 2000 == 0:
            iterator.set_postfix({'loss': f'{loss.item():.6f}'})

    return losses


def evaluate_on_test_set(model, test_x):
    """Evaluate model on fixed test set."""
    model.eval()
    with torch.no_grad():
        x_recon, _ = model(test_x)
        loss = nn.functional.mse_loss(x_recon, test_x).item()
    return loss


def run_comparison(n, m, l, S, n_seeds=20, n_steps=20000):
    """Run paired comparison: with vs without encoder biases."""
    print(f"\n{'='*60}")
    print(f"Configuration: n={n}, m={m}, l={l}, S={S}")
    print(f"Running {n_seeds} seeds...")
    print('='*60)

    # Generate fixed test set
    torch.manual_seed(99999)
    test_x = generate_sparse_data(50000, n, S)

    losses_with = []
    losses_without = []
    z0_locations = []

    for seed in tqdm(range(n_seeds), desc="Seeds"):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # With encoder bias
        model_with = AutoencoderNoBias(n, m, l, encoder_bias=True).to(device)
        train_model(model_with, n_steps=n_steps, S=S, verbose=False)
        loss_with = evaluate_on_test_set(model_with, test_x)
        losses_with.append(loss_with)

        # Get z(0) location
        with torch.no_grad():
            z0 = model_with.encode(torch.zeros(1, n).to(device)).cpu().numpy()[0]
            z0_locations.append(z0)

        # Without encoder bias
        torch.manual_seed(seed)
        np.random.seed(seed)
        model_without = AutoencoderNoBias(n, m, l, encoder_bias=False).to(device)
        train_model(model_without, n_steps=n_steps, S=S, verbose=False)
        loss_without = evaluate_on_test_set(model_without, test_x)
        losses_without.append(loss_without)

    # Statistical analysis
    losses_with = np.array(losses_with)
    losses_without = np.array(losses_without)
    diffs = losses_without - losses_with

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(losses_without, losses_with)

    # Cohen's d (effect size)
    cohens_d = np.mean(diffs) / np.std(diffs, ddof=1)

    # Best-of-k comparisons
    best_with = np.min(losses_with)
    best_without = np.min(losses_without)

    results = {
        'n': n, 'm': m, 'l': l, 'S': S,
        'n_seeds': n_seeds,
        'mean_with': float(np.mean(losses_with)),
        'std_with': float(np.std(losses_with)),
        'mean_without': float(np.mean(losses_without)),
        'std_without': float(np.std(losses_without)),
        'mean_diff': float(np.mean(diffs)),
        'diff_pct': float((np.mean(losses_without) - np.mean(losses_with)) / np.mean(losses_with) * 100),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'best_with': float(best_with),
        'best_without': float(best_without),
        'best_diff_pct': float((best_without - best_with) / best_with * 100),
        'z0_mean_norm': float(np.mean([np.linalg.norm(z) for z in z0_locations])),
        'z0_max_norm': float(np.max([np.linalg.norm(z) for z in z0_locations])),
    }

    # Effect size interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        results['effect'] = 'negligible'
    elif abs_d < 0.5:
        results['effect'] = 'small'
    elif abs_d < 0.8:
        results['effect'] = 'medium'
    else:
        results['effect'] = 'large'

    results['significant'] = bool(p_value < 0.05)

    # Print summary
    print(f"\nResults:")
    print(f"  With bias:    {results['mean_with']:.6f} +/- {results['std_with']:.6f}")
    print(f"  Without bias: {results['mean_without']:.6f} +/- {results['std_without']:.6f}")
    print(f"  Difference:   {results['mean_diff']:.6f} ({results['diff_pct']:+.2f}%)")
    print(f"  p-value:      {results['p_value']:.4f}")
    print(f"  Cohen's d:    {results['cohens_d']:.3f} ({results['effect']})")
    print(f"  Best with:    {results['best_with']:.6f}")
    print(f"  Best without: {results['best_without']:.6f} ({results['best_diff_pct']:+.2f}%)")
    print(f"  z(0) norm:    mean={results['z0_mean_norm']:.4f}, max={results['z0_max_norm']:.4f}")

    return results


def main():
    """Run comprehensive z(0) constraint experiment."""
    print("="*70)
    print("Z(0) CONSTRAINT EXPERIMENT")
    print("Testing whether encoder biases matter for autoencoder performance")
    print("="*70)

    # Configuration grid
    configs = [
        # (n, m, l, S)
        (5, 2, 2, 0.9),    # Small, high compression
        (5, 2, 4, 0.9),    # Small, deeper
        (10, 3, 2, 0.9),   # Medium
        (10, 5, 2, 0.9),   # Medium, less compression
        (10, 3, 4, 0.9),   # Medium, deeper
        (5, 2, 2, 0.95),   # Higher sparsity
        (10, 3, 2, 0.95),  # Medium, higher sparsity
    ]

    all_results = []

    for n, m, l, S in configs:
        result = run_comparison(n, m, l, S, n_seeds=15, n_steps=15000)
        all_results.append(result)

    # Summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)

    n_negligible = sum(1 for r in all_results if r['effect'] == 'negligible')
    n_small = sum(1 for r in all_results if r['effect'] == 'small')
    n_significant = sum(1 for r in all_results if r['significant'])
    avg_diff_pct = np.mean([r['diff_pct'] for r in all_results])
    avg_best_diff_pct = np.mean([r['best_diff_pct'] for r in all_results])

    print(f"\nConfigurations tested: {len(all_results)}")
    print(f"Effect sizes: {n_negligible} negligible, {n_small} small")
    print(f"Statistically significant: {n_significant}")
    print(f"Average difference: {avg_diff_pct:+.2f}%")
    print(f"Average best-model difference: {avg_best_diff_pct:+.2f}%")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if n_negligible + n_small == len(all_results) and abs(avg_diff_pct) < 5:
        conclusion = """
The z(0) degree of freedom (encoder biases) has NEGLIGIBLE practical impact:
- Effect sizes are consistently small or negligible
- Average performance difference is less than 5%
- Best-model comparisons show similar results

INTERPRETATION: The bias terms allow z(0) to be non-zero, but this extra
degree of freedom is not exploited for better representations. The model
achieves similar performance whether z(0) is free or constrained to origin.

THEORETICAL IMPLICATION: The nonlinearity in these autoencoders does NOT
fundamentally depend on shifting the origin in latent space. The benefits
of depth come from input-dependent transformations, not global offsets.
"""
    else:
        conclusion = f"""
Results are mixed across configurations:
- {n_negligible} negligible, {n_small} small effects
- {n_significant} statistically significant differences
- Average difference: {avg_diff_pct:+.2f}%

Further investigation needed to understand when biases matter.
"""

    print(conclusion)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'z0_constraint',
        'timestamp': timestamp,
        'device': str(device),
        'configs': all_results,
        'summary': {
            'n_configs': len(all_results),
            'n_negligible': n_negligible,
            'n_small': n_small,
            'n_significant': n_significant,
            'avg_diff_pct': avg_diff_pct,
            'avg_best_diff_pct': avg_best_diff_pct,
        },
        'conclusion': conclusion.strip()
    }

    output_file = f'/Users/williamcuozzo/Desktop/ai_projects/nonlinear_features/z0_experiment_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == "__main__":
    results = main()
