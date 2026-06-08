"""
Core functions for nonlinear feature encoding experiments.
Shared between main notebook and sanity checks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Autoencoder(nn.Module):
    """
    Autoencoder with configurable depth.

    l = number of linear layers in encoder:
        l=1: x → Linear(n→m) → z                              [simplest, paper's setup]
        l=2: x → Linear(n→n) → ReLU → Linear(n→m) → z
        l=3: x → Linear(n→n) → ReLU → Linear(n→n) → ReLU → Linear(n→m) → z

    Decoder mirrors encoder structure, with ReLU on final output.
    For l=1, uses tied weights (W for encode, W.T for decode) matching the paper.
    """
    def __init__(self, n: int, m: int, l: int = 1, tied_weights: bool = True, activation=nn.ReLU):
        """
        Args:
            n: feature dimension (input/output width)
            m: bottleneck dimension
            l: number of linear layers in encoder (minimum 1)
            tied_weights: if True and l=1, use tied weights (paper setup)
            activation: activation function class (default: ReLU)
        """
        super().__init__()
        assert l >= 1, "l must be at least 1 (need at least one linear layer)"
        self.n = n
        self.m = m
        self.l = l
        self.tied_weights = tied_weights and (l == 1)  # Only tie weights for l=1

        if self.tied_weights:
            # l=1 with tied weights: matches Toy Models paper exactly
            # Encoder: x @ W.T, Decoder: ReLU(z @ W + b)
            self.encoder = nn.Linear(n, m, bias=False)
            self.decoder_bias = nn.Parameter(torch.zeros(n))
        else:
            # Build encoder: (l-1) layers of [Linear(n→n) + ReLU], then Linear(n→m)
            encoder_layers = []
            for i in range(l - 1):
                encoder_layers.append(nn.Linear(n, n))
                encoder_layers.append(activation())
            encoder_layers.append(nn.Linear(n, m))  # final compression, no activation
            self.encoder = nn.Sequential(*encoder_layers)

            # Build decoder: Linear(m→n), then (l-1) layers of [ReLU + Linear(n→n)], then ReLU
            # Final ReLU ensures non-negative output (matching sparse non-negative features)
            decoder_layers = []
            decoder_layers.append(nn.Linear(m, n))
            for i in range(l - 1):
                decoder_layers.append(activation())
                decoder_layers.append(nn.Linear(n, n))
            decoder_layers.append(activation())  # Final ReLU for non-negative output
            self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        if self.tied_weights:
            # Use transposed encoder weight
            return nn.functional.relu(z @ self.encoder.weight + self.decoder_bias)
        else:
            return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# Alias for backwards compatibility
ToyModelAutoencoder = lambda n, m: Autoencoder(n, m, l=1, tied_weights=True)


def generate_sparse_data(n_samples: int, n_features: int, S: float = 0.95) -> torch.Tensor:
    """
    Generate sparse data where each feature is independently active with probability (1 - S).
    Active features have random positive values.

    Args:
        n_samples: number of samples
        n_features: number of features
        S: sparsity = probability of being ZERO (matches Toy Models paper notation).
           S=0.95 means 5% of features are active on average.
    """
    mask = (torch.rand(n_samples, n_features) > S).float()
    values = torch.rand(n_samples, n_features)
    return (mask * values).to(device)


def generate_correlated_features(n_samples: int, n_features: int, n_true_features: int, S: float = 0.95) -> torch.Tensor:
    """
    Generate data from a smaller set of true underlying features.
    The n_features are linear combinations of n_true_features sparse sources.

    Args:
        n_samples: number of samples
        n_features: number of observed features
        n_true_features: number of underlying sparse sources
        S: sparsity = probability of being ZERO (matches Toy Models paper notation)
    """
    sources = generate_sparse_data(n_samples, n_true_features, S)
    mixing = torch.randn(n_true_features, n_features, device=device)
    mixing = mixing / mixing.norm(dim=0, keepdim=True)
    return sources @ mixing


def get_feature_importance(n: int, decay: float = 0.7, device=None) -> torch.Tensor:
    """
    Generate importance weights I_i = decay^i (paper uses decay=0.7).

    Args:
        n: number of features
        decay: decay factor (0.7 in Toy Models paper)
        device: torch device

    Returns:
        Tensor of shape (n,) with importance weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.tensor([decay ** i for i in range(n)], device=device, dtype=torch.float32)


def train_autoencoder(
    model: Autoencoder,
    n_steps: int = 10000,
    batch_size: int = 1024,
    S: float = 0.95,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    importance: torch.Tensor = None,
    loss_threshold: float = None,
    verbose: bool = True
) -> List[float]:
    """
    Train the autoencoder on sparse data.

    Args:
        model: Autoencoder to train
        n_steps: number of training steps
        batch_size: batch size
        S: sparsity = probability of being ZERO (matches Toy Models paper notation)
        lr: learning rate
        weight_decay: L2 regularization (Adam weight_decay parameter)
        importance: optional feature importance weights I_i (shape: n)
        loss_threshold: if set, stop training early when loss drops below this value
        verbose: print progress

    Returns:
        List of loss values per step
    """
    # Paper uses AdamW (decoupled weight decay), not Adam with L2 regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    iterator = tqdm(range(n_steps)) if verbose else range(n_steps)
    for step in iterator:
        x = generate_sparse_data(batch_size, model.n, S)

        optimizer.zero_grad()
        x_recon, z = model(x)

        if importance is not None:
            # Weighted MSE loss: mean over batch of (importance * (x - x_recon)^2)
            loss = (importance * (x - x_recon) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(x_recon, x)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and step % 1000 == 0:
            iterator.set_postfix({'loss': f'{loss.item():.6f}'})

        # Early stopping
        if loss_threshold is not None and loss.item() < loss_threshold:
            if verbose:
                print(f"Early stop at step {step}, loss={loss.item():.6f}")
            break

    return losses


def measure_encoding_linearity(model: Autoencoder, n_samples: int = 1000, S: float = 0.95) -> Dict[str, float]:
    """Measure how linear the learned encoding is.

    Args:
        model: trained autoencoder
        n_samples: number of samples for evaluation
        S: sparsity = probability of being ZERO (matches Toy Models paper notation)
    """
    model.eval()

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, S)
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


def compute_feature_geometry(model) -> Dict[str, float]:
    """
    Compute geometry of feature vectors in the bottleneck.

    Returns:
        Dict with:
        - min_norm: minimum feature vector norm
        - max_norm: maximum feature vector norm
        - min_angle: minimum pairwise angle in degrees
        - norms: list of all norms
        - angles: list of all pairwise angles
    """
    # Get encoder weights
    encoder_weights = None
    for name, param in model.encoder.named_parameters():
        if 'weight' in name:
            encoder_weights = param.detach().cpu().numpy()
            break

    if encoder_weights is None:
        return {'min_norm': 0, 'max_norm': 0, 'min_angle': 0, 'norms': [], 'angles': []}

    n_features = encoder_weights.shape[1]

    # Compute norms
    norms = [np.linalg.norm(encoder_weights[:, i]) for i in range(n_features)]

    # Compute all pairwise angles
    angles = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            v1, v2 = encoder_weights[:, i], encoder_weights[:, j]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                angles.append(angle)

    return {
        'min_norm': min(norms) if norms else 0,
        'max_norm': max(norms) if norms else 0,
        'min_angle': min(angles) if angles else 0,
        'norms': norms,
        'angles': angles,
    }


def compute_jacobian_stats(model: Autoencoder, n_samples: int = 100, S: float = 0.95) -> Dict[str, float]:
    """Compute statistics about the encoder's Jacobian to measure effective nonlinearity.

    Args:
        model: trained autoencoder
        n_samples: number of samples for evaluation
        S: sparsity = probability of being ZERO (matches Toy Models paper notation)
    """
    model.eval()

    x_data = generate_sparse_data(n_samples, model.n, S)

    jacobians = []
    for i in range(min(n_samples, 50)):
        xi = x_data[i:i+1].clone().detach().requires_grad_(True)
        zi = model.encode(xi)

        jac = torch.zeros(model.m, model.n, device=device)
        for j in range(model.m):
            if xi.grad is not None:
                xi.grad.zero_()
            zi[0, j].backward(retain_graph=True)
            jac[j] = xi.grad[0]
        jacobians.append(jac)

    jacobians = torch.stack(jacobians)

    jac_mean = jacobians.mean(dim=0)
    jac_var = ((jacobians - jac_mean) ** 2).mean().item()

    svd = torch.linalg.svdvals(jac_mean)
    svd_normalized = svd / svd.sum()
    effective_rank = torch.exp(-torch.sum(svd_normalized * torch.log(svd_normalized + 1e-10))).item()

    return {
        'jacobian_variance': jac_var,
        'effective_rank': effective_rank,
    }


def run_experiment(
    n: int, m: int, l: int = 1,
    S: float = 0.95,
    n_steps: int = 10000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    importance_decay: float = None,
    tied_weights: bool = True,
    loss_threshold: float = None,
    verbose: bool = True
) -> Dict:
    """
    Run a single experiment with given parameters.

    Args:
        n: input/output dimension (number of features)
        m: bottleneck dimension
        l: number of linear layers in encoder (l=1 is simplest, matches paper)
        S: sparsity = probability of being ZERO (matches Toy Models paper notation).
           S=0.95 means 5% of features are active on average.
        n_steps: training steps
        batch_size: batch size (paper uses 1024)
        lr: learning rate
        weight_decay: L2 regularization (default 1e-2 matches paper)
        importance_decay: if set, use I_i = importance_decay^i weighting (paper uses 0.7)
        tied_weights: if True and l=1, use tied encoder/decoder weights (matches paper)
        loss_threshold: if set, stop training early when loss drops below this value
        verbose: print progress

    Returns:
        Dict with model, metrics, and training info
    """
    model = Autoencoder(n, m, l, tied_weights=tied_weights).to(device)

    # Compute importance weights if specified
    importance = None
    if importance_decay is not None:
        importance = get_feature_importance(n, importance_decay, device=device)

    if verbose:
        print(f"\nExperiment: n={n}, m={m}, l={l}, S={S}")
        if importance_decay is not None:
            print(f"Using importance weighting: I_i = {importance_decay}^i")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    losses = train_autoencoder(
        model, n_steps=n_steps, batch_size=batch_size, S=S,
        lr=lr, weight_decay=weight_decay, importance=importance,
        loss_threshold=loss_threshold,
        verbose=verbose
    )

    linearity_metrics = measure_encoding_linearity(model, S=S)
    jacobian_metrics = compute_jacobian_stats(model, S=S)

    results = {
        'n': n, 'm': m, 'l': l, 'S': S,
        'importance_decay': importance_decay,
        'final_loss': np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses),
        **linearity_metrics,
        **jacobian_metrics,
        'model': model,
        'losses': losses,
    }

    if verbose:
        print(f"Results: linearity={linearity_metrics['linearity_score']:.3f}, "
              f"nonlinear_gain={linearity_metrics['nonlinear_gain']:.3f}, "
              f"jac_var={jacobian_metrics['jacobian_variance']:.6f}")

    return results


def run_experiment_multi_seed(
    n: int, m: int, l: int = 1,
    n_seeds: int = 10,
    S: float = 0.95,
    n_steps: int = 10000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    importance_decay: float = None,
    tied_weights: bool = True,
    loss_threshold: float = None,
    min_norm_threshold: float = None,
    max_norm_threshold: float = None,
    min_angle_threshold: float = None,
    verbose: bool = True
) -> Dict:
    """
    Run multiple seeds and return the best (lowest loss) result.

    The Toy Models paper runs 200+ seeds due to optimization sensitivity.
    This function runs n_seeds experiments and keeps the best one.

    Args:
        n: input/output dimension
        m: bottleneck dimension
        l: number of linear layers in encoder (l=1 is simplest, matches paper)
        n_seeds: number of random seeds to try
        S: sparsity = probability of being ZERO (matches Toy Models paper notation).
           S=0.95 means 5% of features are active on average.
        n_steps: training steps per seed
        batch_size: batch size (paper uses 1024)
        lr: learning rate
        weight_decay: L2 regularization (default 1e-2 matches paper)
        importance_decay: if set, use I_i = importance_decay^i weighting
        tied_weights: if True and l=1, use tied encoder/decoder weights (matches paper)
        loss_threshold: if set, stop searching seeds once best loss is below this.
        min_norm_threshold: if set, require all feature norms >= this value for early stop.
        max_norm_threshold: if set, require all feature norms <= this value for early stop.
        min_angle_threshold: if set, require min pairwise angle >= this (degrees) for early stop.
        verbose: print progress

    Returns:
        Dict with best model, all losses, and seed info
    """
    best_result = None
    best_loss = float('inf')
    all_final_losses = []

    if verbose:
        print(f"\nMulti-seed experiment: n={n}, m={m}, l={l}, S={S}, n_seeds={n_seeds}")
        if importance_decay is not None:
            print(f"Using importance weighting: I_i = {importance_decay}^i")
        stopping_conds = []
        if loss_threshold is not None:
            stopping_conds.append(f"loss<{loss_threshold}")
        if min_norm_threshold is not None:
            stopping_conds.append(f"min_norm>={min_norm_threshold}")
        if max_norm_threshold is not None:
            stopping_conds.append(f"max_norm<={max_norm_threshold}")
        if min_angle_threshold is not None:
            stopping_conds.append(f"min_angle>={min_angle_threshold}°")
        if stopping_conds:
            print(f"Early stopping: {' AND '.join(stopping_conds)}")

    iterator = tqdm(range(n_seeds), desc="Seeds") if verbose else range(n_seeds)

    for seed in iterator:
        torch.manual_seed(seed)
        np.random.seed(seed)

        result = run_experiment(
            n=n, m=m, l=l,
            S=S,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            importance_decay=importance_decay,
            tied_weights=tied_weights,
            loss_threshold=None,  # Don't stop training early - let it converge fully
            verbose=False
        )

        final_loss = result['final_loss']
        all_final_losses.append(final_loss)

        if final_loss < best_loss:
            best_loss = final_loss
            best_result = result
            best_seed = seed

        # Compute geometry for progress display
        geom = compute_feature_geometry(result['model'])

        if verbose:
            iterator.set_postfix({
                'loss': f'{final_loss:.4f}',
                'min_norm': f'{geom["min_norm"]:.2f}',
                'min_angle': f'{geom["min_angle"]:.0f}°'
            })

        # Early stopping across seeds: check all conditions on CURRENT result
        should_stop = True
        stop_reasons = []

        if loss_threshold is not None:
            if final_loss < loss_threshold:
                stop_reasons.append(f"loss={final_loss:.6f}")
            else:
                should_stop = False

        # geom already computed above for progress display
        if min_norm_threshold is not None:
            if geom['min_norm'] >= min_norm_threshold:
                stop_reasons.append(f"min_norm={geom['min_norm']:.3f}")
            else:
                should_stop = False

        if max_norm_threshold is not None:
            if geom['max_norm'] <= max_norm_threshold:
                stop_reasons.append(f"max_norm={geom['max_norm']:.3f}")
            else:
                should_stop = False

        if min_angle_threshold is not None:
            if geom['min_angle'] >= min_angle_threshold:
                stop_reasons.append(f"min_angle={geom['min_angle']:.1f}°")
            else:
                should_stop = False

        # Only stop if at least one threshold was set and all conditions met
        if should_stop and stop_reasons:
            if verbose:
                print(f"\nFound good seed {seed}: {', '.join(stop_reasons)}")
            best_result = result
            best_loss = final_loss
            best_seed = seed
            break

    # Add multi-seed info to result
    best_result['best_seed'] = best_seed
    best_result['all_final_losses'] = all_final_losses
    best_result['n_seeds'] = n_seeds
    best_result['seeds_tried'] = len(all_final_losses)

    # Add geometry info
    geom = compute_feature_geometry(best_result['model'])
    best_result['min_norm'] = geom['min_norm']
    best_result['max_norm'] = geom['max_norm']
    best_result['min_angle'] = geom['min_angle']
    best_result['feature_norms'] = geom['norms']
    best_result['pairwise_angles'] = geom['angles']

    if verbose:
        print(f"\nBest seed: {best_seed}, loss: {best_loss:.6f}")
        print(f"Seeds tried: {len(all_final_losses)}/{n_seeds}")
        print(f"Loss range across seeds: [{min(all_final_losses):.6f}, {max(all_final_losses):.6f}]")
        print(f"Geometry: min_norm={geom['min_norm']:.3f}, min_angle={geom['min_angle']:.1f}°")
        print(f"Results: linearity={best_result['linearity_score']:.3f}, "
              f"nonlinear_gain={best_result['nonlinear_gain']:.3f}")

    return best_result


def run_phase_sweep(
    n_values: List[int],
    m_values: List[int],
    l_values: List[int],
    S: float = 0.95,
    n_steps: int = 3000,
    loss_threshold: float = None
) -> List[Dict]:
    """Sweep over parameter combinations to build phase diagram.

    Args:
        n_values: list of input dimensions to try
        m_values: list of bottleneck dimensions to try
        l_values: list of encoder layers to try (l=1 is linear encoder, l>=2 is nonlinear)
        S: sparsity = probability of being ZERO (matches Toy Models paper notation)
        n_steps: training steps per experiment
        loss_threshold: if set, stop training early when loss drops below this value
    """
    all_results = []

    total = len(n_values) * len(m_values) * len(l_values)
    pbar = tqdm(total=total, desc="Phase sweep")

    for n in n_values:
        for m in m_values:
            for l in l_values:
                if m > n:
                    pbar.update(1)
                    continue
                results = run_experiment(n, m, l, S=S, n_steps=n_steps,
                                         loss_threshold=loss_threshold, verbose=False)
                del results['model']
                del results['losses']
                all_results.append(results)
                pbar.update(1)

    pbar.close()
    return all_results
