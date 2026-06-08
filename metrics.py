"""
Comprehensive metrics for autoencoder nonlinear feature encoding experiments.

Organized into categories:
1. Loss-based metrics
2. Nonlinearity metrics
3. Latent space metrics
4. Robustness metrics
5. Training dynamics metrics

All functions take a trained model and return dicts of scalar metrics
(suitable for sweep DataFrames) unless noted otherwise.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from core import Autoencoder, generate_sparse_data, device


# =============================================================================
# 1. LOSS-BASED METRICS
# =============================================================================

def compute_loss_metrics(
    model: Autoencoder, S: float = 0.9, n_samples: int = 5000
) -> Dict[str, float]:
    """Test MSE loss, per-feature MSE, and generalization gap estimate.

    Uses fresh test data (not seen during training).
    """
    model.eval()
    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, S)
        x_recon, z = model(x)

        # Overall test MSE
        test_mse = nn.functional.mse_loss(x_recon, x).item()

        # Per-feature MSE
        per_feature_mse = ((x_recon - x) ** 2).mean(dim=0).cpu().numpy()

        # Per-feature MSE conditioned on feature being active
        active_mask = x > 0  # shape: (n_samples, n)
        active_mse_list = []
        inactive_mse_list = []
        for i in range(model.n):
            active = active_mask[:, i]
            if active.sum() > 0:
                active_mse_list.append(
                    ((x_recon[active, i] - x[active, i]) ** 2).mean().item()
                )
            else:
                active_mse_list.append(float('nan'))
            inactive = ~active
            if inactive.sum() > 0:
                inactive_mse_list.append(
                    ((x_recon[inactive, i] - x[inactive, i]) ** 2).mean().item()
                )
            else:
                inactive_mse_list.append(float('nan'))

    return {
        'test_mse': test_mse,
        'per_feature_mse': per_feature_mse,  # array
        'per_feature_mse_mean': float(np.nanmean(per_feature_mse)),
        'per_feature_mse_std': float(np.nanstd(per_feature_mse)),
        'per_feature_mse_max': float(np.nanmax(per_feature_mse)),
        'active_feature_mse_mean': float(np.nanmean(active_mse_list)),
        'inactive_feature_mse_mean': float(np.nanmean(inactive_mse_list)),
    }


# =============================================================================
# 2. NONLINEARITY METRICS
# =============================================================================

def compute_nonlinear_gain(
    model: Autoencoder, S: float = 0.9, n_samples: int = 2000
) -> Dict[str, float]:
    """Nonlinear gain: relative MSE improvement over best linear encoder approximation.

    Fits a linear encoder (via least squares) and compares reconstruction quality.
    Equivalent to core.measure_encoding_linearity but with clearer naming.
    """
    model.eval()
    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, S)
        z = model.encode(x)

        # Fit best linear approximation to encoder
        x_bias = torch.cat([x, torch.ones(n_samples, 1, device=device)], dim=1)
        W_linear = torch.linalg.lstsq(x_bias, z).solution
        z_linear = x_bias @ W_linear

        # Linearity score: fraction of encoding variance explained by linear model
        z_var = z.var(dim=0).sum()
        residual_var = (z - z_linear).var(dim=0).sum()
        linearity_score = 1 - (residual_var / z_var).item()

        # Reconstruction quality comparison
        x_recon_full, _ = model(x)
        x_recon_linear = model.decode(z_linear)

        mse_full = nn.functional.mse_loss(x_recon_full, x).item()
        mse_linear = nn.functional.mse_loss(x_recon_linear, x).item()
        nonlinear_gain = (mse_linear - mse_full) / (mse_linear + 1e-8)

    return {
        'nonlinear_gain': nonlinear_gain,
        'linearity_score': linearity_score,
        'mse_full': mse_full,
        'mse_linear': mse_linear,
    }


def compute_arc_chord_ratio(
    model: Autoencoder, n_points: int = 100, n_features: int = None
) -> Dict:
    """Arc/chord ratio for feature trajectories in latent space.

    For each feature i, traces z(t * e_i) for t in [0, 1].
    Arc = sum of segment lengths, Chord = ||z(1*e_i) - z(0)||.

    Linear encoder: ratio = 1.0 exactly (positive homogeneity).
    Nonlinear encoder: ratio > 1.0 (curved path).

    Returns scalar summary stats + per-feature ratios.
    """
    model.eval()
    n = model.n
    if n_features is None:
        n_features = n

    ratios = []
    with torch.no_grad():
        for i in range(n_features):
            t_values = torch.linspace(0, 1, n_points, device=device)
            trajectory_input = torch.zeros(n_points, n, device=device)
            trajectory_input[:, i] = t_values

            z = model.encode(trajectory_input)

            # Arc length
            segments = z[1:] - z[:-1]
            arc_length = torch.norm(segments, dim=1).sum().item()

            # Chord length
            chord_length = torch.norm(z[-1] - z[0]).item()

            if chord_length > 1e-8:
                ratios.append(arc_length / chord_length)
            else:
                ratios.append(1.0)

    ratios = np.array(ratios)
    return {
        'arc_chord_mean': float(ratios.mean()),
        'arc_chord_std': float(ratios.std()),
        'arc_chord_max': float(ratios.max()),
        'arc_chord_median': float(np.median(ratios)),
        'arc_chord_ratios': ratios,  # array
    }


def compute_encoder_hessian(
    model: Autoencoder, S: float = 0.9, n_samples: int = 50
) -> Dict[str, float]:
    """Finite-difference Hessian approximation of the encoder.

    For each sample, estimates the Hessian of each output dimension
    w.r.t. input using central differences. Measures second-order nonlinearity.

    A linear encoder has Hessian ≡ 0.
    """
    model.eval()
    eps = 1e-3

    x_data = generate_sparse_data(n_samples, model.n, S)

    hessian_norms = []
    with torch.no_grad():
        for idx in range(min(n_samples, 20)):
            x0 = x_data[idx:idx+1]  # (1, n)
            z0 = model.encode(x0)    # (1, m)

            # Estimate diagonal of Hessian via central differences
            # H_ii ≈ (f(x+eps*e_i) - 2f(x) + f(x-eps*e_i)) / eps^2
            diag_hessian = torch.zeros(model.m, model.n, device=device)
            for i in range(model.n):
                e_i = torch.zeros(1, model.n, device=device)
                e_i[0, i] = eps

                z_plus = model.encode(x0 + e_i)
                z_minus = model.encode(x0 - e_i)

                diag_hessian[:, i] = (z_plus[0] - 2 * z0[0] + z_minus[0]) / (eps ** 2)

            hessian_norms.append(torch.norm(diag_hessian).item())

    hessian_norms = np.array(hessian_norms)
    return {
        'hessian_diag_norm_mean': float(hessian_norms.mean()),
        'hessian_diag_norm_std': float(hessian_norms.std()),
        'hessian_diag_norm_max': float(hessian_norms.max()),
    }


def compute_feature_trajectories(
    model: Autoencoder, n_points: int = 50, n_features: int = None,
    t_max: float = 1.0
) -> Dict:
    """Compute feature trajectories z(t * e_i) with tick marks at different levels.

    Returns trajectory data for visualization. Also computes:
    - Gradient at origin: dz/dt at t→0 for each feature
    - How the gradient changes with feature magnitude
    """
    model.eval()
    n = model.n
    if n_features is None:
        n_features = min(n, 16)

    t_values = torch.linspace(0, t_max, n_points, device=device)
    tick_levels = [0.1, 0.25, 0.5, 0.75, 1.0]

    trajectories = []
    gradients_at_origin = []
    gradient_changes = []

    with torch.no_grad():
        for i in range(n_features):
            traj_input = torch.zeros(n_points, n, device=device)
            traj_input[:, i] = t_values
            z = model.encode(traj_input).cpu().numpy()
            trajectories.append(z)

            # Gradient at origin: (z(eps) - z(0)) / eps
            if n_points > 1:
                dt = (t_max / (n_points - 1))
                grad_origin = (z[1] - z[0]) / dt
                gradients_at_origin.append(grad_origin)

                # Gradient at midpoint vs origin
                mid = n_points // 2
                if mid + 1 < n_points:
                    grad_mid = (z[mid + 1] - z[mid]) / dt
                    gradient_changes.append(np.linalg.norm(grad_mid - grad_origin))

    gradient_change_arr = np.array(gradient_changes) if gradient_changes else np.array([0.0])

    return {
        'trajectories': trajectories,  # list of (n_points, m) arrays
        't_values': t_values.cpu().numpy(),
        'tick_levels': tick_levels,
        'gradient_at_origin': gradients_at_origin,  # list of (m,) arrays
        'gradient_change_mean': float(gradient_change_arr.mean()),
        'gradient_change_max': float(gradient_change_arr.max()),
    }


# =============================================================================
# 3. LATENT SPACE METRICS
# =============================================================================

def compute_latent_utilization(
    model: Autoencoder, S: float = 0.9, n_samples: int = 5000
) -> Dict[str, float]:
    """How well the model uses its bottleneck dimensions.

    - variance_per_dim: variance of activations in each latent dimension
    - effective_dim: exp(entropy of normalized variances) — how many dims are "used"
    - utilization_ratio: effective_dim / m
    """
    model.eval()
    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, S)
        z = model.encode(x)

        var_per_dim = z.var(dim=0).cpu().numpy()
        total_var = var_per_dim.sum()

        # Effective dimensionality
        var_normalized = var_per_dim / (total_var + 1e-10)
        entropy = -np.sum(var_normalized * np.log(var_normalized + 1e-10))
        effective_dim = np.exp(entropy)

    return {
        'latent_effective_dim': float(effective_dim),
        'latent_utilization_ratio': float(effective_dim / model.m),
        'latent_total_variance': float(total_var),
        'latent_variance_per_dim': var_per_dim,  # array
        'latent_variance_ratio_max_min': float(
            var_per_dim.max() / (var_per_dim.min() + 1e-10)
        ),
    }


def compute_effective_rank(
    model: Autoencoder, S: float = 0.9, n_samples: int = 2000
) -> Dict[str, float]:
    """Effective rank of the encoding via SVD of the latent representations.

    Different from jacobian-based effective rank in core.py — this measures
    the rank of the actual representation, not the local linearization.
    """
    model.eval()
    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, S)
        z = model.encode(x)

        # Center
        z_centered = z - z.mean(dim=0)

        # SVD
        svd = torch.linalg.svdvals(z_centered)
        svd_normalized = svd / (svd.sum() + 1e-10)

        # Effective rank via entropy
        entropy = -torch.sum(svd_normalized * torch.log(svd_normalized + 1e-10))
        effective_rank = torch.exp(entropy).item()

        # Participation ratio (alternative measure)
        participation_ratio = (svd.sum() ** 2 / (svd ** 2).sum()).item()

        # Fraction of variance in top-k dims
        var_explained = (svd ** 2).cumsum(0) / ((svd ** 2).sum() + 1e-10)

    return {
        'effective_rank_entropy': float(effective_rank),
        'effective_rank_participation': float(participation_ratio),
        'variance_in_top1': float(var_explained[0].item()) if len(var_explained) > 0 else 0,
        'variance_in_top3': float(var_explained[min(2, len(var_explained)-1)].item()),
        'singular_values': svd.cpu().numpy(),  # array
    }


def compute_activation_sparsity(
    model: Autoencoder, S: float = 0.9, n_samples: int = 2000
) -> Dict[str, float]:
    """Fraction of ReLU units active (non-zero) in each layer.

    High activation sparsity means the network uses different "circuits"
    for different inputs — a signature of nonlinear encoding.
    """
    model.eval()

    # Only meaningful for multi-layer encoders
    if model.l < 2 or model.tied_weights:
        return {
            'encoder_activation_sparsity_mean': 0.0,
            'decoder_activation_sparsity_mean': 0.0,
            'encoder_dead_units_frac': 0.0,
        }

    x = generate_sparse_data(n_samples, model.n, S)

    encoder_sparsities = []
    decoder_sparsities = []
    dead_units_per_layer = []

    with torch.no_grad():
        # Trace through encoder layers
        h = x
        for layer in model.encoder:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                active_frac = (h > 0).float().mean().item()
                encoder_sparsities.append(1 - active_frac)  # sparsity = fraction of zeros
                # Dead units: never active across all samples
                per_unit_active = (h > 0).float().mean(dim=0)
                dead_units_per_layer.append((per_unit_active == 0).float().mean().item())

        # Trace through decoder layers
        z = model.encode(x)
        h = z
        for layer in model.decoder:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                active_frac = (h > 0).float().mean().item()
                decoder_sparsities.append(1 - active_frac)

    enc_sparsity = np.mean(encoder_sparsities) if encoder_sparsities else 0.0
    dec_sparsity = np.mean(decoder_sparsities) if decoder_sparsities else 0.0
    dead_frac = np.mean(dead_units_per_layer) if dead_units_per_layer else 0.0

    return {
        'encoder_activation_sparsity_mean': float(enc_sparsity),
        'decoder_activation_sparsity_mean': float(dec_sparsity),
        'encoder_dead_units_frac': float(dead_frac),
        'encoder_sparsity_per_layer': encoder_sparsities,  # list
    }


# =============================================================================
# 4. ROBUSTNESS METRICS
# =============================================================================

def compute_input_robustness(
    model: Autoencoder, S: float = 0.9, n_samples: int = 1000,
    noise_levels: List[float] = None
) -> Dict[str, float]:
    """How robust is reconstruction to input noise?

    Adds Gaussian noise at various levels and measures MSE degradation.
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1, 0.2]

    model.eval()
    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, S)
        x_recon_clean, _ = model(x)
        mse_clean = nn.functional.mse_loss(x_recon_clean, x).item()

        degradation = {}
        for sigma in noise_levels:
            x_noisy = x + sigma * torch.randn_like(x)
            x_recon_noisy, _ = model(x_noisy)
            mse_noisy = nn.functional.mse_loss(x_recon_noisy, x).item()
            degradation[sigma] = mse_noisy / (mse_clean + 1e-10)

    return {
        'mse_clean': mse_clean,
        'robustness_noise_0.01': degradation.get(0.01, float('nan')),
        'robustness_noise_0.05': degradation.get(0.05, float('nan')),
        'robustness_noise_0.1': degradation.get(0.1, float('nan')),
        'robustness_noise_0.2': degradation.get(0.2, float('nan')),
    }


def compute_sparsity_robustness(
    model: Autoencoder, train_S: float = 0.9, n_samples: int = 2000
) -> Dict[str, float]:
    """How well does the model generalize to different sparsity levels?

    Tests reconstruction quality at sparsity levels different from training.
    A model that learned generalizable nonlinear features should degrade
    gracefully; one that overfit to the sparsity pattern will collapse.
    """
    model.eval()
    test_S_values = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

    results = {}
    with torch.no_grad():
        for S in test_S_values:
            x = generate_sparse_data(n_samples, model.n, S)
            x_recon, _ = model(x)
            mse = nn.functional.mse_loss(x_recon, x).item()
            results[f'mse_at_S{S}'] = mse

    # Compute relative degradation from training sparsity
    train_mse = results.get(f'mse_at_S{train_S}', results[f'mse_at_S0.9'])

    return {
        **results,
        'sparsity_robustness_ratio_S0.5': results['mse_at_S0.5'] / (train_mse + 1e-10),
        'sparsity_robustness_ratio_S0.99': results['mse_at_S0.99'] / (train_mse + 1e-10),
    }


# =============================================================================
# 5. TRAINING DYNAMICS METRICS
# =============================================================================

def compute_loss_hessian_trace(
    model: Autoencoder, S: float = 0.9, n_samples: int = 512,
    n_hutchinson: int = 5
) -> Dict[str, float]:
    """Cheap Hessian trace estimate via Hutchinson's method.

    Estimates tr(H) where H is the Hessian of the loss w.r.t. parameters.
    Uses n_hutchinson random vectors: tr(H) ≈ E[v^T H v] for random v.
    Each estimate costs one extra forward+backward pass.

    Higher trace → sharper minimum → less generalizable / more complex.
    """
    model.eval()
    x = generate_sparse_data(n_samples, model.n, S)

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    traces = []
    for _ in range(n_hutchinson):
        # Random Rademacher vector
        vs = [torch.randint_like(p, 0, 2).float() * 2 - 1 for p in params]

        # Forward + backward to get gradients
        model.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        grads = torch.autograd.grad(loss, params, create_graph=True)

        # Hessian-vector product: H @ v = d/dp (grad^T v)
        grad_v = sum((g * v).sum() for g, v in zip(grads, vs))
        hvps = torch.autograd.grad(grad_v, params)

        # tr(H) ≈ v^T H v
        trace_est = sum((v * hvp).sum().item() for v, hvp in zip(vs, hvps))
        traces.append(trace_est)

    traces = np.array(traces)
    return {
        'hessian_trace_mean': float(traces.mean()),
        'hessian_trace_std': float(traces.std()),
        'hessian_trace_per_param': float(traces.mean() / n_params),
    }


def compute_training_dynamics(losses: List[float], window: int = 500) -> Dict[str, float]:
    """Extract metrics from the training loss curve.

    - Convergence speed: steps to reach 2x final loss
    - Discrete drops: sudden decreases in loss (phase transitions)
    - Smoothness: variance of loss improvements
    """
    losses = np.array(losses)
    n = len(losses)

    if n < window:
        return {
            'convergence_speed_2x': n,
            'n_discrete_drops': 0,
            'loss_improvement_smoothness': 0.0,
            'final_loss': float(losses[-1]) if n > 0 else float('nan'),
        }

    # Smoothed loss
    kernel = np.ones(window) / window
    smoothed = np.convolve(losses, kernel, mode='valid')

    final_loss = smoothed[-1]

    # Convergence speed: first step where smoothed loss < 2 * final_loss
    threshold = 2 * final_loss
    convergence_step = n  # default: never
    for i, l in enumerate(smoothed):
        if l < threshold:
            convergence_step = i + window
            break

    # Discrete drops: find windows where loss drops by >10% relative to previous window
    drop_threshold = 0.10
    n_drops = 0
    drop_magnitudes = []
    step_size = max(window // 2, 1)
    for i in range(step_size, len(smoothed), step_size):
        prev = smoothed[max(0, i - step_size)]
        curr = smoothed[i]
        rel_drop = (prev - curr) / (prev + 1e-10)
        if rel_drop > drop_threshold:
            n_drops += 1
            drop_magnitudes.append(rel_drop)

    # Loss improvement smoothness: std of windowed improvements
    improvements = -np.diff(smoothed[::step_size])
    smoothness = float(np.std(improvements) / (np.mean(np.abs(improvements)) + 1e-10))

    return {
        'convergence_speed_2x': convergence_step,
        'n_discrete_drops': n_drops,
        'max_drop_magnitude': float(max(drop_magnitudes)) if drop_magnitudes else 0.0,
        'loss_improvement_smoothness': smoothness,
        'final_loss_smoothed': float(final_loss),
    }


# =============================================================================
# COMBINED: Run all scalar metrics for sweep
# =============================================================================

def compute_all_scalar_metrics(
    model: Autoencoder, S: float = 0.9, losses: List[float] = None
) -> Dict[str, float]:
    """Run all metrics that return scalars (suitable for sweep DataFrame).

    Excludes array-valued and visualization metrics.
    Returns a flat dict of scalar values.
    """
    results = {}

    # Loss
    loss = compute_loss_metrics(model, S=S)
    for k, v in loss.items():
        if isinstance(v, (int, float)):
            results[k] = v

    # Nonlinearity
    gain = compute_nonlinear_gain(model, S=S)
    results.update(gain)

    arc = compute_arc_chord_ratio(model)
    for k, v in arc.items():
        if isinstance(v, (int, float)):
            results[k] = v

    hess = compute_encoder_hessian(model, S=S)
    results.update(hess)

    traj = compute_feature_trajectories(model)
    results['gradient_change_mean'] = traj['gradient_change_mean']
    results['gradient_change_max'] = traj['gradient_change_max']

    # Latent space
    util = compute_latent_utilization(model, S=S)
    for k, v in util.items():
        if isinstance(v, (int, float)):
            results[k] = v

    rank = compute_effective_rank(model, S=S)
    for k, v in rank.items():
        if isinstance(v, (int, float)):
            results[k] = v

    sparsity = compute_activation_sparsity(model, S=S)
    for k, v in sparsity.items():
        if isinstance(v, (int, float)):
            results[k] = v

    # Robustness
    robust = compute_input_robustness(model, S=S)
    results.update(robust)

    sparse_robust = compute_sparsity_robustness(model, train_S=S)
    results.update(sparse_robust)

    # Loss landscape
    hess_trace = compute_loss_hessian_trace(model, S=S)
    results.update(hess_trace)

    # Training dynamics (if losses provided)
    if losses is not None:
        dynamics = compute_training_dynamics(losses)
        results.update(dynamics)

    return results
