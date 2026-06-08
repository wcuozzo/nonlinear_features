"""Run metric prediction sweep with gain-based convergence detection.

Convergence strategy: measure nonlinear_gain every `check_interval` steps.
Stop when gain changes < `gain_tol` (relative) for `patience` consecutive checks.
This directly tracks the metric we care about instead of using loss as a proxy.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import time
from core import Autoencoder, generate_sparse_data, measure_encoding_linearity, device


def train_to_convergence(
    model, S=0.9, batch_size=1024, lr_init=1e-3, weight_decay=1e-2,
    max_steps=200_000, check_interval=5000, gain_tol=0.03, patience=3,
    verbose=False, dynamics_interval=1000,
):
    """Train until nonlinear_gain stabilizes.

    Every `check_interval` steps, measures nonlinear_gain.
    Converged when relative change < `gain_tol` for `patience` consecutive checks.

    Returns: (final_gain_info, n_steps_used, converged, gain_history, losses, dynamics)
    where dynamics is a dict of training dynamics logged every `dynamics_interval` steps.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    # Still use LR scheduling for better optimization, just don't use it for convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5000,
        threshold=1e-3, threshold_mode='rel', min_lr=1e-6,
    )

    losses = []
    gain_history = []  # (step, gain) pairs
    stable_count = 0

    # Training dynamics tracking
    dynamics = {
        'grad_norm_per_layer': [],   # (step, [norm_layer0, norm_layer1, ...])
        'weight_norm_per_layer': [], # (step, [norm_layer0, norm_layer1, ...])
        'lr_history': [],            # (step, lr)
        'total_grad_norm': [],       # (step, total_norm)
    }

    def _log_dynamics(step):
        """Log gradient and weight norms per named parameter group."""
        grad_norms = []
        weight_norms = []
        for name, p in model.named_parameters():
            weight_norms.append(p.data.norm().item())
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
            else:
                grad_norms.append(0.0)
        dynamics['grad_norm_per_layer'].append((step, grad_norms))
        dynamics['weight_norm_per_layer'].append((step, weight_norms))
        dynamics['total_grad_norm'].append((step, sum(g**2 for g in grad_norms)**0.5))
        dynamics['lr_history'].append((step, optimizer.param_groups[0]['lr']))

    for step in range(max_steps):
        x = generate_sparse_data(batch_size, model.n, S)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = nn.functional.mse_loss(x_recon, x)
        loss.backward()

        # Log dynamics before optimizer step (gradients are fresh)
        if step % dynamics_interval == 0:
            _log_dynamics(step)

        optimizer.step()
        losses.append(loss.item())

        # Update LR scheduler every 1000 steps
        if step > 0 and step % 1000 == 0:
            recent_loss = np.mean(losses[-1000:])
            scheduler.step(recent_loss)

        # Check gain convergence (fixed seed for measurement stability)
        if step > 0 and step % check_interval == 0:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(99999)
            linearity = measure_encoding_linearity(model, n_samples=2000, S=S)
            torch.random.set_rng_state(rng_state)
            current_gain = linearity['nonlinear_gain']
            gain_history.append((step, current_gain))

            if len(gain_history) >= 2:
                prev_gain = gain_history[-2][1]
                # Use absolute change for small gains, relative for large
                if abs(prev_gain) < 0.01:
                    # For near-zero gains (l=1 cases), use absolute threshold
                    change = abs(current_gain - prev_gain)
                    is_stable = change < 0.005
                else:
                    rel_change = abs(current_gain - prev_gain) / (abs(prev_gain) + 1e-10)
                    is_stable = rel_change < gain_tol

                if is_stable:
                    stable_count += 1
                else:
                    stable_count = 0

                if verbose:
                    print(f"  Step {step}: gain={current_gain:.4f} "
                          f"(Δ={current_gain - prev_gain:+.4f}, "
                          f"stable={stable_count}/{patience})")

                if stable_count >= patience:
                    if verbose:
                        print(f"  Converged at step {step}")
                    return linearity, step, True, gain_history, losses, dynamics

    if verbose:
        print(f"  Hit max_steps={max_steps}")
    linearity = measure_encoding_linearity(model, S=S)
    gain_history.append((max_steps, linearity['nonlinear_gain']))
    return linearity, max_steps, False, gain_history, losses, dynamics


def run_single(n, m, l, S, **kwargs):
    """Run one experiment with convergence detection.

    Returns dict with metrics + model object + loss curve + dynamics.
    """
    model = Autoencoder(n, m, l, tied_weights=(l == 1)).to(device)
    linearity, n_steps, converged, gain_history, losses, dynamics = train_to_convergence(model, S=S, **kwargs)

    return {
        'nonlinear_gain': linearity['nonlinear_gain'],
        'linearity_score': linearity['linearity_score'],
        'mse_full': linearity['mse_full'],
        'mse_linear': linearity['mse_linear'],
        'n_steps': n_steps,
        'converged': converged,
        'gain_history': gain_history,
        'losses': losses,
        'dynamics': dynamics,
        'model': model,
    }


def max_steps_for_config(n, m, l):
    """Upper bound on training steps. Generous to ensure convergence.

    l=1 is a linear problem, converges fast.
    Otherwise scale with n (larger models = slower convergence).
    """
    if l == 1:
        return 20_000
    # Generous budgets based on empirical testing:
    # n=16 needs ~50k, n=64 needs ~100k, n=256 needs ~200k
    base = 50_000
    n_factor = max(1.0, n / 16)  # linear scaling with n
    return min(int(base * np.sqrt(n_factor)), 300_000)


def run_sweep(save_dir='sweep_models'):
    n_values = [16, 32, 64, 128]
    m_values = [2, 4, 8, 16, 32, 64]
    l_values = [1, 2, 3, 4]
    S_values = [0.85, 0.9, 0.95]
    n_seeds = 3

    configs = [(n, m, l, S)
               for n in n_values for m in m_values if m < n
               for l in l_values for S in S_values]

    os.makedirs(save_dir, exist_ok=True)

    print(f"Configs: {len(configs)}, x{n_seeds} seeds = {len(configs) * n_seeds} runs")
    print(f"Device: {device}")
    print(f"Saving models to: {save_dir}/")
    print(f"Max step budgets: n=16→{max_steps_for_config(16,4,3):,}, "
          f"n=64→{max_steps_for_config(64,16,3):,}, "
          f"n=128→{max_steps_for_config(128,16,3):,}")
    print(f"Convergence: gain stable within 3% for 3 consecutive checks (every 5k steps)")

    # Import full metrics suite
    from metrics import compute_all_scalar_metrics

    all_results = []
    start_time = time.time()

    for idx, (n, m, l, S) in enumerate(tqdm(configs, desc="Sweep")):
        seed_results = []
        budget = max_steps_for_config(n, m, l)
        for seed in range(n_seeds):
            torch.manual_seed(seed * 1000 + hash((n, m, l)) % 1000)
            np.random.seed(seed * 1000 + hash((n, m, l)) % 1000)
            res = run_single(n, m, l, S, max_steps=budget)
            seed_results.append(res)

        best_idx = np.argmin([r['mse_full'] for r in seed_results])
        best = seed_results[best_idx]
        gains = [r['nonlinear_gain'] for r in seed_results]
        steps = [r['n_steps'] for r in seed_results]

        # Save best model weights
        model_path = os.path.join(save_dir, f'model_n{n}_m{m}_l{l}_S{S}.pt')
        torch.save(best['model'].state_dict(), model_path)

        # Save training artifacts (losses, gain_history, dynamics)
        artifacts_path = os.path.join(save_dir, f'artifacts_n{n}_m{m}_l{l}_S{S}.npz')
        dynamics = best['dynamics']
        np.savez_compressed(
            artifacts_path,
            losses=np.array(best['losses'], dtype=np.float32),
            gain_history=np.array(best['gain_history']),
            total_grad_norm=np.array(dynamics['total_grad_norm']),
            lr_history=np.array(dynamics['lr_history']),
            # Per-layer norms: save as 2D arrays (n_checkpoints, n_layers)
            grad_norm_per_layer=np.array([g for _, g in dynamics['grad_norm_per_layer']], dtype=np.float32),
            weight_norm_per_layer=np.array([w for _, w in dynamics['weight_norm_per_layer']], dtype=np.float32),
            grad_norm_steps=np.array([s for s, _ in dynamics['grad_norm_per_layer']]),
        )

        # Compute full metrics suite on best model (include losses for dynamics metrics)
        all_metrics = compute_all_scalar_metrics(best['model'], S=S, losses=best['losses'])

        # Add dynamics summary metrics
        grad_norms = np.array([g for _, g in dynamics['total_grad_norm']])
        all_metrics['grad_norm_mean'] = float(grad_norms.mean())
        all_metrics['grad_norm_max'] = float(grad_norms.max())
        all_metrics['grad_norm_final'] = float(grad_norms[-1]) if len(grad_norms) > 0 else 0.0
        # Gradient norm ratio (final / early) — >1 means growing, <1 means vanishing
        if len(grad_norms) > 10:
            early = grad_norms[:5].mean()
            late = grad_norms[-5:].mean()
            all_metrics['grad_norm_ratio'] = float(late / (early + 1e-10))
        else:
            all_metrics['grad_norm_ratio'] = 1.0

        row = {
            'n': n, 'm': m, 'l': l, 'S': S,
            **all_metrics,
            'gain_mean': np.mean(gains),
            'gain_std': np.std(gains),
            'steps_used': best['n_steps'],
            'steps_mean': np.mean(steps),
            'converged': best['converged'],
            'n_converged': sum(r['converged'] for r in seed_results),
        }
        all_results.append(row)

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(configs) - idx - 1) / rate
            n_conv = sum(1 for r in all_results if r['converged'])
            print(f"\n  [{idx+1}/{len(configs)}] {elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining, "
                  f"converged: {n_conv}/{len(all_results)}")
            # Save intermediate results
            pd.DataFrame(all_results).to_csv('sweep_results_full.csv', index=False)

    df = pd.DataFrame(all_results)
    df.to_csv('sweep_results_full.csv', index=False)

    elapsed = time.time() - start_time
    print(f"\nSaved {len(df)} rows to sweep_results_full.csv")
    print(f"Saved {len(df)} models to {save_dir}/")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Metrics per config: {len([c for c in df.columns if c not in ['n','m','l','S']])}")
    print(f"\nNonlinear gain range: [{df['nonlinear_gain'].min():.4f}, {df['nonlinear_gain'].max():.4f}]")
    print(f"Convergence rate: {df['converged'].mean():.1%}")
    print(f"\nBy n:")
    for n_val in sorted(df['n'].unique()):
        sub = df[df['n'] == n_val]
        print(f"  n={n_val:4d}: converged={sub['converged'].mean():.0%}, "
              f"mean_gain={sub['nonlinear_gain'].mean():.4f}")
    print(f"\nBy l:")
    for l_val in sorted(df['l'].unique()):
        sub = df[df['l'] == l_val]
        print(f"  l={l_val}: converged={sub['converged'].mean():.0%}, "
              f"mean_gain={sub['nonlinear_gain'].mean():.4f}")


if __name__ == '__main__':
    run_sweep()
