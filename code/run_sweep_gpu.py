"""
GPU-optimized sweep: batched seeds + large batches.

Key differences from run_sweep_converged.py:
- BatchedAutoencoder trains K seeds in one forward pass via bmm
- 8x default batch size (8192) -- GPU absorbs the extra compute
- Cosine LR schedule with warmup (replaces ReduceLROnPlateau)
- Adaptive K (seeds) per config based on difficulty heuristics
- Produces CSV compatible with existing analysis notebooks
- Optional wandb logging for live monitoring and analysis

Estimated time: ~1-2 hours on a single GPU (T4/A10/A100)
Works on CPU too, just slower.

Usage:
    python run_sweep_gpu.py                     # Full sweep
    python run_sweep_gpu.py --wandb             # With wandb logging
    python run_sweep_gpu.py --validate          # Verify batched == standard
    python run_sweep_gpu.py --batch-size 16384  # Override batch size
    python run_sweep_gpu.py --skip-full-metrics # Faster (basic metrics only)
"""

import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from core import Autoencoder

# Default device — overridden by --device flag or set_device()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable TF32 for A100s — ~3x faster matmuls with negligible precision loss
torch.set_float32_matmul_precision('high')


def set_device(dev):
    """Set the device for this module and core.py."""
    global device
    import core
    device = torch.device(dev)
    core.device = device


# ════════════════════════════════════════════════════════════════════════
# Data generation (GPU-native: avoids CPU→GPU copy)
# ════════════════════════════════════════════════════════════════════════

def generate_sparse_data(n_samples, n_features, S):
    """Sparse data created directly on the active device."""
    mask = (torch.rand(n_samples, n_features, device=device) > S).float()
    values = torch.rand(n_samples, n_features, device=device)
    return mask * values


# ════════════════════════════════════════════════════════════════════════
# Batched Autoencoder
# ════════════════════════════════════════════════════════════════════════

class BatchedAutoencoder(nn.Module):
    """K autoencoders with identical (n, m, l), different initializations.

    Trains all K seeds in a single forward/backward pass using batched
    matrix multiplies (bmm). Weight tensors have shape [K, in, out].

    Architecture matches core.Autoencoder exactly:
      Encoder: (l-1) × [Linear(n,n) + ReLU] + Linear(n,m)
      Decoder: Linear(m,n) + (l-1) × [ReLU + Linear(n,n)] + ReLU
      l=1: tied weights, no encoder bias
    """

    def __init__(self, n, m, l, K, seeds=None):
        super().__init__()
        self.n, self.m, self.l, self.K = n, m, l, K
        self.tied = (l == 1)

        if seeds is None:
            seeds = list(range(K))

        if self.tied:
            self.W = nn.Parameter(torch.empty(K, n, m))
            self.dec_bias = nn.Parameter(torch.zeros(K, 1, n))
        else:
            self.enc_W = nn.ParameterList()
            self.enc_b = nn.ParameterList()
            for i in range(l - 1):
                self.enc_W.append(nn.Parameter(torch.empty(K, n, n)))
                self.enc_b.append(nn.Parameter(torch.empty(K, 1, n)))
            self.enc_W.append(nn.Parameter(torch.empty(K, n, m)))
            self.enc_b.append(nn.Parameter(torch.empty(K, 1, m)))

            self.dec_W = nn.ParameterList()
            self.dec_b = nn.ParameterList()
            self.dec_W.append(nn.Parameter(torch.empty(K, m, n)))
            self.dec_b.append(nn.Parameter(torch.empty(K, 1, n)))
            for i in range(l - 1):
                self.dec_W.append(nn.Parameter(torch.empty(K, n, n)))
                self.dec_b.append(nn.Parameter(torch.empty(K, 1, n)))

        self._init_from_reference(seeds)

    def _init_from_reference(self, seeds):
        """Initialize each seed's weights from a standard Autoencoder.

        This guarantees identical initialization to the non-batched version,
        regardless of parameter storage layout differences.
        """
        with torch.no_grad():
            for k, seed in enumerate(seeds):
                torch.manual_seed(seed)
                ref = Autoencoder(self.n, self.m, self.l, tied_weights=self.tied)
                if self.tied:
                    # nn.Linear stores [out, in]; we store [in, out]
                    self.W.data[k] = ref.encoder.weight.data.T
                    self.dec_bias.data[k, 0] = ref.decoder_bias.data
                else:
                    idx = 0
                    for layer in ref.encoder:
                        if isinstance(layer, nn.Linear):
                            self.enc_W[idx].data[k] = layer.weight.data.T
                            self.enc_b[idx].data[k, 0] = layer.bias.data
                            idx += 1
                    idx = 0
                    for layer in ref.decoder:
                        if isinstance(layer, nn.Linear):
                            self.dec_W[idx].data[k] = layer.weight.data.T
                            self.dec_b[idx].data[k, 0] = layer.bias.data
                            idx += 1

    def encode(self, x):
        """x: [B, n] -> z: [K, B, m]"""
        h = x.unsqueeze(0).expand(self.K, -1, -1)  # [K, B, n]
        if self.tied:
            return torch.bmm(h, self.W)  # [K, B, m]
        for i in range(self.l - 1):
            h = torch.relu(torch.bmm(h, self.enc_W[i]) + self.enc_b[i])
        return torch.bmm(h, self.enc_W[-1]) + self.enc_b[-1]

    def decode(self, z):
        """z: [K, B, m] -> x_hat: [K, B, n]"""
        if self.tied:
            return torch.relu(
                torch.bmm(z, self.W.transpose(1, 2)) + self.dec_bias
            )
        h = torch.bmm(z, self.dec_W[0]) + self.dec_b[0]
        for i in range(1, self.l):
            h = torch.relu(h)
            h = torch.bmm(h, self.dec_W[i]) + self.dec_b[i]
        return torch.relu(h)

    def forward(self, x):
        """x: [B, n] -> (x_hat: [K, B, n], z: [K, B, m])"""
        z = self.encode(x)
        return self.decode(z), z

    def extract_single(self, k):
        """Extract seed k as a standard core.Autoencoder (for metrics)."""
        model = Autoencoder(self.n, self.m, self.l, tied_weights=self.tied)
        model = model.to(next(self.parameters()).device)
        with torch.no_grad():
            if self.tied:
                model.encoder.weight.data.copy_(self.W.data[k].T)
                model.decoder_bias.data.copy_(self.dec_bias.data[k, 0])
            else:
                idx = 0
                for layer in model.encoder:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data.copy_(self.enc_W[idx].data[k].T)
                        layer.bias.data.copy_(self.enc_b[idx].data[k, 0])
                        idx += 1
                idx = 0
                for layer in model.decoder:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data.copy_(self.dec_W[idx].data[k].T)
                        layer.bias.data.copy_(self.dec_b[idx].data[k, 0])
                        idx += 1
        return model


# ════════════════════════════════════════════════════════════════════════
# Batched linearity measurement
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_batched_linearity(model, n_samples=2000, S=0.9):
    """Nonlinear gain + linearity score for all K seeds at once.

    Returns dict with lists of length K:
        gains, linearity_scores, mse_fulls, mse_linears
    """
    model.eval()
    K = model.K

    x = generate_sparse_data(n_samples, model.n, S)         # [N, n]
    z = model.encode(x)                                      # [K, N, m]
    x_hat_full = model.decode(z)                              # [K, N, n]

    # Batched linear fit: z[k] ~ x_aug @ W_lin[k]
    ones = torch.ones(n_samples, 1, device=device)
    x_aug = torch.cat([x, ones], dim=1)                      # [N, n+1]
    x_aug_K = x_aug.unsqueeze(0).expand(K, -1, -1)           # [K, N, n+1]
    W_lin = torch.linalg.lstsq(x_aug_K, z).solution          # [K, n+1, m]
    z_lin = torch.bmm(x_aug_K, W_lin)                         # [K, N, m]

    x_hat_lin = model.decode(z_lin)                           # [K, N, n]

    # Linearity score: 1 - Var(residual) / Var(z)
    z_var = z.var(dim=1).sum(dim=1)                           # [K]
    res_var = (z - z_lin).var(dim=1).sum(dim=1)               # [K]
    linearity_scores = (1 - res_var / (z_var + 1e-10))        # [K]

    # MSE and nonlinear gain
    x_K = x.unsqueeze(0).expand(K, -1, -1)                   # [K, N, n]
    mse_full = ((x_hat_full - x_K) ** 2).mean(dim=(1, 2))    # [K]
    mse_lin = ((x_hat_lin - x_K) ** 2).mean(dim=(1, 2))      # [K]
    gains = (mse_lin - mse_full) / (mse_lin + 1e-8)           # [K]

    model.train()
    return {
        'gains': gains.cpu().tolist(),
        'linearity_scores': linearity_scores.cpu().tolist(),
        'mse_fulls': mse_full.cpu().tolist(),
        'mse_linears': mse_lin.cpu().tolist(),
    }


# ════════════════════════════════════════════════════════════════════════
# LR schedule
# ════════════════════════════════════════════════════════════════════════

def cosine_lr(step, max_steps, lr_peak, warmup=1000, lr_min=1e-6):
    """Linear warmup then cosine decay."""
    warmup = min(warmup, max_steps // 10)
    if step < warmup:
        return lr_peak * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * progress))


# ════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════

def train_batched(
    model, S=0.9, batch_size=8192, lr_base=1e-3, weight_decay=1e-2,
    max_steps=100_000, check_interval=2500, gain_tol=0.03, patience=3,
    verbose=False,
):
    """Train all K seeds to convergence.

    LR is sqrt-scaled from lr_base (calibrated at batch_size=1024).
    Stops when >= ceil(K/2) seeds have individually converged, or max_steps.

    Returns dict with per-seed metrics.
    """
    K = model.K
    lr_peak = lr_base * math.sqrt(batch_size / 1024)

    optimizer = optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)

    # Compile forward pass for ~3x speedup on small models
    compiled_model = torch.compile(model) if torch.cuda.is_available() else model

    gain_history = [[] for _ in range(K)]
    stable_counts = [0] * K
    seed_converged = [False] * K
    min_converged = max(1, math.ceil(K / 2))

    losses_log = []  # (step, [mse_per_seed])
    final_step = max_steps

    for step in range(max_steps):
        # Update LR
        lr = cosine_lr(step, max_steps, lr_peak)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward
        x = generate_sparse_data(batch_size, model.n, S)
        x_hat, z = compiled_model(x)

        # Per-seed MSE, summed for backward (Adam is scale-invariant)
        x_K = x.unsqueeze(0).expand(K, -1, -1)
        mse_per_seed = ((x_hat - x_K) ** 2).mean(dim=(1, 2))
        loss = mse_per_seed.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            losses_log.append((step, mse_per_seed.detach().cpu().tolist()))

        # Convergence check
        if step > 0 and step % check_interval == 0:
            rng_state = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            torch.manual_seed(99999)

            lin = measure_batched_linearity(model, n_samples=2000, S=S)

            torch.random.set_rng_state(rng_state)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)

            for k in range(K):
                gain_history[k].append((step, lin['gains'][k]))

                if len(gain_history[k]) >= 2:
                    prev = gain_history[k][-2][1]
                    curr = lin['gains'][k]
                    if abs(prev) < 0.01:
                        is_stable = abs(curr - prev) < 0.005
                    else:
                        is_stable = abs(curr - prev) / (abs(prev) + 1e-10) < gain_tol
                    stable_counts[k] = stable_counts[k] + 1 if is_stable else 0
                    if stable_counts[k] >= patience:
                        seed_converged[k] = True

            n_conv = sum(seed_converged)
            if verbose:
                gs = ', '.join(f'{g:.3f}' for g in lin['gains'][:6])
                tail = '...' if K > 6 else ''
                print(f"  Step {step}: gains=[{gs}{tail}] "
                      f"conv={n_conv}/{K} lr={lr:.2e}")

            if n_conv >= min_converged:
                final_step = step + 1
                if verbose:
                    print(f"  Converged at step {step} ({n_conv}/{K} seeds)")
                break

    # Final measurement
    torch.manual_seed(99999)
    final = measure_batched_linearity(model, n_samples=2000, S=S)

    return {
        'gains': final['gains'],
        'linearity_scores': final['linearity_scores'],
        'mse_fulls': final['mse_fulls'],
        'mse_linears': final['mse_linears'],
        'gain_histories': gain_history,
        'seed_converged': seed_converged,
        'steps_used': final_step,
        'losses_log': losses_log,
    }


# ════════════════════════════════════════════════════════════════════════
# Config helpers
# ════════════════════════════════════════════════════════════════════════

def get_seeds_for_config(n, m, l, multiplier=1):
    """Adaptive seed count: more for hard configs.

    multiplier scales all counts (e.g. 2 = double seeds everywhere).
    """
    if l == 1:
        return max(3, int(3 * multiplier))
    K = 5  # base for nonlinear
    if m <= 4:
        K += 5  # narrow bottleneck = hard optimization
    if n >= 64 and m >= n // 2:
        K += 3  # high-dimensional near-boundary
    return min(int(K * multiplier), 40)


def get_max_steps(n, m, l):
    """Generous upper bound; convergence detection does the real stopping."""
    if l == 1:
        return 20_000
    base = 40_000
    return min(int(base * math.sqrt(max(1, n / 16))), 200_000)


def make_seeds(K, n, m, l):
    """Deterministic seed list for a (n, m, l) config."""
    base = hash((n, m, l)) % 10000
    return [base + k * 1000 for k in range(K)]


# ════════════════════════════════════════════════════════════════════════
# Per-config training (shared by single-GPU and multi-GPU paths)
# ════════════════════════════════════════════════════════════════════════

def _train_and_evaluate_config(n, m, l, S, K, batch_size, max_steps,
                               save_dir, skip_full_metrics):
    """Train one (n,m,l,S) config with K batched seeds. Return results.

    Assumes `device` is already set for this process.
    Returns (row_dict, gain_histories).
    """
    seeds = make_seeds(K, n, m, l)
    model = BatchedAutoencoder(n, m, l, K, seeds).to(device)

    res = train_batched(model, S=S, batch_size=batch_size, max_steps=max_steps)

    best_k = int(np.argmin(res['mse_fulls']))
    best_model = model.extract_single(best_k)

    # Save model weights (CPU to avoid GPU memory leak in multi-GPU)
    model_path = os.path.join(save_dir, f'model_n{n}_m{m}_l{l}_S{S}.pt')
    torch.save(best_model.cpu().state_dict(), model_path)

    # Full metrics on best model
    extra = {}
    if not skip_full_metrics:
        try:
            from metrics import compute_all_scalar_metrics
            losses_best = [entry[best_k] for _, entry in res['losses_log']]
            extra = compute_all_scalar_metrics(
                best_model.to(device), S=S, losses=losses_best)
        except Exception as e:
            extra = {'metrics_error': str(e)}

    row = {
        'n': n, 'm': m, 'l': l, 'S': S,
        'nonlinear_gain': res['gains'][best_k],
        'linearity_score': res['linearity_scores'][best_k],
        'mse_full': res['mse_fulls'][best_k],
        'mse_linear': res['mse_linears'][best_k],
        'gain_mean': float(np.mean(res['gains'])),
        'gain_std': float(np.std(res['gains'])),
        'steps_used': res['steps_used'],
        'steps_mean': float(res['steps_used']),
        'converged': res['seed_converged'][best_k],
        'n_converged': sum(res['seed_converged']),
        'n_seeds': K,
        **extra,
    }

    # Free GPU memory
    del model, best_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row, res['gain_histories']


def _gpu_worker(gpu_id, task_queue, result_queue, batch_size,
                save_dir, skip_full_metrics):
    """Worker process: pulls configs from queue, trains on one GPU."""
    global device
    import core as _core
    device = torch.device(f'cuda:{gpu_id}')
    _core.device = device

    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, n, m, l, S, K, max_steps = item
        try:
            row, histories = _train_and_evaluate_config(
                n, m, l, S, K, batch_size, max_steps,
                save_dir, skip_full_metrics)
            result_queue.put((idx, row, histories))
        except Exception as e:
            result_queue.put((idx, {
                'n': n, 'm': m, 'l': l, 'S': S,
                'nonlinear_gain': float('nan'),
                'converged': False,
                'error': str(e),
            }, [[] for _ in range(K)]))


# ════════════════════════════════════════════════════════════════════════
# Wandb helpers
# ════════════════════════════════════════════════════════════════════════

def _wandb_log_config(row, idx, total, start_time, all_results):
    """Log per-config metrics to wandb."""
    if not HAS_WANDB or wandb.run is None:
        return

    elapsed = time.time() - start_time
    n_done = idx + 1
    n_conv = sum(1 for r in all_results if r['converged'])

    wandb.log({
        # Config identity
        'config/n': row['n'],
        'config/m': row['m'],
        'config/l': row['l'],
        'config/S': row['S'],
        'config/n_over_m': row['n'] / row['m'],

        # Best seed results
        'best/nonlinear_gain': row['nonlinear_gain'],
        'best/mse_full': row['mse_full'],
        'best/mse_linear': row['mse_linear'],
        'best/linearity_score': row.get('linearity_score', 0),

        # Seed statistics
        'seeds/gain_mean': row['gain_mean'],
        'seeds/gain_std': row['gain_std'],
        'seeds/n_converged': row['n_converged'],
        'seeds/n_seeds': row['n_seeds'],
        'seeds/convergence_rate': row['n_converged'] / row['n_seeds'],

        # Training
        'training/steps_used': row['steps_used'],
        'training/converged': int(row['converged']),

        # Sweep progress
        'sweep/configs_done': n_done,
        'sweep/pct_done': n_done / total,
        'sweep/convergence_rate': n_conv / n_done,
        'sweep/elapsed_min': elapsed / 60,
    }, step=idx)


def _wandb_log_summary(df, all_gain_histories, configs):
    """Log summary tables and phase diagram plots at sweep end."""
    if not HAS_WANDB or wandb.run is None:
        return

    # 1. Full results table (filterable in wandb UI)
    wandb.log({'results': wandb.Table(dataframe=df)})

    # 2. Gain convergence curves table
    #    Each row: config_idx, n, m, l, S, seed_k, check_step, gain
    curve_rows = []
    for cfg_idx, ((n, m, l, S), histories) in enumerate(
            zip(configs, all_gain_histories)):
        for k, seed_hist in enumerate(histories):
            for step, gain in seed_hist:
                curve_rows.append({
                    'config_idx': cfg_idx,
                    'n': n, 'm': m, 'l': l, 'S': S,
                    'seed_k': k, 'step': step, 'gain': gain,
                })
    if curve_rows:
        wandb.log({'gain_curves': wandb.Table(
            dataframe=pd.DataFrame(curve_rows))})

    # 3. Phase diagram heatmaps
    if not HAS_MPL:
        return

    for S_val in sorted(df['S'].unique()):
        sub = df[df['S'] == S_val]
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle(f'Nonlinear Gain Phase Diagram (S={S_val})', fontsize=14)

        for ax_idx, l_val in enumerate(sorted(sub['l'].unique())):
            ax = axes[ax_idx]
            lsub = sub[sub['l'] == l_val]

            # Pivot to heatmap: rows=n, cols=m
            pivot = lsub.pivot_table(
                values='nonlinear_gain', index='n', columns='m',
                aggfunc='first')
            pivot = pivot.sort_index(ascending=True)

            im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                           vmin=0, vmax=1, origin='lower')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns.astype(int))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index.astype(int))
            ax.set_xlabel('m (bottleneck)')
            ax.set_ylabel('n (input dim)')
            ax.set_title(f'l={l_val}')

            # Annotate cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        color = 'white' if val < 0.3 or val > 0.7 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=7, color=color)

        fig.colorbar(im, ax=axes, shrink=0.8, label='Nonlinear Gain')
        plt.tight_layout()
        wandb.log({f'phase_diagram/S={S_val}': wandb.Image(fig)})
        plt.close(fig)

    # 4. Convergence rate by depth
    fig, ax = plt.subplots(figsize=(8, 5))
    for l_val in sorted(df['l'].unique()):
        lsub = df[df['l'] == l_val]
        ratios = lsub.groupby('n')['converged'].mean()
        ax.plot(ratios.index, ratios.values, 'o-', label=f'l={l_val}')
    ax.set_xlabel('n (input dimension)')
    ax.set_ylabel('Best-seed convergence rate')
    ax.set_title('Convergence Rate by Depth and Input Dimension')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({'convergence_by_depth': wandb.Image(fig)})
    plt.close(fig)

    # 5. Summary metrics
    wandb.run.summary['total_configs'] = len(df)
    wandb.run.summary['overall_convergence_rate'] = df['converged'].mean()
    wandb.run.summary['mean_nonlinear_gain'] = df['nonlinear_gain'].mean()
    wandb.run.summary['max_nonlinear_gain'] = df['nonlinear_gain'].max()
    wandb.run.summary['worse_than_linear_frac'] = (
        df[df['l'] > 1]['nonlinear_gain'] < 0).mean()


# ════════════════════════════════════════════════════════════════════════
# Main sweep
# ════════════════════════════════════════════════════════════════════════

def run_sweep(batch_size=8192, save_dir='sweep_gpu', skip_full_metrics=False,
              use_wandb=False, wandb_project=None, wandb_name=None,
              n_gpus=1, seed_multiplier=1):
    n_values = [16, 32, 64, 128]
    m_values = [2, 4, 8, 16, 32, 64]
    l_values = [1, 2, 3, 4]
    S_values = [0.85, 0.9, 0.95]

    configs = [(n, m, l, S)
               for n in n_values for m in m_values if m < n
               for l in l_values for S in S_values]

    os.makedirs(save_dir, exist_ok=True)

    lr_peak = 1e-3 * math.sqrt(batch_size / 1024)
    total_seeds = sum(get_seeds_for_config(n, m, l, seed_multiplier)
                      for n, m, l, _ in configs)
    print(f"Configs: {len(configs)}, total seeds: {total_seeds}")
    print(f"Device: {device}, GPUs: {n_gpus}, seed multiplier: {seed_multiplier}")
    print(f"Batch size: {batch_size}, LR: 1e-3 * sqrt({batch_size}/1024) = {lr_peak:.4f}")

    # Wandb init (main process only)
    if use_wandb:
        if not HAS_WANDB:
            print("Warning: wandb not installed, continuing without it")
            use_wandb = False
        else:
            wandb.init(
                project=wandb_project or 'nonlinear-feature-phase-diagram',
                name=wandb_name or f'sweep-{time.strftime("%Y%m%d-%H%M%S")}',
                config={
                    'batch_size': batch_size,
                    'lr_base': 1e-3,
                    'lr_peak': lr_peak,
                    'weight_decay': 1e-2,
                    'n_values': n_values,
                    'm_values': m_values,
                    'l_values': l_values,
                    'S_values': S_values,
                    'n_configs': len(configs),
                    'total_seeds': total_seeds,
                    'seed_multiplier': seed_multiplier,
                    'n_gpus': n_gpus,
                    'skip_full_metrics': skip_full_metrics,
                    'device': str(device),
                    'save_dir': save_dir,
                },
            )
            print(f"Wandb run: {wandb.run.url}")

    all_results = [None] * len(configs)
    all_gain_histories = [None] * len(configs)
    start = time.time()

    # ── Multi-GPU path ──────────────────────────────────────────────
    if n_gpus > 1:
        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')

        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        # Fill task queue
        for idx, (n, m, l, S) in enumerate(configs):
            K = get_seeds_for_config(n, m, l, seed_multiplier)
            max_steps = get_max_steps(n, m, l)
            task_queue.put((idx, n, m, l, S, K, max_steps))
        for _ in range(n_gpus):
            task_queue.put(None)  # poison pills

        # Launch workers
        workers = []
        for gpu_id in range(n_gpus):
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, task_queue, result_queue,
                      batch_size, save_dir, skip_full_metrics))
            p.start()
            workers.append(p)
        print(f"Launched {n_gpus} workers on cuda:0..cuda:{n_gpus-1}")

        # Collect results
        pbar = tqdm(total=len(configs), desc=f'Sweep ({n_gpus} GPUs)')
        completed = 0

        while completed < len(configs):
            # Check worker health
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0 and completed < len(configs):
                print(f"\nAll workers exited after {completed}/{len(configs)}")
                break

            try:
                idx, row, histories = result_queue.get(timeout=60)
            except Exception:
                continue

            all_results[idx] = row
            all_gain_histories[idx] = histories
            completed += 1
            pbar.update(1)

            if 'error' in row:
                pbar.write(f"  FAIL n={row['n']} m={row['m']} "
                           f"l={row['l']} S={row['S']}: {row['error']}")
            elif use_wandb:
                valid = [r for r in all_results
                         if r is not None and 'error' not in r]
                _wandb_log_config(row, completed - 1, len(configs),
                                  start, valid)

            if completed % 20 == 0:
                valid = [r for r in all_results
                         if r is not None and 'error' not in r]
                if valid:
                    pd.DataFrame(valid).to_csv(
                        os.path.join(save_dir, 'sweep_results.csv'),
                        index=False)

        pbar.close()
        for p in workers:
            p.join(timeout=30)

        # Report errors
        errors = [r for r in all_results if r is not None and 'error' in r]
        if errors:
            print(f"\n{len(errors)} configs failed:")
            for r in errors[:10]:
                print(f"  n={r['n']} m={r['m']} l={r['l']} "
                      f"S={r['S']}: {r['error']}")

        # Keep only successful results
        good_idx = [i for i, r in enumerate(all_results)
                    if r is not None and 'error' not in r]
        all_results = [all_results[i] for i in good_idx]
        all_gain_histories = [all_gain_histories[i] for i in good_idx]
        configs_for_wandb = [configs[i] for i in good_idx]

    # ── Single-GPU path ─────────────────────────────────────────────
    else:
        configs_for_wandb = configs
        results_list = []
        histories_list = []

        for idx, (n, m, l, S) in enumerate(tqdm(configs, desc='Sweep')):
            K = get_seeds_for_config(n, m, l, seed_multiplier)
            max_steps = get_max_steps(n, m, l)

            row, histories = _train_and_evaluate_config(
                n, m, l, S, K, batch_size, max_steps,
                save_dir, skip_full_metrics)

            results_list.append(row)
            histories_list.append(histories)

            if use_wandb:
                _wandb_log_config(row, idx, len(configs), start, results_list)

            if (idx + 1) % 20 == 0:
                elapsed = time.time() - start
                rate = (idx + 1) / elapsed
                remaining = (len(configs) - idx - 1) / rate
                n_conv = sum(1 for r in results_list if r.get('converged'))
                print(f"\n  [{idx+1}/{len(configs)}] {elapsed/60:.1f}m elapsed, "
                      f"~{remaining/60:.1f}m remaining, "
                      f"converged: {n_conv}/{len(results_list)}")
                pd.DataFrame(results_list).to_csv(
                    os.path.join(save_dir, 'sweep_results.csv'), index=False)

        all_results = results_list
        all_gain_histories = histories_list

    # ── Post-processing (both paths) ────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(save_dir, 'sweep_results.csv')
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - start
    print(f"\nDone! {len(df)} configs in {elapsed/60:.1f} minutes")
    print(f"Results: {csv_path}")
    print(f"Models:  {save_dir}/model_*.pt")
    if len(df) > 0:
        print(f"\nNonlinear gain: [{df['nonlinear_gain'].min():.4f}, "
              f"{df['nonlinear_gain'].max():.4f}]")
        print(f"Convergence: {df['converged'].mean():.0%} of best seeds, "
              f"{df['n_converged'].sum()}/{df['n_seeds'].sum()} total")
        for lv in sorted(df['l'].unique()):
            sub = df[df['l'] == lv]
            print(f"  l={lv}: conv={sub['converged'].mean():.0%}, "
                  f"gain={sub['nonlinear_gain'].mean():.4f}")

    # Wandb summary
    if use_wandb:
        print("Logging wandb summary...")
        _wandb_log_summary(df, all_gain_histories, configs_for_wandb)
        wandb.finish()
        print("Wandb run finished.")

    return df


# ════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════

def validate():
    """Verify BatchedAutoencoder matches standard Autoencoder exactly."""
    print("=== Forward pass validation ===")

    cases = [(8, 3, 1), (8, 3, 2), (16, 4, 3), (32, 8, 4)]
    all_pass = True

    for n, m, l in cases:
        seed = 42
        torch.manual_seed(seed)
        std = Autoencoder(n, m, l, tied_weights=(l == 1)).to(device)

        batched = BatchedAutoencoder(n, m, l, K=1, seeds=[seed]).to(device)
        extracted = batched.extract_single(0)

        torch.manual_seed(0)
        x = generate_sparse_data(200, n, 0.9)

        std.eval(); batched.eval(); extracted.eval()
        with torch.no_grad():
            xh_std, z_std = std(x)
            xh_bat, z_bat = batched(x)
            xh_ext, z_ext = extracted(x)

        z_err = (z_std - z_bat[0]).abs().max().item()
        xh_err = (xh_std - xh_bat[0]).abs().max().item()
        ext_err = (xh_std - xh_ext).abs().max().item()
        ok = max(z_err, xh_err, ext_err) < 1e-5
        all_pass = all_pass and ok
        print(f"  n={n:3d} m={m:2d} l={l}: "
              f"z_err={z_err:.2e}  xh_err={xh_err:.2e}  "
              f"ext_err={ext_err:.2e}  [{'PASS' if ok else 'FAIL'}]")

    print("\n=== Training step validation (20 steps, K=1 vs standard) ===")
    n, m, l = 8, 3, 2
    seed = 42
    lr, wd = 1e-3, 1e-2

    torch.manual_seed(seed)
    std = Autoencoder(n, m, l, tied_weights=False).to(device)
    std_opt = optim.AdamW(std.parameters(), lr=lr, weight_decay=wd)

    batched = BatchedAutoencoder(n, m, l, K=1, seeds=[seed]).to(device)
    bat_opt = optim.AdamW(batched.parameters(), lr=lr, weight_decay=wd)

    for step in range(20):
        torch.manual_seed(1000 + step)
        x = generate_sparse_data(64, n, 0.9)

        # Standard step
        std_opt.zero_grad()
        xh_s, _ = std(x)
        loss_s = nn.functional.mse_loss(xh_s, x)
        loss_s.backward()
        std_opt.step()

        # Batched step (same data -- same seed, same device)
        bat_opt.zero_grad()
        xh_b, _ = batched(x)
        x_K = x.unsqueeze(0)
        loss_b = ((xh_b - x_K) ** 2).mean(dim=(1, 2)).sum()
        loss_b.backward()
        bat_opt.step()

    # Compare outputs after training
    ext = batched.extract_single(0)
    std.eval(); ext.eval()
    with torch.no_grad():
        torch.manual_seed(9999)
        x_test = generate_sparse_data(100, n, 0.9)
        xh_s, _ = std(x_test)
        xh_e, _ = ext(x_test)
        err = (xh_s - xh_e).abs().max().item()

    ok = err < 1e-4
    all_pass = all_pass and ok
    print(f"  After 20 steps: max output diff = {err:.2e}  "
          f"[{'PASS' if ok else 'WARN' if err < 1e-2 else 'FAIL'}]")

    print("\n=== Batched linearity measurement ===")
    n, m, l = 16, 4, 2
    seed = 42
    torch.manual_seed(seed)
    std = Autoencoder(n, m, l, tied_weights=False).to(device)
    # Quick train
    std_opt = optim.AdamW(std.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(500):
        x = generate_sparse_data(256, n, 0.9)
        std_opt.zero_grad()
        xh, _ = std(x)
        nn.functional.mse_loss(xh, x).backward()
        std_opt.step()

    batched = BatchedAutoencoder(n, m, l, K=1, seeds=[seed]).to(device)
    # Copy trained weights into batched model
    with torch.no_grad():
        idx = 0
        for layer in std.encoder:
            if isinstance(layer, nn.Linear):
                batched.enc_W[idx].data[0] = layer.weight.data.T
                batched.enc_b[idx].data[0, 0] = layer.bias.data
                idx += 1
        idx = 0
        for layer in std.decoder:
            if isinstance(layer, nn.Linear):
                batched.dec_W[idx].data[0] = layer.weight.data.T
                batched.dec_b[idx].data[0, 0] = layer.bias.data
                idx += 1

    torch.manual_seed(12345)
    from core import measure_encoding_linearity
    std_lin = measure_encoding_linearity(std, n_samples=2000, S=0.9)

    torch.manual_seed(12345)
    bat_lin = measure_batched_linearity(batched, n_samples=2000, S=0.9)

    gain_err = abs(std_lin['nonlinear_gain'] - bat_lin['gains'][0])
    mse_err = abs(std_lin['mse_full'] - bat_lin['mse_fulls'][0])
    ok = gain_err < 5e-3 and mse_err < 5e-3
    all_pass = all_pass and ok
    print(f"  gain err={gain_err:.2e}  mse err={mse_err:.2e}  "
          f"[{'PASS' if ok else 'FAIL'}]")

    print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GPU-optimized phase diagram sweep')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation checks instead of sweep')
    parser.add_argument('--batch-size', type=int, default=8192,
                        help='Training batch size (default: 8192)')
    parser.add_argument('--save-dir', default='sweep_gpu',
                        help='Output directory (default: sweep_gpu)')
    parser.add_argument('--skip-full-metrics', action='store_true',
                        help='Skip expensive per-model metric suite')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb-project', default=None,
                        help='Wandb project name (default: nonlinear-feature-phase-diagram)')
    parser.add_argument('--wandb-name', default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--device', default=None,
                        help='Device for single-GPU mode (e.g. cuda:0, cpu). '
                             'Ignored when --n-gpus > 1')
    parser.add_argument('--n-gpus', type=int, default=1,
                        help='Number of GPUs for parallel training (default: 1)')
    parser.add_argument('--seed-multiplier', type=float, default=1,
                        help='Multiply seed counts by this factor (default: 1). '
                             'Use 2-3 with multi-GPU for more robust results')
    args = parser.parse_args()

    if args.device and args.n_gpus <= 1:
        set_device(args.device)

    if args.validate:
        validate()
    else:
        run_sweep(
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            skip_full_metrics=args.skip_full_metrics,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            n_gpus=args.n_gpus,
            seed_multiplier=args.seed_multiplier,
        )
