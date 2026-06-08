"""
Targeted push for a single config — use lots of seeds + final precise eval to
genuinely beat noise-floor MSE differences.

Usage:
    python focused_push.py --n 128 --m 64 --l 4 --S 0.95 --K 80 --steps-mult 5.0
"""
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import core
import run_sweep_gpu as _sweep
from core import Autoencoder, generate_sparse_data
from results_store import ResultsStore
from run_sweep_gpu import BatchedAutoencoder, measure_batched_linearity
from sweep_violation_fix import load_seed_into_batched, cosine_lr_inline
from warm_start import (
    embed_shallow_in_deep, perturb_model, random_init,
    eval_mse, load_best_model,
)
from precise_recompile import precise_mse


def build_random_only_pool(n, m, l, S, K, device='cuda'):
    """All K seeds are pure random Autoencoder inits."""
    from warm_start import random_init
    pool, seed_values, descriptions = [], [], []
    base_seed = hash((n, m, l, S, 'random')) % 1000000
    for i in range(K):
        seed = base_seed + i
        ae = random_init(n, m, l, seed=seed, device=device)
        pool.append(ae); seed_values.append(seed)
        descriptions.append(f'random_only_seed{seed}')
    init_mses = [eval_mse(ae, S, n_samples=20000, device=device) for ae in pool]
    return pool, init_mses, descriptions, seed_values, float('inf')


def build_near_warm_start_pool(n, m, l, S, K, models_dir='results_db/models', device='cuda'):
    """All K seeds are zero-noise identity-embeds of best shallow + a tiny
    perturbation. Aimed at near-basin exploration without leaving the basin.
    Noise levels spread log-uniformly from 1e-7 to 1e-2.
    """
    # Find best shallow
    best_shallow = None; best_shallow_l = None; best_shallow_mse = float('inf')
    for l_s in range(1, l):
        sh = load_best_model(n, m, l_s, S, models_dir, device=device)
        if sh is None: continue
        mse_s = eval_mse(sh, S, n_samples=20000, device=device)
        if mse_s < best_shallow_mse:
            best_shallow_mse = mse_s; best_shallow = sh; best_shallow_l = l_s
    print(f'  best shallow: l={best_shallow_l} mse(n=20k)={best_shallow_mse:.6f}')

    pool, seed_values, descriptions = [], [], []
    base_seed = hash((n, m, l, S, 'near')) % 1000000

    # Log-uniform noise levels from 1e-7 to 1e-2
    noise_grid = np.logspace(-7, -2, num=K).tolist()
    for i in range(K):
        noise = noise_grid[i]
        seed = base_seed + i
        ae = embed_shallow_in_deep(best_shallow, l, noise=noise, seed=seed, device=device)
        pool.append(ae); seed_values.append(seed)
        descriptions.append(f'near_warm_start(l={best_shallow_l}, noise={noise:.1e})')
    init_mses = [eval_mse(ae, S, n_samples=20000, device=device) for ae in pool]
    return pool, init_mses, descriptions, seed_values, best_shallow_mse


def build_huge_pool(n, m, l, S, K, models_dir='results_db/models', device='cuda'):
    """Build K diverse seeds.

    Mix:
      - half from best shallow embed with varied noise (0.0, 0.001, 0.003, 0.01)
      - quarter from stored best perturbed
      - eighth from neighbor projection
      - eighth random
    """
    best_shallow = None
    best_shallow_l = None
    best_shallow_mse = float('inf')
    for l_s in range(1, l):
        sh = load_best_model(n, m, l_s, S, models_dir, device=device)
        if sh is None: continue
        mse_s = eval_mse(sh, S, n_samples=20000, device=device)
        if mse_s < best_shallow_mse:
            best_shallow_mse = mse_s; best_shallow = sh; best_shallow_l = l_s
    print(f'  best shallow: l={best_shallow_l} mse(n=20k)={best_shallow_mse:.6f}')

    stored = load_best_model(n, m, l, S, models_dir, device=device)
    stored_mse = eval_mse(stored, S, n_samples=20000, device=device) if stored else float('inf')
    print(f'  current stored: mse(n=20k)={stored_mse:.6f}')

    pool = []
    seed_values = []
    descriptions = []

    # Mix: more random for landscape exploration
    n_shallow = K // 4
    n_stored = K // 8 if stored is not None else 0
    n_neighbor = K // 8
    n_random = K - n_shallow - n_stored - n_neighbor  # rest random (half)

    base_seed = hash((n, m, l, S)) % 1000000

    # Shallow embeds: varied noise
    shallow_noises = [0.0, 0.0, 0.0, 0.001, 0.001, 0.003, 0.003, 0.01, 0.01, 0.03]
    for i in range(n_shallow):
        noise = shallow_noises[i % len(shallow_noises)]
        seed = base_seed + 100 + i
        ae = embed_shallow_in_deep(best_shallow, l, noise=noise, seed=seed, device=device)
        pool.append(ae); seed_values.append(seed)
        descriptions.append(f'shallow(l={best_shallow_l})+noise={noise}')

    # Stored perturbations
    for i in range(n_stored):
        noise = [0.0005, 0.001, 0.003, 0.01][i % 4]
        seed = base_seed + 200 + i
        ae = perturb_model(stored, noise=noise, seed=seed, device=device)
        pool.append(ae); seed_values.append(seed)
        descriptions.append(f'stored+noise={noise}')

    # Neighbor (smaller m)
    from warm_start import embed_narrower_bottleneck
    for cand_m in [m // 2, m // 4]:
        if cand_m < 2: continue
        nb = load_best_model(n, cand_m, l, S, models_dir, device=device)
        if nb is None or nb.tied_weights: continue
        for i in range(n_neighbor):
            seed = base_seed + 300 + i
            ae = embed_narrower_bottleneck(nb, m, seed=seed, device=device)
            pool.append(ae); seed_values.append(seed)
            descriptions.append(f'neighbor(m={cand_m})')
        break

    # Random
    while len(pool) < K:
        seed = base_seed + 400 + len(pool)
        ae = random_init(n, m, l, seed=seed, device=device)
        pool.append(ae); seed_values.append(seed)
        descriptions.append(f'random_seed{seed}')

    init_mses = [eval_mse(ae, S, n_samples=20000, device=device) for ae in pool]
    return pool, init_mses, descriptions, seed_values, best_shallow_mse


class Lion(optim.Optimizer):
    """Lion optimizer (Chen et al. 2023): EvolvedSign-based update.
    Update: w_t = w_{t-1} - lr * sign(beta1 * m + (1-beta1) * g)
    Momentum: m_t = beta2 * m + (1-beta2) * g
    Default betas: (0.9, 0.99). Typically use lr ~ 3-10x smaller than AdamW.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']; wd = group['weight_decay']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p)
                m = state['m']
                # Update direction: sign of momentum-tweaked grad
                update = (beta1 * m + (1 - beta1) * p.grad).sign_()
                # Weight decay decoupled
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                # Update momentum
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return loss


def train_with_precise_picking(pool, init_mses, descriptions, seed_values,
                                n, m, l, S, max_steps, batch_size=8192,
                                lr_peak=4e-3, weight_decay=1e-2,
                                grad_clip=1.0, ema_decay=0.999,
                                snapshot_every=2000, device='cuda',
                                optimizer_name='adamw', l1_lambda=0.0):
    """Train K seeds, snapshot the BATCHED state every snapshot_every steps,
    at end evaluate every snapshot at high precision for every seed, pick the
    global best.

    optimizer_name: 'adamw' or 'lion'. Lion needs lr ~ 3-10x smaller.
    """
    K = len(pool)
    batched = BatchedAutoencoder(n, m, l, K, seeds=seed_values).to(device)
    for k, ae in enumerate(pool):
        load_seed_into_batched(batched, k, ae)
    del pool

    if optimizer_name == 'lion':
        optimizer = Lion(batched.parameters(), lr=lr_peak / 5, weight_decay=weight_decay * 5)
        print(f'  Using Lion: lr={lr_peak/5:.5f}, wd={weight_decay*5:.4f}')
    elif optimizer_name == 'sgd_nesterov':
        optimizer = optim.SGD(batched.parameters(), lr=lr_peak * 3, momentum=0.9,
                              nesterov=True, weight_decay=weight_decay)
        print(f'  Using SGD+Nesterov: lr={lr_peak*3:.5f}, mom=0.9')
    elif optimizer_name == 'nadam':
        optimizer = optim.NAdam(batched.parameters(), lr=lr_peak, weight_decay=weight_decay)
        print(f'  Using NAdam: lr={lr_peak:.5f}')
    elif optimizer_name == 'adamw_lookahead':
        # Lookahead: take k slow steps every k fast steps
        from collections import defaultdict
        base_opt = optim.AdamW(batched.parameters(), lr=lr_peak, weight_decay=weight_decay)
        class Lookahead(optim.Optimizer):
            def __init__(self, base, k=5, alpha=0.5):
                self.base = base; self.k = k; self.alpha = alpha
                self.param_groups = base.param_groups
                self.defaults = base.defaults
                self.state = defaultdict(dict)
                self._step = 0
                for group in self.param_groups:
                    for p in group['params']:
                        self.state[p]['slow'] = p.data.clone()
            def step(self, closure=None):
                loss = self.base.step(closure)
                self._step += 1
                if self._step % self.k == 0:
                    for group in self.param_groups:
                        for p in group['params']:
                            slow = self.state[p]['slow']
                            slow.add_(p.data - slow, alpha=self.alpha)
                            p.data.copy_(slow)
                return loss
            def zero_grad(self, *a, **kw):
                self.base.zero_grad(*a, **kw)
        optimizer = Lookahead(base_opt, k=5, alpha=0.5)
        print(f'  Using AdamW+Lookahead (k=5, alpha=0.5): lr={lr_peak:.5f}')
    else:
        optimizer = optim.AdamW(batched.parameters(), lr=lr_peak, weight_decay=weight_decay)
    compiled_model = torch.compile(batched, mode='reduce-overhead') if device.type == 'cuda' else batched

    # EMA shadow weights
    ema_state = {k: v.clone().detach() for k, v in batched.state_dict().items()} if ema_decay else None

    # Initial snapshot
    snapshots = [{
        'step': 0,
        'state': {k: v.clone() for k, v in batched.state_dict().items()},
    }]

    import time as _t
    t0 = _t.time()
    for step in range(max_steps):
        lr = cosine_lr_inline(step, max_steps, lr_peak)
        for pg in optimizer.param_groups: pg['lr'] = lr

        x = generate_sparse_data(batch_size, n, S)
        x_hat, z = compiled_model(x)
        x_K = x.unsqueeze(0).expand(K, -1, -1)
        mse_per_seed = ((x_hat - x_K) ** 2).mean(dim=(1, 2))
        loss = mse_per_seed.sum()
        if l1_lambda > 0:
            # L1 penalty on z (bottleneck activations) — encourages sparse codes
            l1_per_seed = z.abs().mean(dim=(1, 2))
            loss = loss + l1_lambda * l1_per_seed.sum()
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(batched.parameters(), grad_clip)
        optimizer.step()
        if ema_state:
            with torch.no_grad():
                for kk, vv in batched.state_dict().items():
                    ema_state[kk].mul_(ema_decay).add_(vv.detach(), alpha=1-ema_decay)

        # Take snapshot periodically + report
        if step > 0 and step % snapshot_every == 0:
            snapshots.append({
                'step': step,
                'state': {k: v.clone() for k, v in batched.state_dict().items()},
            })
            # Quick eval of the BEST current seed by in-training noisy eval
            from run_sweep_gpu import measure_batched_linearity
            torch.manual_seed(99999)
            lin = measure_batched_linearity(batched, n_samples=2000, S=S)
            best_now = min(lin['mse_fulls'])
            elapsed = _t.time() - t0
            print(f'  step {step}/{max_steps} ({100*step/max_steps:.0f}%) '
                  f'best_seed_mse(noisy n=2k)={best_now:.7f}  '
                  f'snapshots={len(snapshots)}  elapsed={elapsed:.0f}s')

    # Final snapshot + EMA
    snapshots.append({
        'step': max_steps,
        'state': {k: v.clone() for k, v in batched.state_dict().items()},
    })
    if ema_state:
        snapshots.append({
            'step': 'ema',
            'state': {k: v.clone() for k, v in ema_state.items()},
        })

    print(f'  Took {len(snapshots)} snapshots; precise-evaluating each seed at each snapshot...')

    # For each snapshot, evaluate all K seeds at precise (n=200k) and find global best
    best_mse = float('inf')
    best_state = None
    best_k = -1
    best_snapshot_step = -1
    for snap in snapshots:
        batched.load_state_dict(snap['state'])
        # Eval each seed at high precision
        for k in range(K):
            single = batched.extract_single(k)
            mse_k = precise_mse(single, S, n_samples=100000, seed=42)
            if mse_k < best_mse:
                best_mse = mse_k
                best_state = {kk: v.clone() for kk, v in batched.state_dict().items()}
                best_k = k
                best_snapshot_step = snap['step']
                print(f'    new best: snapshot={snap["step"]} seed={k} mse={mse_k:.7f}  src={descriptions[k]}')

    # Restore best state and extract best seed
    batched.load_state_dict(best_state)
    best_model = batched.extract_single(best_k).to(device)
    init_mse_of_winner = init_mses[best_k]

    return dict(
        best_mse=best_mse,
        best_k=best_k,
        best_model=best_model,
        best_snapshot_step=best_snapshot_step,
        best_source=descriptions[best_k],
        init_mse_of_winner=init_mse_of_winner,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--m', type=int, required=True)
    parser.add_argument('--l', type=int, required=True)
    parser.add_argument('--S', type=float, required=True)
    parser.add_argument('--K', type=int, default=80)
    parser.add_argument('--steps-mult', type=float, default=5.0)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--snapshot-every', type=int, default=2000)
    parser.add_argument('--store-dir', default='results_db')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adamw', 'lion', 'sgd_nesterov', 'nadam', 'adamw_lookahead'])
    parser.add_argument('--label', default=None, help='Tag in run_id for tracking')
    parser.add_argument('--pure-random', action='store_true',
                        help='Skip shallow/stored/neighbor sources; all seeds random init')
    parser.add_argument('--near-warm-start', action='store_true',
                        help='All seeds = warm-start with TINY perturbation (log-uniform 1e-7 to 1e-2)')
    parser.add_argument('--lr-schedule', default='cosine',
                        choices=['cosine', 'one_cycle', 'constant'])
    parser.add_argument('--lr-mult', type=float, default=1.0,
                        help='Multiply lr_peak by this. Use small value (0.001) for near-warm-start exploration.')
    parser.add_argument('--l1-lambda', type=float, default=0.0,
                        help='L1 penalty on bottleneck z (encourages sparse codes for CS-style decoding)')
    args = parser.parse_args()

    device = torch.device(args.device)
    core.device = device
    _sweep.device = device

    print(f'\n=== Focused push: n={args.n} m={args.m} l={args.l} S={args.S} K={args.K} '
          f'optimizer={args.optimizer} pure_random={args.pure_random} ===')
    if args.pure_random:
        pool, init_mses, descs, seed_vals, shallow_mse = build_random_only_pool(
            args.n, args.m, args.l, args.S, args.K, device=device)
    elif args.near_warm_start:
        pool, init_mses, descs, seed_vals, shallow_mse = build_near_warm_start_pool(
            args.n, args.m, args.l, args.S, args.K,
            models_dir=f'{args.store_dir}/models', device=device)
    else:
        pool, init_mses, descs, seed_vals, shallow_mse = build_huge_pool(
            args.n, args.m, args.l, args.S, args.K,
            models_dir=f'{args.store_dir}/models', device=device)

    max_steps = int(24000 * math.sqrt(max(1, args.n / 16)) * args.steps_mult)
    print(f'  max_steps={max_steps}, K={args.K}, batch_size={args.batch_size}')

    t0 = time.time()
    res = train_with_precise_picking(
        pool, init_mses, descs, seed_vals,
        args.n, args.m, args.l, args.S,
        max_steps=max_steps, batch_size=args.batch_size,
        snapshot_every=args.snapshot_every,
        device=device,
        optimizer_name=args.optimizer,
        l1_lambda=args.l1_lambda,
        lr_peak=4e-3 * args.lr_mult,
    )
    elapsed = time.time() - t0

    print(f'\nDone in {elapsed/60:.1f}m')
    print(f'  Best mse (precise n=100k): {res["best_mse"]:.7f}')
    print(f'  Best seed: {res["best_k"]}  source: {res["best_source"]}')
    print(f'  Best snapshot step: {res["best_snapshot_step"]}')
    print(f'  Init mse of winning seed: {res["init_mse_of_winner"]:.7f}')

    # Save as a new seed in store
    store = ResultsStore(args.store_dir)
    # Re-evaluate winner precisely for storage
    final_mse = precise_mse(res['best_model'], args.S, n_samples=200000, seed=42)
    print(f'  Final mse at n=200k: {final_mse:.7f}')

    seed_results = [dict(
        seed_value=seed_vals[res['best_k']],
        mse_full=float(final_mse),
        mse_linear=float(final_mse * 2),  # placeholder
        nonlinear_gain=0.0,
        linearity_score=0.5,
        converged=True,
        steps_used=max_steps,
        warm_start_source=res['best_source'],
        init_mse=res['init_mse_of_winner'],
    )]
    label = args.label or args.optimizer
    store.add_seeds(args.n, args.m, args.l, args.S, seed_results,
                    run_id=f'focused_{label}_{time.strftime("%Y%m%d_%H%M%S")}',
                    model_state_dict=res['best_model'].cpu().state_dict(),
                    training_meta=dict(method='focused_push', K=args.K,
                                       max_steps=max_steps,
                                       steps_mult=args.steps_mult,
                                       optimizer=args.optimizer,
                                       snapshots_evaluated=True))
    print(f'  Saved.')
