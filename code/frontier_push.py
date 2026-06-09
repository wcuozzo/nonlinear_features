"""
Frontier-push: for a target (n, m, l, S), train K seeds from MULTIPLE init
sources to look for lower minima the chain-recipe missed.

Sources:
  1. Shallow embed (from best l-1 or l-2) with varied noise levels
  2. Stored OLD/NEW best model perturbed (refines current basin)
  3. Neighbor bottleneck projection (from (n, m_narrow, l))
  4. Fresh random inits (chance to find a basin no warm-start found)

All K seeds train simultaneously via BatchedAutoencoder with longer max_steps.

Usage:
    python frontier_push.py --configs 128,64,3,0.95 64,32,4,0.9 --K 30
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import core
import run_sweep_gpu as _sweep
from core import Autoencoder, generate_sparse_data
from results_store import ResultsStore
from run_sweep_gpu import BatchedAutoencoder, measure_batched_linearity
from sweep_violation_fix import (
    load_seed_into_batched, cosine_lr_inline,
    train_pool_batched, build_near_warm_start_pool,
)
from warm_start import (
    embed_shallow_in_deep,
    embed_narrower_bottleneck,
    perturb_model,
    random_init,
    eval_mse,
    load_best_model,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_diverse_pool(n, m, l, S, K, models_dir='results_db/models',
                        verbose=True):
    """Construct K initializations from mixed sources.

    Layout (for K=30):
        - 12 shallow embeds (varied noise 0.0, 0.001, 0.003, 0.01, 0.03, 0.1)
        - 6 stored-best perturbations (noise 0.001, 0.003, 0.01, 0.03)
        - 6 neighbor projections (smallest narrower m available, varied noise)
        - 6 fresh randoms
    """
    pool = []
    descriptions = []
    seed_values = []

    # ── Find best shallow source(s) across all l' < l ──
    best_shallow = None
    best_shallow_l = None
    best_shallow_mse = float('inf')
    for l_s in range(1, l):
        sh = load_best_model(n, m, l_s, S, models_dir, device=device)
        if sh is None:
            continue
        mse_s = eval_mse(sh, S, device=device)
        if mse_s < best_shallow_mse:
            best_shallow_mse = mse_s
            best_shallow = sh
            best_shallow_l = l_s

    # ── Find stored model at THIS (n, m, l, S) ──
    stored = load_best_model(n, m, l, S, models_dir, device=device)
    stored_mse = eval_mse(stored, S, device=device) if stored is not None else float('inf')

    # ── Find best neighbor (smaller m at same l, S) ──
    neighbor = None
    neighbor_m = None
    for m_cand in [m // 2, m // 4, 2]:
        if m_cand >= m or m_cand < 2:
            continue
        cand = load_best_model(n, m_cand, l, S, models_dir, device=device)
        if cand is not None and not cand.tied_weights:
            neighbor = cand
            neighbor_m = m_cand
            break

    if verbose:
        print(f'  shallow: l={best_shallow_l} mse={best_shallow_mse:.5f}')
        print(f'  stored:  mse={stored_mse:.5f}')
        print(f'  neighbor: m={neighbor_m}')

    # ── Allocate seeds per source ──
    base_seed = hash((n, m, l, S)) % 1000000
    n_shallow = min(12, K // 2) if best_shallow is not None else 0
    n_stored = min(6, max(0, (K - n_shallow) // 3)) if stored is not None else 0
    n_neighbor = min(6, max(0, (K - n_shallow - n_stored) // 2)) if neighbor is not None else 0
    n_random = K - n_shallow - n_stored - n_neighbor

    # 1) SHALLOW EMBEDS
    shallow_noises = [0.0, 0.0, 0.001, 0.001, 0.003, 0.003, 0.01, 0.01, 0.03, 0.03, 0.1, 0.1]
    for i in range(n_shallow):
        noise = shallow_noises[i % len(shallow_noises)]
        seed = base_seed + 100 + i
        ae = embed_shallow_in_deep(best_shallow, l, noise=noise, seed=seed, device=device)
        pool.append(ae)
        descriptions.append(f'shallow(l={best_shallow_l})+noise={noise}')
        seed_values.append(seed)

    # 2) STORED PERTURBATIONS
    stored_noises = [0.001, 0.003, 0.01, 0.03]
    for i in range(n_stored):
        noise = stored_noises[i % len(stored_noises)]
        seed = base_seed + 200 + i
        ae = perturb_model(stored, noise=noise, seed=seed, device=device)
        pool.append(ae)
        descriptions.append(f'stored+noise={noise}')
        seed_values.append(seed)

    # 3) NEIGHBOR PROJECTION
    for i in range(n_neighbor):
        seed = base_seed + 300 + i
        ae = embed_narrower_bottleneck(neighbor, m, seed=seed, device=device)
        pool.append(ae)
        descriptions.append(f'neighbor(m={neighbor_m})_seed{i}')
        seed_values.append(seed)

    # 4) RANDOM
    for i in range(n_random):
        seed = base_seed + 400 + i
        ae = random_init(n, m, l, seed=seed, device=device)
        pool.append(ae)
        descriptions.append(f'random_seed{seed}')
        seed_values.append(seed)

    init_mses = [eval_mse(ae, S, device=device) for ae in pool]
    return pool, init_mses, descriptions, seed_values, best_shallow_mse


def train_diverse_pool(pool, init_mses, descriptions, seed_values,
                       n, m, l, S, max_steps, batch_size=8192,
                       lr_peak=4e-3, weight_decay=1e-2,
                       floor_check_every=1000,
                       grad_clip=None, ema_decay=None):
    """Train K seeds (from diverse pool) simultaneously, with floor enforcement."""
    K = len(pool)

    # Pack into BatchedAutoencoder
    batched = BatchedAutoencoder(n, m, l, K, seeds=seed_values).to(device)
    for k, ae in enumerate(pool):
        load_seed_into_batched(batched, k, ae)
    del pool  # free Autoencoder copies

    optimizer = optim.AdamW(batched.parameters(), lr=lr_peak,
                             weight_decay=weight_decay)
    compiled_model = (torch.compile(batched, mode='reduce-overhead')
                      if device.type == 'cuda' else batched)

    # Initialize floor at warm-start
    best_mse_per_seed = list(init_mses)
    best_overall_mse = float(min(init_mses))
    best_state_dict = {k: v.clone() for k, v in batched.state_dict().items()}
    losses_log = []

    # Optional EMA of weights
    ema_state = None
    if ema_decay is not None:
        ema_state = {k: v.clone().detach() for k, v in batched.state_dict().items()}

    for step in range(max_steps):
        lr = cosine_lr_inline(step, max_steps, lr_peak)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = generate_sparse_data(batch_size, n, S)
        x_hat, _ = compiled_model(x)
        x_K = x.unsqueeze(0).expand(K, -1, -1)
        mse_per_seed = ((x_hat - x_K) ** 2).mean(dim=(1, 2))
        loss = mse_per_seed.sum()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(batched.parameters(), grad_clip)
        optimizer.step()

        # EMA update
        if ema_state is not None:
            with torch.no_grad():
                for k, v in batched.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)

        if step % 400 == 0:
            losses_log.append((step, mse_per_seed.detach().cpu().tolist()))

        if step > 0 and step % floor_check_every == 0:
            torch.manual_seed(99999)
            lin = measure_batched_linearity(batched, n_samples=2000, S=S)
            for k in range(K):
                if lin['mse_fulls'][k] < best_mse_per_seed[k]:
                    best_mse_per_seed[k] = lin['mse_fulls'][k]
            cur_best = min(lin['mse_fulls'])
            if cur_best < best_overall_mse:
                best_overall_mse = cur_best
                best_state_dict = {k: v.clone() for k, v in batched.state_dict().items()}

    # Floor enforcement + EMA evaluation
    torch.manual_seed(99999)
    final_lin = measure_batched_linearity(batched, n_samples=2000, S=S)
    # If using EMA, also check whether EMA weights give better MSE
    if ema_state is not None:
        cur_state = {k: v.clone() for k, v in batched.state_dict().items()}
        batched.load_state_dict(ema_state)
        torch.manual_seed(99999)
        ema_lin = measure_batched_linearity(batched, n_samples=2000, S=S)
        if min(ema_lin['mse_fulls']) < min(final_lin['mse_fulls']):
            final_lin = ema_lin  # EMA is better, keep its weights loaded
        else:
            batched.load_state_dict(cur_state)  # restore point estimate
    if best_state_dict is not None:
        if min(final_lin['mse_fulls']) > best_overall_mse:
            batched.load_state_dict(best_state_dict)
            torch.manual_seed(99999)
            final_lin = measure_batched_linearity(batched, n_samples=2000, S=S)

    final_mses = final_lin['mse_fulls']
    best_k = int(np.argmin(final_mses))
    best_mse = float(final_mses[best_k])
    best_model = batched.extract_single(best_k).to(device)

    return dict(
        final_mses=final_mses,
        final_gains=final_lin['gains'],
        final_lin_scores=final_lin['linearity_scores'],
        final_mse_linears=final_lin['mse_linears'],
        best_k=best_k,
        best_mse=best_mse,
        best_model=best_model,
        descriptions=descriptions,
        init_mses=init_mses,
        seed_values=seed_values,
        losses_log=losses_log,
    )


def get_steps(n, multiplier=1.0):
    """Longer than fix sweep: 2x base. Optional multiplier for longer-train experiments."""
    return int(24000 * math.sqrt(max(1, n / 16)) * multiplier)


def push_one_config(n, m, l, S, K, store, batch_size=8192, verbose=True,
                    steps_mult=1.0, grad_clip=None, ema_decay=None,
                    near_warm_start_K=None, near_warm_start_lr_mult=0.3,
                    lr_peak=4e-3):
    """Push frontier on one (n, m, l, S).

    Two arms:
      - DIVERSE-POOL (K seeds): existing mix of shallow/stored/neighbor/random,
        trained at lr_peak. Broad coverage of init basins.
      - NEAR-WARM-START (near_warm_start_K seeds, default max(4, K // 4)):
        zero-noise embed of best shallow + log-uniform tiny perturbation 1e-7..1e-2,
        trained at lr_peak * near_warm_start_lr_mult. Refines without leaving
        the basin. Set near_warm_start_K=0 to disable.
    """
    if near_warm_start_K is None:
        near_warm_start_K = max(4, K // 4)

    if verbose:
        print(f'\n=== {n}, {m}, {l}, {S} (K={K}, nws_K={near_warm_start_K}, '
              f'steps_mult={steps_mult}, grad_clip={grad_clip}, ema={ema_decay}) ===')

    pool, init_mses, descriptions, seed_values, shallow_mse = build_diverse_pool(
        n, m, l, S, K, verbose=verbose)

    max_steps = get_steps(n, multiplier=steps_mult)
    if verbose:
        print(f'  max_steps={max_steps}, K={K}, batch_size={batch_size}')

    # ── ARM 1: diverse pool at full LR ────────────────────────────────
    t0 = time.time()
    res_diverse = train_diverse_pool(pool, init_mses, descriptions, seed_values,
                             n, m, l, S, max_steps=max_steps,
                             batch_size=batch_size,
                             lr_peak=lr_peak,
                             grad_clip=grad_clip, ema_decay=ema_decay)
    elapsed_diverse = time.time() - t0

    # ── ARM 2: near-warm-start at reduced LR (optional) ──────────────
    res_nws = None
    nws_descriptions = None
    nws_init_mses = None
    elapsed_nws = 0.0
    if near_warm_start_K > 0:
        # Find best shallow source
        best_shallow = None; best_shallow_l = None; best_shallow_mse = float('inf')
        for l_s in range(1, l):
            sh = load_best_model(n, m, l_s, S, device=device)
            if sh is None: continue
            mse_s = eval_mse(sh, S, device=device)
            if mse_s < best_shallow_mse:
                best_shallow_mse = mse_s; best_shallow = sh; best_shallow_l = l_s
        if best_shallow is not None:
            if verbose:
                print(f'  [nws] best shallow l={best_shallow_l} mse={best_shallow_mse:.6f} '
                      f'-> {near_warm_start_K} seeds at lr={lr_peak * near_warm_start_lr_mult:.5f}')
            ae_nws, init_nws, seed_nws, desc_nws, _noises = build_near_warm_start_pool(
                best_shallow, l, n, m, S, near_warm_start_K,
                base_seed_tag=(n, m, l, S, 'frontier_nws'))
            t1 = time.time()
            res_nws = train_pool_batched(
                ae_nws, init_nws, seed_nws,
                n, m, l, S, max_steps=max_steps,
                batch_size=batch_size,
                lr_peak=lr_peak * near_warm_start_lr_mult,
                grad_clip=grad_clip, ema_decay=ema_decay,
            )
            elapsed_nws = time.time() - t1
            nws_descriptions = desc_nws
            nws_init_mses = init_nws
            del ae_nws
        else:
            if verbose:
                print(f'  [nws] no shallow source found, skipping arm 2')

    # ── Save seed_results from both arms ──────────────────────────────
    run_id = f'frontier_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    seed_results = []
    for k in range(K):
        per_seed_curve = [(s, vs[k]) for s, vs in res_diverse['losses_log'][::3]]
        seed_results.append(dict(
            seed_value=res_diverse['seed_values'][k],
            mse_full=float(res_diverse['final_mses'][k]),
            mse_linear=float(res_diverse['final_mse_linears'][k]),
            nonlinear_gain=float(res_diverse['final_gains'][k]),
            linearity_score=float(res_diverse['final_lin_scores'][k]),
            converged=True,
            steps_used=max_steps,
            loss_curve=per_seed_curve,
            init_mse=res_diverse['init_mses'][k],
            warm_start_source=res_diverse['descriptions'][k],
            warm_start_arm='diverse_pool',
            arm_lr_peak=lr_peak,
        ))
    if res_nws is not None:
        for k in range(len(res_nws['final_mses'])):
            per_seed_curve = [(s, vs[k]) for s, vs in res_nws['losses_log'][::3]]
            seed_results.append(dict(
                seed_value=res_nws['seed_values'][k],
                mse_full=float(res_nws['final_mses'][k]),
                mse_linear=float(res_nws['final_mse_linears'][k]),
                nonlinear_gain=float(res_nws['final_gains'][k]),
                linearity_score=float(res_nws['final_lin_scores'][k]),
                converged=True,
                steps_used=max_steps,
                loss_curve=per_seed_curve,
                init_mse=nws_init_mses[k],
                warm_start_source=nws_descriptions[k],
                warm_start_arm='near_warm_start',
                arm_lr_peak=lr_peak * near_warm_start_lr_mult,
            ))

    # ── Pick global best across arms ─────────────────────────────────
    if res_nws is not None and res_nws['best_mse'] < res_diverse['best_mse']:
        best_res = res_nws
        winning_arm = 'near_warm_start'
        winning_desc = nws_descriptions[res_nws['best_k']]
    else:
        best_res = res_diverse
        winning_arm = 'diverse_pool'
        winning_desc = res_diverse['descriptions'][res_diverse['best_k']]

    training_meta = dict(
        method='frontier_push',
        K=K,
        near_warm_start_K=near_warm_start_K,
        near_warm_start_lr_mult=near_warm_start_lr_mult,
        max_steps=max_steps,
        batch_size=batch_size,
        elapsed_sec=elapsed_diverse + elapsed_nws,
        winning_arm=winning_arm,
    )

    store.add_seeds(n, m, l, S, seed_results,
                    run_id=run_id,
                    model_state_dict=best_res['best_model'].cpu().state_dict(),
                    training_meta=training_meta)

    # Source breakdown of final MSEs (diverse pool only — nws is uniform source)
    if verbose:
        from collections import defaultdict
        by_source = defaultdict(list)
        for desc, mse in zip(res_diverse['descriptions'], res_diverse['final_mses']):
            kind = desc.split('(')[0].split('+')[0].split('_')[0]
            by_source[kind].append(mse)
        for kind, mses in by_source.items():
            print(f'  {kind:10s}: min={min(mses):.6f} median={float(np.median(mses)):.6f} '
                  f'max={max(mses):.6f}  (n={len(mses)})')
        if res_nws is not None:
            nws_mses = list(res_nws['final_mses'])
            print(f'  {"near_nws":10s}: min={min(nws_mses):.6f} median={float(np.median(nws_mses)):.6f} '
                  f'max={max(nws_mses):.6f}  (n={len(nws_mses)})')
        print(f'  Overall best: {best_res["best_mse"]:.6f} '
              f'(arm={winning_arm}, source={winning_desc})')
        print(f'  Elapsed: diverse={elapsed_diverse:.1f}s nws={elapsed_nws:.1f}s')

    return best_res['best_mse']


def _worker(gpu_id, task_queue, result_queue, K, batch_size, store_dir,
            steps_mult=1.0, grad_clip=None, ema_decay=None,
            near_warm_start_K=None, near_warm_start_lr_mult=0.3,
            master_seed=42):
    global device
    device = torch.device(f'cuda:{gpu_id}')
    core.device = device
    _sweep.device = device
    torch.cuda.set_device(gpu_id)
    torch.manual_seed(master_seed + gpu_id)
    np.random.seed(master_seed + gpu_id)
    import sweep_violation_fix as _svf
    _svf.device = device

    store = ResultsStore(store_dir)
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, n, m, l, S = item
        try:
            best_mse = push_one_config(n, m, l, S, K, store,
                                       batch_size=batch_size, verbose=True,
                                       steps_mult=steps_mult,
                                       grad_clip=grad_clip,
                                       ema_decay=ema_decay,
                                       near_warm_start_K=near_warm_start_K,
                                       near_warm_start_lr_mult=near_warm_start_lr_mult)
            result_queue.put((idx, n, m, l, S, best_mse, None))
        except Exception as e:
            import traceback
            result_queue.put((idx, n, m, l, S, None, str(e) + '\n' + traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='results_db')
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--n-gpus', type=int, default=8)
    parser.add_argument('--configs', nargs='+', required=True,
                        help='List of n,m,l,S e.g. 128,64,3,0.95 64,32,4,0.9')
    parser.add_argument('--steps-mult', type=float, default=1.0,
                        help='Multiply max_steps by this factor (3.0 = train 3x longer)')
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='Clip gradient norm to this value (e.g. 1.0). None = no clipping.')
    parser.add_argument('--ema-decay', type=float, default=None,
                        help='EMA decay rate for Polyak weight averaging (e.g. 0.999). None = off.')
    parser.add_argument('--near-warm-start-K', type=int, default=None,
                        help='Additional near-warm-start seeds: zero-noise embed of '
                             'best shallow + log-uniform tiny perturbation 1e-7..1e-2, '
                             'trained at lr_peak * lr_mult. Default: max(4, K // 4). '
                             'Set to 0 to disable.')
    parser.add_argument('--near-warm-start-lr-mult', type=float, default=0.3,
                        help='LR multiplier for the near-warm-start arm. Default 0.3.')
    parser.add_argument('--master-seed', type=int, default=42)
    args = parser.parse_args()

    configs = []
    for c in args.configs:
        n, m, l, S = c.split(',')
        configs.append((int(n), int(m), int(l), float(S)))

    print(f'Frontier-push on {len(configs)} configs with K={args.K}')

    import torch.multiprocessing as mp
    ctx = mp.get_context('spawn')
    tq = ctx.Queue()
    rq = ctx.Queue()
    for idx, (n, m, l, S) in enumerate(configs):
        tq.put((idx, n, m, l, S))
    for _ in range(args.n_gpus):
        tq.put(None)

    workers = []
    for gid in range(args.n_gpus):
        p = ctx.Process(target=_worker,
                        args=(gid, tq, rq, args.K, args.batch_size, args.store_dir,
                              args.steps_mult, args.grad_clip, args.ema_decay,
                              args.near_warm_start_K, args.near_warm_start_lr_mult,
                              args.master_seed))
        p.start()
        workers.append(p)

    completed = 0
    start = time.time()
    while completed < len(configs):
        if sum(1 for w in workers if w.is_alive()) == 0 and completed < len(configs):
            print(f'Workers exited at {completed}/{len(configs)}')
            break
        try:
            idx, n, m, l, S, mse, err = rq.get(timeout=1200)
            completed += 1
            elapsed = (time.time() - start) / 60
            if err:
                print(f'[{completed}/{len(configs)}] FAIL n={n} m={m} l={l} S={S}: {err[:200]}')
            else:
                print(f'[{completed}/{len(configs)}] n={n} m={m} l={l} S={S}: best_mse={mse:.6f} [total {elapsed:.1f}m]')
        except Exception:
            continue
    for w in workers:
        w.join(timeout=30)
    print(f'\nDone in {(time.time()-start)/60:.1f}m')
