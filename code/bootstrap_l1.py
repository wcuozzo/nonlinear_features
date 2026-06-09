"""
Bootstrap missing l=1 model files.

The original sweep generated l=1 seed metrics but never saved the actual .pt
model weights for many configs. Without them, sweep_violation_fix.py can't
warm-start chains for those configs.

This script:
  1. For each (n, m, S) config that has l=1 seed data but no l=1 model file,
     trains a tied l=1 autoencoder from random init for a small number of steps.
  2. Picks best of K=8 seeds, saves the model.
  3. Optionally adds the new seed results to the store (additive).

Usage:
    python bootstrap_l1.py --store-dir results_db --K 8 --n-gpus 8
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import core
from core import Autoencoder, generate_sparse_data
from results_store import ResultsStore

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


def cosine_lr(step, max_steps, peak=4e-3, warmup=500, lr_min=1e-6):
    warmup = min(warmup, max_steps // 10)
    if step < warmup:
        return peak * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return lr_min + 0.5 * (peak - lr_min) * (1 + math.cos(math.pi * progress))


def eval_mse(model, S, n_samples=8000, seed=99999):
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, S)
        x_hat, _ = model(x)
        return ((x_hat - x) ** 2).mean().item()


def train_l1(n, m, S, seed, max_steps=8000, batch_size=4096):
    torch.manual_seed(seed)
    model = Autoencoder(n, m, l=1, tied_weights=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-2)

    best_mse = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for step in range(max_steps):
        lr = cosine_lr(step, max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = generate_sparse_data(batch_size, n, S)
        x_hat, _ = model(x)
        loss = ((x_hat - x) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step > 0 and step % 1000 == 0:
            cur = eval_mse(model, S)
            if cur < best_mse:
                best_mse = cur
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, eval_mse(model, S)


def find_missing(store_dir):
    """Return list of (n, m, S) needing l=1 model bootstrap."""
    seeds_dir = Path(store_dir) / 'seeds'
    models_dir = Path(store_dir) / 'models'
    missing = []
    for path in sorted(seeds_dir.glob('*_l1_*.json')):
        data = json.load(open(path))
        cfg = data['config']
        n, m, S = cfg['n'], cfg['m'], cfg['S']
        model_path = models_dir / f'model_n{n}_m{m}_l1_S{S}.pt'
        if not model_path.exists():
            missing.append((n, m, S))
    return missing


def bootstrap_config(n, m, S, K, store, batch_size=4096):
    """Train K seeds of l=1, save best."""
    seeds = [hash((n, m, 1, S, k)) % 100000 for k in range(K)]
    best_mse = float('inf')
    best_model = None
    seed_results = []
    for k, s in enumerate(seeds):
        model, mse = train_l1(n, m, S, seed=s, batch_size=batch_size)
        seed_results.append(dict(
            seed_value=s, mse_full=mse, mse_linear=mse,
            nonlinear_gain=0.0, linearity_score=1.0,
            converged=True, steps_used=8000,
        ))
        if mse < best_mse:
            best_mse = mse
            best_model = model

    store.add_seeds(n, m, 1, S, seed_results,
                    run_id=f'bootstrap_l1_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}',
                    model_state_dict=best_model.cpu().state_dict() if best_model else None,
                    training_meta=dict(method='bootstrap_l1', K=K))
    return best_mse


def _gpu_worker(gpu_id, task_queue, result_queue, K, batch_size, store_dir,
                master_seed=42):
    global device
    import core as _core
    device = torch.device(f'cuda:{gpu_id}')
    _core.device = device
    torch.manual_seed(master_seed + gpu_id)
    store = ResultsStore(store_dir)
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, n, m, S = item
        try:
            t0 = time.time()
            mse = bootstrap_config(n, m, S, K, store, batch_size)
            result_queue.put((idx, n, m, S, mse, time.time() - t0))
        except Exception as e:
            import traceback
            result_queue.put((idx, n, m, S, None, str(e) + '\n' + traceback.format_exc()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='results_db')
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--n-gpus', type=int, default=1)
    parser.add_argument('--master-seed', type=int, default=42)
    args = parser.parse_args()

    missing = find_missing(args.store_dir)
    print(f'Missing l=1 models: {len(missing)}')
    for n, m, S in missing:
        print(f'  n={n} m={m} S={S}')
    if not missing:
        return

    start = time.time()

    if args.n_gpus > 1:
        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')
        tq = ctx.Queue()
        rq = ctx.Queue()
        for idx, (n, m, S) in enumerate(missing):
            tq.put((idx, n, m, S))
        for _ in range(args.n_gpus):
            tq.put(None)
        workers = []
        for gid in range(args.n_gpus):
            p = ctx.Process(target=_gpu_worker,
                            args=(gid, tq, rq, args.K, args.batch_size, args.store_dir,
                                  args.master_seed))
            p.start()
            workers.append(p)
        completed = 0
        while completed < len(missing):
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0 and completed < len(missing):
                print('Workers exited early')
                break
            try:
                idx, n, m, S, mse, info = rq.get(timeout=600)
                completed += 1
                if mse is None:
                    print(f'[{completed}/{len(missing)}] FAIL n={n} m={m} S={S}: {info}')
                else:
                    print(f'[{completed}/{len(missing)}] n={n} m={m} S={S}: mse={mse:.5f} ({info:.1f}s)')
            except Exception:
                continue
        for w in workers:
            w.join(timeout=30)
    else:
        store = ResultsStore(args.store_dir)
        for i, (n, m, S) in enumerate(missing):
            t0 = time.time()
            mse = bootstrap_config(n, m, S, args.K, store, args.batch_size)
            print(f'[{i+1}/{len(missing)}] n={n} m={m} S={S}: mse={mse:.5f} ({time.time()-t0:.1f}s)')

    print(f'Bootstrap done in {(time.time()-start)/60:.1f}m')
    store = ResultsStore(args.store_dir)
    store.compile()


if __name__ == '__main__':
    main()
