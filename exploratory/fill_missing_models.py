"""
Fill in missing .pt model files by running the progressive chain on specified
(n, m, S) groups. Wraps fix_group from sweep_violation_fix.

Usage:
    python fill_missing_models.py --n-gpus 8 --K 10
"""
import argparse
import os
import time

import torch
import torch.multiprocessing as mp

import core
import run_sweep_gpu as _sweep
from results_store import ResultsStore
from sweep_violation_fix import fix_group


# 10 (n, m, S) groups all missing l=2,3,4
MISSING_GROUPS = [
    (64, 2, 0.85), (64, 4, 0.85), (64, 4, 0.9), (64, 4, 0.95),
    (128, 2, 0.85), (128, 2, 0.95), (128, 4, 0.85), (128, 4, 0.9), (128, 4, 0.95),
    (128, 8, 0.85),
]


def _worker(gpu_id, task_queue, result_queue, K, batch_size, store_dir):
    device = torch.device(f'cuda:{gpu_id}')
    core.device = device
    _sweep.device = device
    torch.cuda.set_device(gpu_id)
    import sweep_violation_fix as _svf
    _svf.device = device

    store = ResultsStore(store_dir)
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, n, m, S = item
        try:
            t0 = time.time()
            # max_l=4, no specific "violated_ls" — pass [2,3,4] to force full chain training
            res = fix_group(n, m, S, max_l=4, K=K, violated_ls=[2, 3, 4],
                            store=store, batch_size=batch_size,
                            models_dir=os.path.join(store_dir, 'models'),
                            verbose=True)
            res['elapsed_sec'] = time.time() - t0
            res['gpu_id'] = gpu_id
            result_queue.put((idx, res))
        except Exception as e:
            import traceback
            result_queue.put((idx, dict(n=n, m=m, S=S, error=str(e),
                                        traceback=traceback.format_exc())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='results_db')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--n-gpus', type=int, default=8)
    args = parser.parse_args()

    print(f'Filling missing models for {len(MISSING_GROUPS)} groups')
    for n, m, S in MISSING_GROUPS:
        print(f'  n={n} m={m} S={S} (l=2,3,4)')

    ctx = mp.get_context('spawn')
    tq = ctx.Queue()
    rq = ctx.Queue()
    for idx, (n, m, S) in enumerate(MISSING_GROUPS):
        tq.put((idx, n, m, S))
    for _ in range(args.n_gpus):
        tq.put(None)

    workers = []
    for gid in range(args.n_gpus):
        p = ctx.Process(target=_worker,
                        args=(gid, tq, rq, args.K, args.batch_size, args.store_dir))
        p.start()
        workers.append(p)

    completed = 0
    start = time.time()
    while completed < len(MISSING_GROUPS):
        if sum(1 for w in workers if w.is_alive()) == 0 and completed < len(MISSING_GROUPS):
            print(f'Workers exited at {completed}/{len(MISSING_GROUPS)}')
            break
        try:
            idx, res = rq.get(timeout=1800)
            completed += 1
            el = (time.time() - start) / 60
            if 'error' in res:
                print(f'[{completed}/{len(MISSING_GROUPS)}] FAIL n={res["n"]} m={res["m"]} S={res["S"]}: {res["error"]}')
            else:
                stages = '  '.join(f'l={l}:{mse:.5f}' for l, mse in res['stage_results'].items())
                print(f'[{completed}/{len(MISSING_GROUPS)}] n={res["n"]} m={res["m"]} S={res["S"]} GPU{res.get("gpu_id","?")}: {stages}  [total {el:.1f}m]')
        except Exception:
            continue
    for w in workers:
        w.join(timeout=30)
    print(f'\nDone in {(time.time()-start)/60:.1f}m')
