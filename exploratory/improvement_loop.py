"""
Heartbeat improvement loop. Each invocation:
  1. Picks the next idea from `improvement_backlog.md`
  2. Selects a target subset of configs (most "promising" given the idea)
  3. Runs the improvement
  4. Triggers a precise_recompile + sanity check
  5. Logs to RUN_NOTES.md (what was tried, whether it helped, by how much)
  6. Marks the idea consumed in backlog (or "ongoing" if reseeding)

Ideas implemented (cycles through these on consecutive invocations):
  - "more_seeds": K=60 multi-source on top-variance configs
  - "bigger_batch": batch_size=32k on configs that converge slowly
  - "longer_train": 3x max_steps on configs still descending
  - "warm_restart": SGDR-style cosine with restarts
  - "polyak_ema": EMA of weights instead of point estimate
  - "sam": Sharpness-aware minimization (SAM optimizer step)
  - "lion": Lion optimizer instead of AdamW
  - "anti_collapse": penalty on small feature norms during training
  - "fresh_random": pure random init (no warm-start), K=50 — sometimes random finds something warm-start misses
  - "depth_5_test": try l=5 architectures on configs where l=4 improvement was big
"""

import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


BACKLOG_FILE = Path('improvement_backlog.md')
NOTES_FILE = Path('RUN_NOTES.md')


# Default backlog of ideas
DEFAULT_BACKLOG = """\
# Improvement Backlog

Each line is one idea. Lines starting with `x ` are consumed.

- more_seeds           K=60 multi-source on top-variance configs
- fresh_random         pure random init K=50, longer training (no warm-start)
- bigger_batch         batch_size=32k on slow-converging configs
- longer_train         3x max_steps on configs still descending
- polyak_ema           EMA of weights instead of point estimate
- lion                 Lion optimizer instead of AdamW
- anti_collapse        penalty on small feature norms during training
- sam                  Sharpness-aware minimization step
- warm_restart         SGDR cosine with restarts
- depth_5_test         try l=5 on configs where l=4 had big improvement
"""


def ensure_backlog():
    if not BACKLOG_FILE.exists():
        BACKLOG_FILE.write_text(DEFAULT_BACKLOG)


def pick_next_idea():
    """Return the next un-consumed idea from the backlog."""
    ensure_backlog()
    lines = BACKLOG_FILE.read_text().splitlines()
    for line in lines:
        if line.startswith('- '):
            return line[2:].strip().split(None, 1)
    return None


def mark_consumed(idea_key, note=''):
    """Mark idea as consumed in backlog."""
    lines = BACKLOG_FILE.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f'- {idea_key}'):
            lines[i] = f'x {idea_key}  ({note})'
            break
    BACKLOG_FILE.write_text('\n'.join(lines) + '\n')


def append_to_notes(text):
    NOTES_FILE.touch(exist_ok=True)
    with NOTES_FILE.open('a') as f:
        f.write(text)


def get_top_variance_configs(store_dir, top_k=12):
    """Find configs where seed MSE varied widely — most likely to improve with more seeds."""
    seeds_dir = Path(store_dir) / 'seeds'
    candidates = []
    for path in seeds_dir.glob('*.json'):
        data = json.load(open(path))
        cfg = data['config']
        if cfg['l'] == 1:
            continue
        seeds = data['seeds']
        if len(seeds) < 10:
            continue
        mses = [s['mse_full'] for s in seeds]
        if min(mses) < 1e-7:
            continue
        ratio = max(mses) / min(mses)
        # Prefer m/n in middle range and not yet at the noise floor
        candidates.append(dict(n=cfg['n'], m=cfg['m'], l=cfg['l'], S=cfg['S'],
                               min_mse=min(mses), ratio=ratio))
    candidates.sort(key=lambda c: -c['ratio'])
    return candidates[:top_k]


def get_high_loss_configs(store_dir, top_k=10):
    """Find configs with highest current MSE — most room to improve."""
    precise = pd.read_csv(Path(store_dir) / 'compiled' / 'sweep_results_precise.csv')
    # Exclude l=1 (linear, can't improve much)
    precise = precise[precise.l > 1]
    return [dict(n=int(r.n), m=int(r.m), l=int(r.l), S=float(r.S), mse=float(r.mse_full))
            for _, r in precise.nlargest(top_k, 'mse_full').iterrows()]


# ────────────────────────────────────────────────────────────────────────
# Strategies
# ────────────────────────────────────────────────────────────────────────

def strategy_more_seeds(store_dir, n_gpus):
    """Run frontier_push with K=60 on top-variance configs."""
    configs = get_top_variance_configs(store_dir, top_k=12)
    if not configs:
        return 'no candidates', None
    args = ' '.join(f"{c['n']},{c['m']},{c['l']},{c['S']}" for c in configs)
    cmd = (f"python3 frontier_push.py --n-gpus {n_gpus} --K 60 "
           f"--batch-size 8192 --configs {args}")
    return cmd, configs


def strategy_fresh_random(store_dir, n_gpus):
    """Try purely random init with K=50 on high-loss configs.
    Implemented by frontier_push but with build_diverse_pool weighted toward
    random_init. The current frontier_push only includes 4-6 random seeds out
    of K=30. We'll instead just bump K significantly and let the random share
    grow proportionally. (Or use --K 100 to amplify.)
    """
    configs = get_high_loss_configs(store_dir, top_k=10)
    if not configs:
        return 'no candidates', None
    args = ' '.join(f"{c['n']},{c['m']},{c['l']},{c['S']}" for c in configs)
    cmd = (f"python3 frontier_push.py --n-gpus {n_gpus} --K 100 "
           f"--batch-size 8192 --configs {args}")
    return cmd, configs


def strategy_bigger_batch(store_dir, n_gpus):
    """K=30, batch_size=32k. Bigger batch = lower-variance gradient = potentially better minimum."""
    configs = get_top_variance_configs(store_dir, top_k=10)
    args = ' '.join(f"{c['n']},{c['m']},{c['l']},{c['S']}" for c in configs)
    cmd = (f"python3 frontier_push.py --n-gpus {n_gpus} --K 30 "
           f"--batch-size 32768 --configs {args}")
    return cmd, configs


# Map of strategy keys to functions
STRATEGIES = {
    'more_seeds': strategy_more_seeds,
    'fresh_random': strategy_fresh_random,
    'bigger_batch': strategy_bigger_batch,
}


def snapshot_mse(store_dir, configs):
    """Get current best MSE per target config (from compiled CSV)."""
    df = pd.read_csv(Path(store_dir) / 'compiled' / 'sweep_results_precise.csv')
    snap = {}
    for c in configs:
        sub = df[(df.n == c['n']) & (df.m == c['m']) & (df.l == c['l']) & (df.S == c['S'])]
        if len(sub):
            snap[(c['n'], c['m'], c['l'], c['S'])] = float(sub.mse_full.iloc[0])
    return snap


def compare_snapshots(before, after, configs):
    rows = []
    for c in configs:
        key = (c['n'], c['m'], c['l'], c['S'])
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            continue
        rows.append(dict(n=c['n'], m=c['m'], l=c['l'], S=c['S'],
                         before=b, after=a, ratio=a / b if b > 0 else 1.0))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='results_db')
    parser.add_argument('--n-gpus', type=int, default=8)
    parser.add_argument('--strategy', default=None,
                        help='Override the next-idea pick')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'== improvement loop @ {timestamp} ==')

    # 1. Pick idea
    if args.strategy:
        idea_key = args.strategy
        idea_desc = STRATEGIES.get(idea_key).__doc__ or ''
    else:
        idea = pick_next_idea()
        if not idea:
            print('No ideas left in backlog. Stopping.')
            return
        idea_key = idea[0]
        idea_desc = idea[1] if len(idea) > 1 else ''
    print(f'Idea: {idea_key} — {idea_desc}')

    if idea_key not in STRATEGIES:
        msg = (f'\n## {timestamp} — idea: `{idea_key}`\n'
               f'**SKIPPED** — strategy not yet implemented.\n')
        append_to_notes(msg)
        mark_consumed(idea_key, 'not impl')
        return

    cmd, configs = STRATEGIES[idea_key](args.store_dir, args.n_gpus)
    if configs is None or not configs:
        msg = f'\n## {timestamp} — idea: `{idea_key}`\nNo candidates. Skipped.\n'
        append_to_notes(msg)
        mark_consumed(idea_key, 'no candidates')
        return

    print(f'Running: {cmd}')
    print(f'Configs ({len(configs)}): ' + ', '.join(f'{c["n"]},{c["m"]},{c["l"]},{c["S"]}' for c in configs))

    # 2. Snapshot before
    before = snapshot_mse(args.store_dir, configs)

    # 3. Run it
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0

    if ret.returncode != 0:
        msg = f'\n## {timestamp} — idea: `{idea_key}`\n**FAILED** (return code {ret.returncode})\n'
        append_to_notes(msg)
        mark_consumed(idea_key, 'failed')
        return

    # 4. Precise recompile + sanity
    subprocess.run('python3 precise_recompile.py --device cuda:0', shell=True)
    sanity = subprocess.run('python3 check_results.py --precise',
                            shell=True, capture_output=True, text=True)

    after = snapshot_mse(args.store_dir, configs)
    delta = compare_snapshots(before, after, configs)

    # 5. Log
    n_better = (delta['ratio'] < 0.99).sum() if len(delta) else 0
    n_worse = (delta['ratio'] > 1.01).sum() if len(delta) else 0
    median_ratio = delta['ratio'].median() if len(delta) else 1.0
    best_improvement = (1 - delta['ratio'].min()) if len(delta) else 0.0

    sanity_pass = 'CLEAN' in sanity.stdout
    msg = (f'\n## {timestamp} — idea: `{idea_key}`\n'
           f'- Configs targeted: {len(configs)}\n'
           f'- Elapsed: {elapsed/60:.1f}m\n'
           f'- Improved (>1% lower MSE): {n_better}\n'
           f'- Worsened (>1% higher MSE): {n_worse}\n'
           f'- Median ratio (after/before): {median_ratio:.4f}\n'
           f'- Best single improvement: {best_improvement*100:.1f}% lower MSE\n'
           f'- Sanity check: {"PASS" if sanity_pass else "ISSUES"}\n')
    if len(delta):
        msg += '\nTop 3 wins:\n'
        msg += delta.nsmallest(3, 'ratio')[['n','m','l','S','before','after','ratio']].to_string(index=False) + '\n'
    append_to_notes(msg)

    print(msg)
    print(f'Sanity output (tail):\n{sanity.stdout[-1000:]}')

    mark_consumed(idea_key, f'{n_better} improved, median ratio {median_ratio:.3f}')


if __name__ == '__main__':
    main()
