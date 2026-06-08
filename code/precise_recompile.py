"""
Re-evaluate every saved .pt model on a fixed deterministic high-precision
sample, then write a new compiled CSV with these accurate MSEs.

Motivation: the compiled CSV's mse_full per config is the mse measured at
training time on whatever sample/seed combo was used. Different configs may
have used different samples → noisy comparisons. At MSE level ~0.01,
n_samples=2000 has ~1% std; small "violations" may be pure eval noise.

This script:
  - Loads each .pt model
  - Evals on a single fixed (seed, n_samples=200000) sample, per S
  - Writes results_db/compiled/sweep_results_precise.csv
  - Optionally re-runs find_violations against the precise CSV
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import core
from core import Autoencoder, generate_sparse_data

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


def precise_mse(model, S, n_samples=200000, seed=42):
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        # Process in chunks to fit memory at high n_samples
        chunk = 50000
        total_sq = 0.0
        total_n = 0
        for start in range(0, n_samples, chunk):
            cs = min(chunk, n_samples - start)
            x = generate_sparse_data(cs, model.n, S)
            x_hat, _ = model(x)
            total_sq += ((x_hat - x) ** 2).sum().item()
            total_n += cs * model.n
    return total_sq / total_n


def precise_compile(store_dir='results_db', n_samples=200000, seed=42):
    core.device = device
    models_dir = Path(store_dir) / 'models'

    rows = []
    files = sorted(models_dir.glob('model_*.pt'))
    for path in tqdm(files, desc='Re-eval'):
        stem = path.stem  # model_n128_m2_l4_S0.95
        parts = stem.split('_')
        # parts: ['model','n128','m2','l4','S0.95']
        n = int(parts[1][1:])
        m = int(parts[2][1:])
        l = int(parts[3][1:])
        S = float(parts[4][1:])

        model = Autoencoder(n, m, l, tied_weights=(l == 1)).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        mse = precise_mse(model, S, n_samples=n_samples, seed=seed)
        rows.append(dict(n=n, m=m, l=l, S=S, mse_full=mse))

    df = pd.DataFrame(rows).sort_values(['n', 'm', 'l', 'S']).reset_index(drop=True)

    # Compute two clean metrics measuring how much depth helps over the linear
    # baseline (the actual trained l=1 model at same n, m, S — NOT a linear
    # approximation of a deeper model, which would unfairly include decoder
    # mismatch in the linear baseline).
    #   nonlinear_mse_decrease = mse_l1 - mse_l                (absolute drop)
    #   pct_nonlinear_mse_decrease = (mse_l1 - mse_l) / mse_l1 (fractional drop)
    # By construction both are 0 at l=1.
    l1_lookup = df[df.l == 1].set_index(['n', 'm', 'S'])['mse_full'].to_dict()
    def absolute_decrease(row):
        key = (int(row.n), int(row.m), float(row.S))
        if key not in l1_lookup:
            return float('nan')
        return l1_lookup[key] - row.mse_full
    def pct_decrease(row):
        key = (int(row.n), int(row.m), float(row.S))
        if key not in l1_lookup:
            return float('nan')
        mse_l1 = l1_lookup[key]
        if mse_l1 <= 0:
            return float('nan')
        return (mse_l1 - row.mse_full) / mse_l1
    df['nonlinear_mse_decrease'] = df.apply(absolute_decrease, axis=1)
    df['pct_nonlinear_mse_decrease'] = df.apply(pct_decrease, axis=1)

    out = Path(store_dir) / 'compiled' / 'sweep_results_precise.csv'
    df.to_csv(out, index=False)
    print(f'Wrote {out} ({len(df)} configs)')
    return df


def find_violations_precise(df):
    """Same shape as find_violations but on the precise CSV."""
    df_idx = df.set_index(['n', 'm', 'l', 'S'])
    n_vals = sorted(df.n.unique())
    m_vals = sorted(df.m.unique())
    S_vals = sorted(df.S.unique())

    violations = {}
    for _, row in df.iterrows():
        n, m, l, S = int(row.n), int(row.m), int(row.l), row.S
        mse = row.mse_full

        for l2 in range(int(l) + 1, 5):
            k = (n, m, l2, S)
            if k in df_idx.index:
                mse2 = float(df_idx.loc[k, 'mse_full'])
                if mse2 > mse * 1.001:
                    if k not in violations or mse < violations[k]['mse_target']:
                        violations[k] = dict(type='depth',
                                             mse_target=mse, mse_current=mse2,
                                             gap=mse2 / mse,
                                             shallow_l=int(l))
        # Other axes (bottleneck, input_dim, sparsity) likewise:
        mi = m_vals.index(m)
        for mi2 in range(mi + 1, len(m_vals)):
            m2 = m_vals[mi2]
            if m2 >= n:
                continue
            k = (n, m2, l, S)
            if k in df_idx.index:
                mse2 = float(df_idx.loc[k, 'mse_full'])
                if mse2 > mse * 1.001:
                    if k not in violations or mse < violations[k]['mse_target']:
                        violations[k] = dict(type='bottleneck',
                                             mse_target=mse, mse_current=mse2,
                                             gap=mse2 / mse,
                                             shallow_l=int(l))
        ni = n_vals.index(n)
        for ni2 in range(0, ni):
            n2 = n_vals[ni2]
            if m >= n2:
                continue
            k = (n2, m, l, S)
            if k in df_idx.index:
                mse2 = float(df_idx.loc[k, 'mse_full'])
                if mse2 > mse * 1.001:
                    if k not in violations or mse < violations[k]['mse_target']:
                        violations[k] = dict(type='input_dim',
                                             mse_target=mse, mse_current=mse2,
                                             gap=mse2 / mse,
                                             shallow_l=int(l))
        si = S_vals.index(S)
        for si2 in range(si + 1, len(S_vals)):
            S2 = S_vals[si2]
            k = (n, m, l, S2)
            if k in df_idx.index:
                mse2 = float(df_idx.loc[k, 'mse_full'])
                if mse2 > mse * 1.001:
                    if k not in violations or mse < violations[k]['mse_target']:
                        violations[k] = dict(type='sparsity',
                                             mse_target=mse, mse_current=mse2,
                                             gap=mse2 / mse,
                                             shallow_l=int(l))

    return pd.DataFrame(
        [dict(n=k[0], m=k[1], l=k[2], S=k[3], **v) for k, v in violations.items()]
    ).sort_values('gap', ascending=False).reset_index(drop=True) if violations else pd.DataFrame()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='results_db')
    parser.add_argument('--n-samples', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
        core.device = device
    print(f'Device: {device}')

    t0 = time.time()
    df = precise_compile(args.store_dir, args.n_samples, args.seed)
    print(f'Recompiled in {time.time()-t0:.1f}s')

    viols = find_violations_precise(df)
    print(f'\nPrecise violations: {len(viols)}')
    if not viols.empty:
        # Print by axis
        for ax in ['depth', 'bottleneck', 'input_dim', 'sparsity']:
            sub = viols[viols['type'] == ax]
            if len(sub):
                print(f'\n  {ax} ({len(sub)}):')
                print(sub.head(10).to_string())
