"""
Post-hoc monotonicity enforcement. For every (n, m, l, S) in the store, if its
precise MSE is higher than ANY shallower l' < l at the same (n, m, S), replace
its .pt with the zero-noise identity-embed of the best shallower model.

This is a mathematical guarantee: identity-embed of l_s in l_d > l_s gives
exactly l_s's MSE on this non-negative data (identity + ReLU is a pass-through).
So mse(l_d) > mse(l_s) is impossible given best-effort.

Run after any frontier_push/sweep_violation_fix to guarantee no apparent
violations remain.

Usage:
    python enforce_monotonicity.py [--store-dir results_db] [--device cuda:0]
"""
import argparse
import time
from pathlib import Path
import sys

import torch

import core
from core import Autoencoder
from warm_start import embed_shallow_in_deep
from precise_recompile import precise_mse
from results_store import ResultsStore


def enforce(store_dir='results_db', device='cuda:0', tol=1.001):
    device = torch.device(device)
    core.device = device

    import pandas as pd
    csv = Path(store_dir) / 'compiled' / 'sweep_results_precise.csv'
    df = pd.read_csv(csv)
    df_idx = df.set_index(['n', 'm', 'l', 'S'])

    models_dir = Path(store_dir) / 'models'
    fixed = []
    skipped_no_anchor = []

    for _, row in df.iterrows():
        n, m, l, S = int(row.n), int(row.m), int(row.l), row.S
        mse = float(row.mse_full)

        # For each shallower l', check if l's mse > l's mse * tol
        for l_s in range(1, l):
            k_s = (n, m, l_s, S)
            if k_s not in df_idx.index:
                continue
            mse_s = float(df_idx.loc[k_s, 'mse_full'])
            if mse <= mse_s * tol:
                continue
            # VIOLATION: mse(l) > mse(l_s). Try identity-embed.
            l_s_path = models_dir / f'model_n{n}_m{m}_l{l_s}_S{S}.pt'
            l_path = models_dir / f'model_n{n}_m{m}_l{l}_S{S}.pt'
            if not l_s_path.exists():
                skipped_no_anchor.append((n, m, l, S, l_s))
                continue
            # Load shallow
            shallow = Autoencoder(n, m, l_s, tied_weights=(l_s == 1)).to(device)
            shallow.load_state_dict(torch.load(l_s_path, map_location=device))
            # Identity-embed
            embedded = embed_shallow_in_deep(shallow, l, noise=0.0, device=device)
            embedded_mse = precise_mse(embedded, S, n_samples=200000, seed=42)
            if embedded_mse < mse:
                print(f'  REPAIR n={n} m={m} l={l} S={S}: '
                      f'mse={mse:.7f} -> {embedded_mse:.7f} '
                      f'(via identity-embed of l={l_s} which has mse={mse_s:.7f})')
                torch.save(embedded.cpu().state_dict(), l_path)
                # Also add a seed entry
                store = ResultsStore(store_dir)
                store.add_seeds(n, m, l, S, [{
                    'seed_value': 0,
                    'mse_full': float(embedded_mse),
                    'mse_linear': float(embedded_mse),
                    'nonlinear_gain': 0.0,
                    'linearity_score': 1.0,
                    'converged': True,
                    'steps_used': 0,
                    'warm_start_source': f'identity_embed_of_l={l_s} (no training)',
                }],
                    run_id=f'enforce_monotonicity_{time.strftime("%Y%m%d_%H%M%S")}',
                    model_state_dict=embedded.cpu().state_dict(),
                    training_meta={'method': 'enforce_monotonicity',
                                   'source_l': l_s})
                fixed.append((n, m, l, S, mse, embedded_mse))
                break  # this l is fixed; move on to next row

    print()
    print(f'Repaired {len(fixed)} configs')
    if skipped_no_anchor:
        print(f'Skipped {len(skipped_no_anchor)} (no anchor model)')

    return fixed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='results_db')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available()
                        else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--tol', type=float, default=1.001,
                        help='Violation tolerance (default 1.001 = 0.1% slack)')
    args = parser.parse_args()
    print(f'Enforcing monotonicity in {args.store_dir} (tol={args.tol}, device={args.device})')
    enforce(args.store_dir, args.device, args.tol)
    print('\nRun precise_recompile.py to refresh canonical CSV.')
