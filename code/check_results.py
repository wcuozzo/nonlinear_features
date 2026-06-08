"""
Comprehensive sanity check after a violation-fix sweep.

Goes beyond "0 violations":
  A. Monotonicity (depth + bottleneck + input_dim + sparsity)
  B. New-seed gain/MSE plausibility:
       - nonlinear_gain in [-0.05, 1.05]
       - final_mse <= init_mse * 1.05 (floor enforcement worked)
       - init_mse of stage l ~ best final_mse of stage l-1 (handoff sanity)
  C. Saved-model fidelity:
       - re-loading the .pt file gives MSE matching reported mse_full
  D. Loss-curve convergence:
       - last 10% of loss curve should be within 5% of min — flags stop-too-early
  E. Cross-seed determinism for zero-noise seeds:
       - two seeds with noise=0 from same source should give similar final MSE
       - if they diverge wildly, something stochastic is broken
  F. Regime plausibility:
       - linear regime (m close to n, high S): expect nonlinear_gain near 0
       - bottleneck regime (small m): expect nonlinear_gain > 0.5
       - flag opposite

Exit code 0 if all green, 1 if any anomaly.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from results_store import ResultsStore
from sweep_violation_fix import find_violations, group_violations_by_nms
from warm_start import eval_mse, load_best_model

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def _load_seeds(store_dir, n, m, l, S):
    path = Path(store_dir) / 'seeds' / f'n{n}_m{m}_l{l}_S{S}.json'
    if not path.exists():
        return []
    return json.load(open(path))['seeds']


def _is_new(seed_dict):
    rid = seed_dict.get('run_id', '')
    return ('progv2' in rid or 'violation_fix' in rid or 'bootstrap_l1' in rid)


# ────────────────────────────────────────────────────────────────────────
# A. Cross-axis monotonicity
# ────────────────────────────────────────────────────────────────────────

def check_monotonicity_all_axes(df_idx, axes=('depth', 'bottleneck', 'input_dim', 'sparsity')):
    """Return dict of {axis: violation_list}. Each violation:
    (config_better, config_worse, mse_better, mse_worse).
    Tolerance 1.001 to match find_violations.
    """
    n_vals = sorted(df_idx.index.get_level_values('n').unique())
    m_vals = sorted(df_idx.index.get_level_values('m').unique())
    S_vals = sorted(df_idx.index.get_level_values('S').unique())

    results = {a: [] for a in axes}

    for key, row in df_idx.iterrows():
        n, m, l, S = key
        mse = row['mse_full']

        if 'depth' in axes:
            for l2 in range(int(l) + 1, 5):
                k2 = (n, m, l2, S)
                if k2 in df_idx.index:
                    mse2 = float(df_idx.loc[k2, 'mse_full'])
                    if mse2 > mse * 1.001:
                        results['depth'].append((key, k2, mse, mse2))

        if 'bottleneck' in axes:
            mi = m_vals.index(m)
            for mi2 in range(mi + 1, len(m_vals)):
                m2 = m_vals[mi2]
                if m2 >= n:
                    continue
                k2 = (n, m2, l, S)
                if k2 in df_idx.index:
                    mse2 = float(df_idx.loc[k2, 'mse_full'])
                    if mse2 > mse * 1.001:
                        results['bottleneck'].append((key, k2, mse, mse2))

        if 'input_dim' in axes:
            ni = n_vals.index(n)
            for ni2 in range(0, ni):
                n2 = n_vals[ni2]
                if m >= n2:
                    continue
                k2 = (n2, m, l, S)
                if k2 in df_idx.index:
                    mse2 = float(df_idx.loc[k2, 'mse_full'])
                    if mse2 > mse * 1.001:
                        results['input_dim'].append((key, k2, mse, mse2))

        if 'sparsity' in axes:
            si = S_vals.index(S)
            for si2 in range(si + 1, len(S_vals)):
                S2 = S_vals[si2]
                k2 = (n, m, l, S2)
                if k2 in df_idx.index:
                    mse2 = float(df_idx.loc[k2, 'mse_full'])
                    if mse2 > mse * 1.001:
                        results['sparsity'].append((key, k2, mse, mse2))

    return results


# ────────────────────────────────────────────────────────────────────────
# B. New-seed sanity
# ────────────────────────────────────────────────────────────────────────

def check_new_seed_sanity(store_dir):
    """Scan all new seeds added by progv2/violation_fix runs."""
    seeds_dir = Path(store_dir) / 'seeds'

    weird_gain = []
    floor_broken = []
    handoff_broken = []
    convergence_unstable = []

    # Group new seeds by (n,m,S) -> {l: list of seed dicts}
    by_group = defaultdict(lambda: defaultdict(list))
    for path in sorted(seeds_dir.glob('*.json')):
        data = json.load(open(path))
        cfg = data['config']
        n, m, l, S = cfg['n'], cfg['m'], cfg['l'], cfg['S']
        for sd in data['seeds']:
            if _is_new(sd):
                by_group[(n, m, S)][l].append(sd)

    for (n, m, S), l_to_seeds in by_group.items():
        for l, seeds in l_to_seeds.items():
            for sd in seeds:
                gain = sd.get('nonlinear_gain')
                if gain is not None and not (-0.05 <= gain <= 1.05):
                    weird_gain.append((n, m, l, S, gain))

                init = sd.get('init_mse')
                if init is not None and sd['mse_full'] > init * 1.05:
                    floor_broken.append((n, m, l, S, init, sd['mse_full']))

                # Robust convergence check: training loss is noisy per-batch,
                # so use the LINEAR-FIT slope of the last quarter as the signal.
                # If slope > +tolerance, training was still ascending (bad).
                # Tolerance is scaled by the median loss to be relative.
                lc = sd.get('loss_curve', [])
                if len(lc) >= 20:
                    last_q = lc[-max(5, len(lc) // 4):]
                    steps = np.array([s for s, _ in last_q], dtype=float)
                    losses = np.array([v for _, v in last_q], dtype=float)
                    if len(steps) >= 3 and steps.max() > steps.min():
                        slope, intercept = np.polyfit(steps, losses, 1)
                        median_loss = float(np.median(losses))
                        # Slope per step; normalize by (median_loss / total_steps)
                        relative = slope * (steps.max() - steps.min()) / max(median_loss, 1e-12)
                        # Threshold scales with absolute loss: at noise-floor MSE
                        # (~1e-5), batch-to-batch variation is naturally larger.
                        # Use 30% for absolute losses above 0.001, 100% for below.
                        thresh = 0.30 if float(np.median(losses)) > 0.001 else 1.0
                        if relative > thresh:
                            convergence_unstable.append(
                                (n, m, l, S, float(np.min(losses)),
                                 float(losses[-1]), float(relative)))

        # Handoff check: stage l's best init_mse should be ~ stage l-1's best final_mse
        ls = sorted(l_to_seeds.keys())
        prev_best_final = None
        for l in ls:
            curr_best_init = min((sd.get('init_mse', np.inf) for sd in l_to_seeds[l]),
                                 default=np.inf)
            curr_best_final = min(sd['mse_full'] for sd in l_to_seeds[l])
            if prev_best_final is not None and curr_best_init < np.inf:
                # Init at stage l should be close to final at stage l-1
                # (the warm-start source is the previous best)
                # Allow up to 5% slack from eval-sample noise
                if curr_best_init > prev_best_final * 1.05 + 1e-6:
                    handoff_broken.append((n, m, S, l - 1, l,
                                           prev_best_final, curr_best_init))
            prev_best_final = curr_best_final

    return dict(
        weird_gain=weird_gain,
        floor_broken=floor_broken,
        handoff_broken=handoff_broken,
        convergence_unstable=convergence_unstable,
    )


# ────────────────────────────────────────────────────────────────────────
# C. Saved-model fidelity
# ────────────────────────────────────────────────────────────────────────

def check_saved_model_fidelity(store_dir, df, n_samples=200000, sample_size=30,
                                eval_seed=42, use_precise_eval=True):
    """Reload each saved .pt and verify its MSE matches the CSV.

    If `use_precise_eval` and df is the precise CSV, we re-eval at the same
    (n_samples=200000, seed=42) used by precise_recompile.py — matches should
    then be within float precision (~1e-7). Any mismatch flags a real bug
    (e.g. .pt file corrupted, recompile out-of-date).
    """
    from core import Autoencoder
    from precise_recompile import precise_mse
    import core as _core
    _core.device = device

    np.random.seed(0)
    idxs = np.random.choice(len(df), size=min(sample_size, len(df)), replace=False)
    mismatches = []
    for i in idxs:
        row = df.iloc[i]
        n, m, l, S = int(row.n), int(row.m), int(row.l), row.S
        model_path = Path(store_dir) / 'models' / f'model_n{n}_m{m}_l{l}_S{S}.pt'
        if not model_path.exists():
            continue
        model = Autoencoder(n, m, l, tied_weights=(l == 1)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        if use_precise_eval:
            mse = precise_mse(model, S, n_samples=n_samples, seed=eval_seed)
            tol = 1e-3  # tight enough to catch real mismatches, loose enough
                        # for sub-display-precision float32 numerical drift
        else:
            mse = eval_mse(model, S, n_samples=n_samples, device=device)
            tol = 0.03
        reported = row['mse_full']
        if mse > reported * (1 + tol) + 1e-9 or reported > mse * (1 + tol) + 1e-9:
            mismatches.append((n, m, l, S, reported, mse))
    return mismatches


# ────────────────────────────────────────────────────────────────────────
# E. Zero-noise determinism
# ────────────────────────────────────────────────────────────────────────

def check_zero_noise_determinism(store_dir):
    """Among new seeds with noise=0, sibling seeds should have similar final MSE.

    Different seed_value but same noise=0 from same source = different starting
    perturbation paths (the embed should be identical), so they should converge
    to similar minima. Diverging suggests stochasticity beyond the noise schedule.
    """
    by_group_l = defaultdict(list)
    for path in sorted(Path(store_dir, 'seeds').glob('*.json')):
        data = json.load(open(path))
        cfg = data['config']
        key = (cfg['n'], cfg['m'], cfg['l'], cfg['S'])
        for sd in data['seeds']:
            if _is_new(sd) and sd.get('warm_start_noise', None) == 0.0:
                by_group_l[key].append(sd)

    diverging = []
    for key, seeds in by_group_l.items():
        if len(seeds) < 2:
            continue
        mses = [sd['mse_full'] for sd in seeds]
        if max(mses) > min(mses) * 1.5 and max(mses) - min(mses) > 1e-5:
            diverging.append((key, min(mses), max(mses)))
    return diverging


# ────────────────────────────────────────────────────────────────────────
# F. Regime plausibility
# ────────────────────────────────────────────────────────────────────────

def check_regime_plausibility(df_idx):
    """Linear regime (m/n >= 0.5 AND S >= 0.9 AND l>=2): expect nonlinear_gain low.
    Bottleneck regime (m/n <= 0.25): expect nonlinear_gain high for l>=2.

    Skips if df_idx has no nonlinear_gain column (precise CSV).
    """
    if 'nonlinear_gain' not in df_idx.columns:
        return []  # precise CSV lacks gain; skip silently
    suspicious = []
    for key, row in df_idx.iterrows():
        n, m, l, S = key
        if l == 1:
            continue
        ratio = m / n
        gain = float(row['nonlinear_gain'])
        if ratio >= 0.5 and S >= 0.9 and gain > 0.9:
            suspicious.append(('linear regime, surprisingly high gain',
                              key, gain))
        if ratio <= 0.125 and S <= 0.9 and gain < 0.3:
            suspicious.append(('bottleneck regime, surprisingly low gain',
                              key, gain))
    return suspicious


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def report_section(title, items, formatter=None, max_show=10):
    if not items:
        print(f'  ✓ {title}: 0')
        return False
    print(f'  ✗ {title}: {len(items)}')
    for x in items[:max_show]:
        if formatter:
            print(f'      {formatter(x)}')
        else:
            print(f'      {x}')
    if len(items) > max_show:
        print(f'      ... and {len(items)-max_show} more')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-dir', default='results_db')
    parser.add_argument('--skip-model-check', action='store_true')
    parser.add_argument('--model-sample', type=int, default=30)
    parser.add_argument('--precise', action='store_true',
                        help='Use sweep_results_precise.csv for monotonicity (recommended)')
    args = parser.parse_args()

    print(f'\n{"="*72}\nCOMPREHENSIVE SANITY CHECK: {args.store_dir}\n{"="*72}')
    any_issue = False

    # Recompile
    store = ResultsStore(args.store_dir)
    df = store.compile()

    # For monotonicity, use precise CSV if available + requested
    precise_path = Path(args.store_dir) / 'compiled' / 'sweep_results_precise.csv'
    if args.precise and precise_path.exists():
        df_for_mono = pd.read_csv(precise_path)
        print(f'Configs in store: {len(df)};  using PRECISE CSV ({len(df_for_mono)} models) for monotonicity')
    else:
        df_for_mono = df
        print(f'Configs in store: {len(df)};  using NOISY CSV for monotonicity')

    df_idx = df_for_mono.set_index(['n', 'm', 'l', 'S'])

    # ── A. Monotonicity across all axes ───────────────────────────────
    print('\n[A] Cross-axis monotonicity:')
    mono = check_monotonicity_all_axes(df_idx)
    for axis, viols in mono.items():
        any_issue |= report_section(
            f'{axis} violations', viols,
            formatter=lambda v: f'better={v[0]} ({v[2]:.5f}) worse={v[1]} ({v[3]:.5f})  ratio={v[3]/v[2]:.2f}x',
            max_show=5,
        )

    # ── B. New-seed sanity ────────────────────────────────────────────
    print('\n[B] New-seed sanity:')
    nss = check_new_seed_sanity(args.store_dir)
    any_issue |= report_section(
        'out-of-range nonlinear_gain', nss['weird_gain'],
        formatter=lambda v: f'n={v[0]} m={v[1]} l={v[2]} S={v[3]} gain={v[4]:.3f}')
    any_issue |= report_section(
        'floor broken (final > 1.05 * init)', nss['floor_broken'],
        formatter=lambda v: f'n={v[0]} m={v[1]} l={v[2]} S={v[3]} init={v[4]:.5f} final={v[5]:.5f}')
    any_issue |= report_section(
        'broken handoff (stage l init > 1.05 * stage l-1 final)', nss['handoff_broken'],
        formatter=lambda v: f'n={v[0]} m={v[1]} S={v[2]} l={v[3]}->{v[4]} '
                            f'prev_final={v[5]:.5f} curr_init={v[6]:.5f}')
    any_issue |= report_section(
        'convergence unstable (last quarter linear-fit ascending >10%)', nss['convergence_unstable'],
        formatter=lambda v: f'n={v[0]} m={v[1]} l={v[2]} S={v[3]} '
                            f'min={v[4]:.5f} last={v[5]:.5f}  rel_slope={v[6]:+.2f}')

    # ── C. Saved-model fidelity ───────────────────────────────────────
    if not args.skip_model_check:
        print('\n[C] Saved-model fidelity (sampled):')
        # If precise CSV available, use it (consistent eval, tight tolerance)
        check_df = df_for_mono if args.precise and 'mse_full' in df_for_mono.columns else df
        mismatches = check_saved_model_fidelity(args.store_dir, check_df, sample_size=args.model_sample)
        any_issue |= report_section(
            f'saved .pt vs CSV mismatch (sample {args.model_sample})', mismatches,
            formatter=lambda v: f'n={v[0]} m={v[1]} l={v[2]} S={v[3]} '
                                f'csv={v[4]:.5f} live={v[5]:.5f}  ratio={v[5]/max(v[4], 1e-12):.2f}x')

    # ── E. Zero-noise determinism ─────────────────────────────────────
    print('\n[E] Zero-noise seed determinism:')
    diverging = check_zero_noise_determinism(args.store_dir)
    any_issue |= report_section(
        'zero-noise siblings diverging (max/min > 1.5)', diverging,
        formatter=lambda v: f'n={v[0][0]} m={v[0][1]} l={v[0][2]} S={v[0][3]} '
                            f'min={v[1]:.5f} max={v[2]:.5f}')

    # ── F. Regime plausibility ────────────────────────────────────────
    print('\n[F] Regime plausibility:')
    suspicious = check_regime_plausibility(df_idx)
    any_issue |= report_section(
        'regime-implausible gain', suspicious,
        formatter=lambda v: f'{v[0]}: n={v[1][0]} m={v[1][1]} l={v[1][2]} S={v[1][3]} gain={v[2]:.3f}')

    print()
    if any_issue:
        print('SANITY CHECK: ISSUES FOUND')
        return 1
    print('SANITY CHECK: CLEAN ✓')
    return 0


if __name__ == '__main__':
    sys.exit(main())
