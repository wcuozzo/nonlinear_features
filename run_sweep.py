"""Run metric prediction sweep and save results to CSV."""
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from core import run_experiment

# Sweep grid
n_values = [16, 32, 64, 128, 256]
m_values = [2, 4, 8, 16, 32, 64]
l_values = [1, 2, 3, 4]
S_values = [0.85, 0.9, 0.95]
n_seeds = 3
n_steps = 5000

configs = [(n, m, l, S)
           for n in n_values for m in m_values if m < n
           for l in l_values for S in S_values]

print(f"Configs: {len(configs)}, x{n_seeds} seeds = {len(configs) * n_seeds} runs")

all_results = []

for n, m, l, S in tqdm(configs, desc="Sweep"):
    seed_gains = []
    seed_losses = []
    seed_linearities = []

    for seed in range(n_seeds):
        torch.manual_seed(seed * 1000 + hash((n, m, l)) % 1000)
        np.random.seed(seed * 1000 + hash((n, m, l)) % 1000)

        res = run_experiment(
            n=n, m=m, l=l, S=S,
            n_steps=n_steps,
            tied_weights=(l == 1),
            verbose=False
        )
        seed_gains.append(res['nonlinear_gain'])
        seed_losses.append(res['final_loss'])
        seed_linearities.append(res['linearity_score'])

    best_idx = np.argmin(seed_losses)

    all_results.append({
        'n': n, 'm': m, 'l': l, 'S': S,
        'nonlinear_gain': seed_gains[best_idx],
        'linearity_score': seed_linearities[best_idx],
        'final_loss': seed_losses[best_idx],
        'gain_mean': np.mean(seed_gains),
        'gain_std': np.std(seed_gains),
        'loss_mean': np.mean(seed_losses),
    })

df = pd.DataFrame(all_results)
df.to_csv('metric_prediction_data.csv', index=False)
print(f"\nSaved {len(df)} rows to metric_prediction_data.csv")
print(f"Nonlinear gain range: [{df['nonlinear_gain'].min():.4f}, {df['nonlinear_gain'].max():.4f}]")
print(f"Mean gain std across seeds: {df['gain_std'].mean():.4f}")
