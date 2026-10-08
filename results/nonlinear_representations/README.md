# Nonlinear representations

The main figures use one selected 216-configuration sweep, recorded in
[canonical_source.json](canonical_source.json). That sweep was trained with an earlier version
of the training scripts: its l = 1 models used a fixed 8,000 steps, and most selected deeper
models used 72,000·√(n/16) steps (the `steps_used` column of
[sweep_results.csv](sweep_results/compiled/sweep_results.csv)). The included
code is the current implementation of the procedure, so a new run can differ in learned
weights and fitted coefficients.

[canonical_analysis.py](canonical_analysis.py) produces the README's MSE heatmap,
scaling-law fits, feature trajectories, speeds, and decoder regions.
[architecture_figure.py](architecture_figure.py) produces the architecture diagram.

See the [reproduction guide](../../REPRODUCING.md) for commands to check the
saved results, regenerate the figures, and train new models.
