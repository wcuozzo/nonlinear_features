# Reproducing the results

The README presents three sets of results: nonlinear representations across encoder
depths, the mechanisms enabled by untied read and write weights, and computation in a
nonlinear code. Saved outputs are
included, so reading the results does not require a GPU or retraining.
Files and scripts named "canonical" refer to the single 216-configuration sweep used for
the README's figures.

## Install

Use Python 3.11 or newer. Tested with Python 3.13.5.
From this repository's root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements pin the versions used for the local checks. GPU-enabled PyTorch
installation may require a platform-specific wheel. The numerical checks and
figure regeneration below use CPU.

## Check the saved results

```sh
python scripts/verify_results.py
python -m unittest discover -s tests -p 'test*.py'
```

The first command loads all 216 selected checkpoints, checks their shapes and
finite weights, refits the displayed scaling law, and evaluates the pictured
tied and untied models on their recorded independent test draw. It performs no
training and needs no network access. The second runs all regression tests (about
10 seconds on a CPU): the sweep audit, launcher and resolver checks, a two-step run
of the pentagon notebook, and the shallow-figure export. It runs no training sweep,
and the reduced notebook run checks execution, not convergence or research results.

`scripts/check_results.py` audits new training stores and requires their seed
logs. For the bundled snapshot, use `scripts/verify_results.py` as shown above;
the training auditor stops without writing files when those logs are missing.

## Regenerate the main representation figures

```sh
python results/nonlinear_representations/canonical_analysis.py
python results/nonlinear_representations/architecture_figure.py
```

The first command regenerates the heatmap, scaling fit, feature trajectories,
speeds, and decoder-region figures under `figures/`, together with their
`canonical_analysis.json` provenance record. It overwrites those generated files
in this checkout. Pixel-level differences can occur across plotting environments.

## Regenerate the shallow-model diagrams

```sh
python scripts/export_shallow_figures.py
```

This recreates the five-feature read/write diagram, its three mechanism panels,
and the pentagon diagram from the included weights, without retraining. It writes
the five corresponding PNGs under `figures/`. To preview them elsewhere, pass
`--output-dir /tmp/shallow-figures`. The exporter preserves the saved geometry and
computes the displayed values from the weights; styling and image size can differ
slightly from the committed PNGs.

## Source notebooks for the shallow examples

```sh
python -m jupyter lab
```

- [How untying helps](results/tied_untied/how_untying_helps.ipynb): training setup for the five-feature untied example.
- [Toy Models replication](results/tied_untied/toy_models_sanity_check.ipynb): training and geometry checks for the tied pentagon example.

These two notebooks cover the tied vs. untied section. The main representation results are
produced by the training and analysis scripts described above. The
[weight-decay sweep](results/tied_untied/weight_decay_results/README.md) and
[unit-norm searches](results/tied_untied/one_sided_norm_results/REPORT.md)
also use standalone scripts.

Notebook outputs are retained. Executing the notebooks trains new models and can
take substantial time. The Toy Models notebook's first cell locates this checkout before
importing shared code, so it can be opened from its own folder or the repository root.

## Which data belong to which result?

| Files | Role |
|---|---|
| `results/nonlinear_representations/sweep_results/` | The 216-row all-untied sweep and matching checkpoints used by the main README figures. |
| `figures/untied_read_write_example.json` | Weights and evaluation details for the pictured five-feature example. |
| `results/tied_untied/*_results/` | Saved regularization and normalization searches. |
| `results/tied_untied/pentagon_n5_m2_l1_S0.95.pt` | Saved model for the pentagon reproduction. |
| `results/computation_in_nonlinear_codes/` | Trained weights, audits, and stage records for the computation example; its [README](results/computation_in_nonlinear_codes/README.md) lists them. |

The saved CSV and checkpoints reproduce the reported analysis. The saved sweep was
trained with an earlier version of the training scripts: its l = 1 models used a fixed
8,000 steps, and most selected deeper models used 72,000·√(n/16) steps (the `steps_used`
column of `results/nonlinear_representations/sweep_results/compiled/sweep_results.csv`).
The included code is the current implementation of the procedure and is intended
to reproduce comparable results; a new run can differ in learned weights, individual
losses, and fitted coefficients.

The shallow examples and regularization searches are numerical solutions, not
proofs of global optimality. Their sampling intervals do not measure uncertainty
over all possible optimization runs.

## Train a new sweep

The full sweep is substantial. Without `STORE_DIR`, the launcher writes to a new dated
folder under `new-runs/`; it refuses to write into an existing results store:

```sh
STORE_DIR=new-runs/my-sweep MASTER_SEED=42 N_GPUS=1 bash scripts/canonical_sweep.sh
```

For the script's smaller end-to-end smoke configuration:

```sh
SMOKE=1 STORE_DIR=new-runs/smoke N_GPUS=1 DEVICE=cpu bash scripts/canonical_sweep.sh
```

The smoke run checks the training workflow; its losses are not research results.
It takes about 5–6 minutes on a CPU. At smoke budgets the final audit usually reports
floor or handoff warnings and the script exits with status 1; that is expected and is
not a crash. Its retraining stage uses 4 diverse initializations plus up to 2 near-warm-start
initializations, with a 1,500-step base budget. `RESOLVE_K` and
`RESOLVE_STEPS_BASE` override the diverse-initialization count and step budget.
To analyze a new completed run, change `store` in
[canonical_source.json](results/nonlinear_representations/canonical_source.json)
and rerun the analysis. The launcher audits the run when it finishes; to audit it again, run
`python scripts/check_results.py --store-dir new-runs/my-sweep --precise`.

## Computation in nonlinear codes

The notebook checks every claim about the trained example and the counts in the
README's training table from the saved results. It uses only the Python standard
library and takes about a minute:

```sh
python -m jupyter nbconvert --to notebook --execute --inplace \
    results/computation_in_nonlinear_codes/computation_in_nonlinear_codes.ipynb
python results/computation_in_nonlinear_codes/architecture_figure.py
```

The second command redraws the network diagram under `figures/`.

To retrain from scratch, run these from `results/computation_in_nonlinear_codes/`.
They write into `results/` there, so move the supplied results aside or use a fresh
checkout first.

Seed 1's saved stage-1 record is a full-run record (`width-4-seed-1.json`, which also
includes an XOR circuit) rather than a compressor-only one; the notebook reads its
initial-autoencoder result from that file.

```sh
# 1. Initial autoencoders: seeds 0-9, best of 3 restarts each
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python width4_codec_experiment.py --seed "$seed" --compressor-only
done
# 2. XOR circuits on the two frozen encoders that reconstruct all 16 inputs (seeds 4 and 9)
python focused_stage2_experiment.py
# 3. Joint decoder refinement of the 13 qualifying circuits, then exact audits
python joint_decoder_refinement.py --output results/my_refinement
python audit_joint_reconstruction.py --folder results/my_refinement
```

`joint_refinement_initial_circuits.json` is the hand-off between stages 2 and 3: the
13 XOR circuits from stage 2 that keep an operand of each XOR unreadable. It is a
saved file rather than a script output; stage 3 checks that it reproduces those
circuits' original final codes to floating-point precision. Retrained models can differ from the saved
ones, so the counts in a new run need not match the README's.
