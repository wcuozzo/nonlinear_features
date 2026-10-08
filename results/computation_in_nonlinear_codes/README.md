# Computation in nonlinear codes

[computation_in_nonlinear_codes.ipynb](computation_in_nonlinear_codes.ipynb) checks every
claim about the trained example in the main README, using exact rational arithmetic on the
saved weights. It needs only the Python standard library and runs in about a minute. It
also checks the training-attempt counts in the README's table.
[architecture_figure.py](architecture_figure.py) draws the network diagram.

The training scripts in this folder regenerate the results from scratch in three stages:
the initial autoencoders, the XOR circuits on two frozen encoders, and the joint decoder
refinement. See the [reproduction guide](../../REPRODUCING.md) for commands.

## Scripts

| Script | Role |
|---|---|
| [width4_codec_experiment.py](width4_codec_experiment.py) | Stage 1: trains the 4→2→4 autoencoder for one seed (`--compressor-only`). |
| [focused_stage2_experiment.py](focused_stage2_experiment.py) | Stage 2: trains XOR circuits on the frozen encoders that reconstruct all 16 inputs. Needs the `.pt` files written by a stage-1 rerun. |
| [joint_decoder_refinement.py](joint_decoder_refinement.py) | Stage 3: joint decoder refinement of the 13 circuits in `joint_refinement_initial_circuits.json`. |
| [audit_joint_reconstruction.py](audit_joint_reconstruction.py) | Stage 3 audit: evaluates every refinement checkpoint in exact arithmetic and writes the certificates and summary. |
| [minimal_learned_example.py](minimal_learned_example.py) | Helper: model definitions, truth table, and training routines used by all stages. Run directly, it trains one small example. |
| [joint_reconstruction_experiment.py](joint_reconstruction_experiment.py) | Helper: circuit forward pass and weight export used by stage 3. Its own command-line sweep needs files that are not included. |
| [focused_stage2_decoder.py](focused_stage2_decoder.py) | Helper: decoder parameter packing and forward pass used by stage 3. Run directly, it fits decoders to stage-2 circuits and needs the `.pt` files from a stage-2 rerun. |
| [exact_focused_stage2_audit.py](exact_focused_stage2_audit.py) | Helper: exact rational forward pass used by the stage-3 audit. Run directly, it audits a decoder candidate from `focused_stage2_decoder.py`, so it also needs a stage-2 rerun. |
| [verify_width4_codec_experiment.py](verify_width4_codec_experiment.py) | Helper: certificate verifier used by the audits. Run directly, it rechecks the saved full-run record `results/width4_codec/width-4-seed-1.json`. |
| [architecture_figure.py](architecture_figure.py) | Draws the network diagram (`figures/fig_trained_xor_architecture.png`) from the layer sizes of the selected example. |

`results/joint_decoder_refinement/training-source.py` is the exact script that produced the
refinement run; its SHA-256 matches `source_sha256` in that folder's `protocol.json`. The
published `joint_decoder_refinement.py` differs from it only in comments, its docstring and final
log message, and in checking the starting codes to float32 precision instead of bitwise, so that it
runs on other PyTorch builds.

## Data

| Files | Role |
|---|---|
| `results/joint_decoder_refinement/` | The refinement study behind the README's example: protocol, all trained weights, the audited summary, and the selected example (`selected-example-exact.json`). Certificate files are included for the three β = 1 runs that meet every requirement (the README's example uses β = 1) and for one frozen β = 0 control per source circuit. The saved files store β as `lambda`. |
| `results/focused_stage2/` | XOR training on the frozen encoders: 51/128 successful circuits and their numerical readability screen. |
| `results/width4_codec/` | Initial autoencoder seeds 0–9; seed 1 is stored as its full-run record. |
| `joint_refinement_initial_circuits.json` | The 13 XOR circuits passed from the XOR stage to the refinement study. |
