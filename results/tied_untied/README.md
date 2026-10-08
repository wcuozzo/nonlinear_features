# Tied vs. untied weights

The data and reconstruction objective follow the [nonlinear representations](../nonlinear_representations/README.md)
section. This section changes the relation between encoder and decoder weights, primarily
at l = 1; both tied and untied encoders are linear at that depth.

Two source notebooks are included for the examples in the main README:

- [How untying helps](how_untying_helps.ipynb): training setup for the five-feature untied example.
- [Toy Models replication](toy_models_sanity_check.ipynb): training and geometry checks for the tied pentagon example.

## Supporting regularization experiments

- [Weight decay](weight_decay_results/README.md): saved curve, weights, and reproduction commands.
- [One-sided unit norms](one_sided_norm_results/REPORT.md): multistart normalization search.

The untied improvement over tied weights appears as about 40.5% in `scripts/verify_results.py`
and at zero decay in the weight-decay sweep, 40.6% in the one-sided search, and about 42% in the
notebook output. Each figure is measured on its own random test draw, and the two searches and
the notebook also evaluate their own trained or re-selected pair of models.

The notebooks train new models when executed. The [reproduction guide](../../REPRODUCING.md)
provides a quick check using the supplied weights instead.
