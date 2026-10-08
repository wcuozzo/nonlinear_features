"""Width-matched encoder/decoder training, using the recipe in minimal_learned_example.py.

Stage 1: train the 4→2→4 autoencoder for one seed (--compressor-only); without that
flag it also trains an XOR circuit. --audit-only needs a .pt file from a previous run.
Certificates use exact rational arithmetic on the stored float32 activations.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
import sympy as sp
import torch
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression

import minimal_learned_example as original


def stabilize_code(encoder, decoder, x):
    """Absorb a change of coordinates into existing affine layers; no new layer."""
    with torch.no_grad():
        code = encoder(x)
        mean, scale = code.mean(0), code.std(0)
        scale[scale < 1e-6] = 1
        encoder.net[-1].weight.div_(scale[:, None])
        encoder.net[-1].bias.sub_(mean).div_(scale)
        first = decoder.net[0]
        first.bias.add_(first.weight @ mean)
        first.weight.mul_(scale[None, :])


def train_stabilized(x, z0, original_decoder, progress):
    """Same loss/budget; identity residual initialization with data-spanning hinges."""
    y = original.xor_targets(x)
    candidates = []
    for restart in range(10):
        torch.manual_seed(100000 + original.SEED + 10000 * restart + 100 * original.DEPTH)
        transition, head = original.Transition(), torch.nn.Linear(2, 2)
        decoder = copy.deepcopy(original_decoder)
        for p in decoder.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            for inside, outside in zip(transition.ins, transition.outs):
                projections = z0 @ inside.weight.T
                inside.bias.copy_(-torch.stack([torch.quantile(projections[:, j], q)
                                               for j, q in enumerate([0.25, 0.75])]))
                outside.weight.zero_()
                outside.bias.zero_()
        optimizer = torch.optim.Adam([*transition.parameters(), *head.parameters(), *decoder.parameters()], lr=0.003)
        for _ in range(10000):
            z4 = transition(z0)
            task_loss = torch.nn.functional.binary_cross_entropy_with_logits(head(z4), y)
            input_loss = torch.nn.functional.binary_cross_entropy_with_logits(decoder(z4), x)
            loss = task_loss + 0.1 * input_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            z4 = transition(z0)
            task_exact = original.exact_pattern(head(z4), y)
            input_exact = original.exact_pattern(decoder(z4), x)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(head(z4), y)
                    + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(decoder(z4), x)).item()
        selection = (-(task_exact + input_exact), loss)
        candidates.append((selection, transition, head, decoder))
        progress({"restart": restart, "task_exact": task_exact, "input_exact": input_exact, "loss": loss})
    _, transition, head, decoder = min(candidates, key=lambda c: c[0])
    return transition, head, decoder


def certify(values, labels, exact_values=None):
    """Return an independently checkable separator or convex-hull intersection."""
    values = np.asarray(values, dtype=float)
    rational_values = ([[sp.Rational(float(v)) for v in row] for row in values]
                       if exact_values is None else [[sp.Rational(v) for v in row] for row in exact_values])
    labels = np.asarray(labels, dtype=int)
    mean, scale = values.mean(0), values.std(0)
    scale[scale < 1e-10] = 1
    standardized = (values - mean) / scale
    signs = 2 * labels - 1
    design = np.column_stack((standardized, np.ones(len(values))))
    separator = linprog(np.zeros(design.shape[1]), A_ub=-signs[:, None] * design,
                        b_ub=-np.ones(len(values)), bounds=[(None, None)] * design.shape[1],
                        method="highs")
    if separator.success:
        w = separator.x[:-1] / scale
        b = separator.x[-1] - mean @ w
        coefficients = [sp.Rational(float(v)) for v in [*w, b]]
        margins = [int(sign) * sum(c * v
                   for c, v in zip(coefficients, [*row, 1]))
                   for row, sign in zip(rational_values, signs)]
        assert min(margins) > 0, "Numerical separator failed exact verification"
        return {"separable": True, "coefficients": list(map(str, coefficients)),
                "minimum_exact_margin": str(min(margins))}

    pos, neg = np.flatnonzero(labels), np.flatnonzero(1 - labels)
    order = np.concatenate((pos, neg))
    eq = np.vstack((np.concatenate((standardized[pos].T, -standardized[neg].T), axis=1),
                    np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                    np.r_[np.zeros(len(pos)), np.ones(len(neg))]))
    target = np.r_[np.zeros(values.shape[1]), 1, 1]
    hull = linprog(np.zeros(len(values)), A_eq=eq, b_eq=target,
                   bounds=(0, None), method="highs")
    assert hull.success, f"No verified certificate: separator {separator.status}, hull {hull.status}"
    support = np.flatnonzero(hull.x > 1e-10)
    columns = []
    for j in support:
        is_pos = j < len(pos)
        columns.append([v * (1 if is_pos else -1)
                        for v in rational_values[order[j]]] + [int(is_pos), int(not is_pos)])
    matrix = sp.Matrix.hstack(*(sp.Matrix(c) for c in columns))
    exact_target = sp.Matrix([0] * values.shape[1] + [1, 1])
    try:
        weights, parameters = matrix.gauss_jordan_solve(exact_target)
        weights = weights.subs({p: 0 for p in parameters})
        assert all(w >= 0 for w in weights)
        assert matrix * weights == exact_target
    except (ValueError, AssertionError):
        # Near-degenerate float32 states can fool numerical feasibility checks.
        # Solve the full rational problem instead of treating that as nonseparability.
        support = np.arange(len(values))
        columns = [[v * (1 if labels[index] else -1)
                    for v in rational_values[index]] + [int(labels[index]), int(1-labels[index])]
                   for index in order]
        matrix = sp.Matrix.hstack(*(sp.Matrix(c) for c in columns))
        # Row reduction exposes constraints hidden by tiny floating-point residuals.
        reduced, _ = matrix.row_join(exact_target).rref()
        rows = [i for i in range(reduced.rows) if any(reduced.row(i))]
        reduced = reduced[rows, :]
        numerical = np.asarray(reduced.tolist(), dtype=float)
        row_scale = np.maximum(np.abs(numerical).max(axis=1), 1)
        numerical /= row_scale[:, None]
        refined = linprog(np.zeros(len(values)), A_eq=numerical[:, :-1],
                          b_eq=numerical[:, -1], bounds=(0, None), method="highs")
        if not refined.success:
            return {"separable": None, "reason": "Near-degenerate certificate unresolved",
                    "numerical_separable": False}
        support = np.flatnonzero(refined.x > 1e-10)
        matrix = matrix[:, list(support)]
        try:
            weights, parameters = matrix.gauss_jordan_solve(exact_target)
            weights = weights.subs({p: 0 for p in parameters})
            assert all(w >= 0 for w in weights) and matrix * weights == exact_target
        except (ValueError, AssertionError):
            return {"separable": None, "reason": "Exact hull verification unresolved",
                    "numerical_separable": False}
    return {"separable": False, "exact_hull_support": [
        {"input_index": int(order[j]), "label": int(labels[order[j]]), "weight": str(w)}
        for j, w in zip(support, weights) if w > 0]}


def logistic_r2(values, labels):
    values = np.asarray(values, dtype=float)
    scale = values.std(0)
    scale[scale < 1e-10] = 1
    standardized = (values - values.mean(0)) / scale
    scores, converged = [], []
    for y in labels.T:
        with warnings.catch_warnings(record=True) as caught:
            model = LogisticRegression(C=float("inf"), max_iter=20000, tol=1e-12)
            model.fit(standardized, y)
        p = model.predict_proba(standardized)[:, 1]
        scores.append(float(1 - np.square(y - p).sum() / np.square(y - y.mean()).sum()))
        converged.append(not any(isinstance(w.message, sklearn.exceptions.ConvergenceWarning)
                                 for w in caught))
    return scores, converged


def evaluate(encoder, d0, transition, head, d4):
    x = original.truth_table()
    y = original.xor_targets(x)
    with torch.no_grad():
        z0 = encoder(x)
        trace = transition.trace(z0)
        z4 = trace[-1][1]
        task_logits, input_logits = head(z4), d4(z4)
        initial_input_logits = d0(z0)
        initial_exact = original.exact_pattern(initial_input_logits, x)
    bits = x.numpy().astype(int)
    interfaces = []
    for i in range(original.DEPTH):
        values = torch.cat((trace[2 * i][1], trace[2 * i + 1][1]), dim=1).numpy()
        certs = [certify(values, bits[:, j]) for j in range(4)]
        r2, converged = logistic_r2(values, bits)
        interfaces.append({"name": f"z{i}+h{i+1}", "activations": values.tolist(),
                           "input_bits_linearly_separable": [c["separable"] for c in certs],
                           "certificates": certs, "logistic_probability_r2": r2,
                           "logistic_converged": converged})
    target_audits = []
    for i in range(original.DEPTH + 1):
        values = trace[2 * i][1].numpy()
        certs = [certify(values, y[:, j].numpy()) for j in range(2)]
        target_audits.append({"name": f"z{i}", "activations": values.tolist(),
                              "xor_linearly_separable": [c["separable"] for c in certs],
                              "certificates": certs})
    persistent = [j for j in range(4) if all(r["input_bits_linearly_separable"][j] is False
                                           for r in interfaces)]
    exact_task = original.exact_pattern(task_logits, y)
    exact_input = original.exact_pattern(input_logits, x)
    every_pair = all(any(j in persistent for j in pair) for pair in [(0, 1), (2, 3)])
    new_readout = all(v is False for v in target_audits[0]["xor_linearly_separable"])
    return {
        "input_patterns": bits.tolist(), "xor_targets": y.int().tolist(),
        "initial_input_exact": initial_exact,
        "initial_input_logits": initial_input_logits.tolist(),
        "initial_affine_fvu": original.affine_fvu(x, z0),
        "initial_xor_linearly_separable": target_audits[0]["xor_linearly_separable"],
        "final_task_exact": exact_task, "final_input_exact": exact_input,
        "task_logits": task_logits.tolist(), "input_logits": input_logits.tolist(),
        "interfaces": interfaces, "residual_target_audits": target_audits,
        "persistent_nonlinear_input_bits": persistent,
        "every_xor_has_persistent_nonlinear_input": every_pair,
        "same_x1_x3_persist": all(j in persistent for j in [1, 3]),
        "full_result_passes": bool(initial_exact == exact_task == exact_input == 1
                                   and new_readout and every_pair),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-width", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--compressor-only", action="store_true")
    parser.add_argument("--reuse-compressor", action="store_true")
    parser.add_argument("--stabilized", action="store_true")
    args = parser.parse_args()
    original.SEED = args.seed
    folder = Path(__file__).parent / "results" / "width4_codec"
    folder.mkdir(exist_ok=True)
    stem = folder / f"width-{args.hidden_width}-seed-{args.seed}"
    compressor_checkpoint = stem.with_name(stem.name + "-compressor.pt")
    if args.stabilized:
        stem = stem.with_name(stem.name + "-stabilized")
    checkpoint = stem.with_suffix(".pt")
    started = time.monotonic()
    x = original.truth_table()
    if args.audit_only:
        saved = torch.load(checkpoint, weights_only=True)
        encoder = original.MLP(4, 2, args.hidden_width)
        d0, d4 = [original.MLP(2, 4, args.hidden_width) for _ in range(2)]
        transition, head = original.Transition(), torch.nn.Linear(2, 2)
        for key, model in [("encoder", encoder), ("d0", d0), ("d4", d4),
                           ("transition", transition), ("head", head)]:
            model.load_state_dict(saved[key])
    else:
        print(f"Training compressor: seed={args.seed}, hidden_width={args.hidden_width}", flush=True)
        if args.reuse_compressor:
            compressor_saved = torch.load(compressor_checkpoint, weights_only=True)
            encoder = original.MLP(4, 2, args.hidden_width)
            d0 = original.MLP(2, 4, args.hidden_width)
            encoder.load_state_dict(compressor_saved["encoder"])
            d0.load_state_dict(compressor_saved["d0"])
            for p in [*encoder.parameters(), *d0.parameters()]:
                p.requires_grad_(False)
            loss = compressor_saved["loss"]
        else:
            encoder, d0, loss = original.train_compressor(x, hidden_width=args.hidden_width)
            torch.save({"encoder": encoder.state_dict(), "d0": d0.state_dict(), "loss": loss},
                       compressor_checkpoint)
        with torch.no_grad():
            z0 = encoder(x)
            initial_exact = original.exact_pattern(d0(z0), x)
        compressor_result = {"stage": "compressor", "seed": args.seed,
                          "hidden_width": args.hidden_width, "input_exact": initial_exact,
                          "loss": loss, "affine_fvu": original.affine_fvu(x, z0),
                          "xor_separable": original.separability_by_column(z0, original.xor_targets(x))}
        print(json.dumps(compressor_result), flush=True)
        stem.with_name(stem.name + "-compressor.json").write_text(json.dumps(compressor_result, indent=2) + "\n")
        if args.compressor_only:
            return
        restart_results = []
        def progress(r):
            restart_results.append(r)
            print(json.dumps({"stage": "computation", **r}), flush=True)
        if args.stabilized:
            stabilize_code(encoder, d0, x)
            with torch.no_grad():
                z0 = encoder(x)
                assert original.exact_pattern(d0(z0), x) == initial_exact
            transition, head, d4 = train_stabilized(x, z0, d0, progress)
        else:
            transition, head, d4 = original.train_computation(
                x, original.xor_targets(x), z0, d0, progress=progress)
        saved = {name: model.state_dict() for name, model in
                 [("encoder", encoder), ("d0", d0), ("d4", d4), ("transition", transition), ("head", head)]}
        saved["training_seconds"] = time.monotonic() - started
        saved["restart_results"] = restart_results
        torch.save(saved, checkpoint)
    result = evaluate(encoder, d0, transition, head, d4)
    result["restart_results"] = saved.get("restart_results", [])
    result["protocol"] = {"seed": args.seed, "hidden_width": args.hidden_width,
                          "compressor_restarts": 3, "compressor_steps": 5000,
                          "task_restarts": 10, "task_steps": 10000,
                          "learning_rate": 0.003, "reconstruction_weight": 0.1,
                          "stabilized": args.stabilized,
                          "torch": torch.__version__, "numpy": np.__version__,
                          "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                          "platform": platform.platform(),
                          "training_seconds": saved["training_seconds"]}
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in [
        "initial_input_exact", "initial_xor_linearly_separable", "final_task_exact", "final_input_exact",
        "persistent_nonlinear_input_bits", "every_xor_has_persistent_nonlinear_input", "full_result_passes"]}), flush=True)
    print(f"Saved {stem.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
