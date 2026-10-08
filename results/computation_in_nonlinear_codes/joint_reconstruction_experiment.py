"""Matched XOR/reconstruction sweep; no readability term or readability selection.

The encoder is frozen. Positive reconstruction weights train the residual circuit,
XOR head and decoder together. At weight zero the decoder follows the changing
code with a stopped gradient, so it cannot influence the XOR circuit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import minimal_learned_example as base
from focused_stage2_experiment import initialize, unpack_transition
from focused_stage2_decoder import forward_decoder, pack_decoder, unpack_decoder

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "joint_reconstruction"


def load_code(seed):
    """Use portable, already-standardized weights, rather than ignored .pt files."""
    source = json.loads((HERE / "joint_training_initial_codes.json").read_text())
    saved = source['encoders'][str(seed)]
    encoder, decoder = base.MLP(4, 2, 4), base.MLP(2, 4, 4)
    for model, key in [(encoder, 'encoder'), (decoder, 'd0')]:
        model.load_state_dict({k: torch.tensor(v) for k, v in saved[key].items()})
        model.requires_grad_(False)
    with torch.no_grad():
        z0 = encoder(base.truth_table())
        assert base.exact_pattern(decoder(z0), base.truth_table()) == 1.
    return encoder, decoder, z0


def tensor_state(model):
    return {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}


def task_state(row):
    layers, hw, hb = unpack_transition(row[None])
    state = {}
    for k, (a, ba, b, bb) in enumerate(layers):
        for name, value in [(f"ins.{k}.weight", a[0]), (f"ins.{k}.bias", ba[0, 0]),
                            (f"outs.{k}.weight", b[0]), (f"outs.{k}.bias", bb[0, 0])]:
            state[name] = value.tolist()
    return state, {"weight": hw[0].tolist(), "bias": hb[0, 0].tolist()}


def decoder_state(row):
    state = {}
    for k, (w, b) in zip((0, 2, 4), unpack_decoder(row[None])):
        state[f"net.{k}.weight"] = w[0].tolist()
        state[f"net.{k}.bias"] = b[0, 0].tolist()
    return state


def forward_task(parameters, z):
    layers, hw, hb = unpack_transition(parameters)
    for a, ba, b, bb in layers:
        z = z + (z @ a.transpose(1, 2) + ba).relu() @ b.transpose(1, 2) + bb
    return z @ hw.transpose(1, 2) + hb, z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--runs-per-code", type=int, default=64)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--weights", type=float, nargs="+", default=[0., .1, 1.])
    parser.add_argument("--branch-scales", type=float, nargs=4, default=[.1, .3, .7, 1.])
    parser.add_argument("--decoder-init", choices=["mixed", "d0"], default="mixed")
    parser.add_argument("--standardize-decoder", action="store_true")
    parser.add_argument("--checkpoint-selection", choices=["loss", "accuracy"], default="loss")
    parser.add_argument("--decoder-rate-multiplier", type=float, default=1.)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    assert args.runs_per_code % 16 == 0 and args.steps > 0
    assert args.decoder_rate_multiplier > 0
    assert all(v >= 0 for v in args.weights) and len(set(args.weights)) == len(args.weights)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "trained-weights.json").exists():
        raise FileExistsError("Use a new --output directory to preserve a completed sweep.")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    codes = [load_code(seed) for seed in [4, 9]]
    x = base.truth_table()
    y = base.xor_targets(x)
    z_base = torch.stack([c[2] for c in codes]).repeat_interleave(args.runs_per_code, 0)
    count = len(z_base)
    rates_base = torch.tensor([.001, .003, .01, .03]).repeat_interleave(args.runs_per_code // 4).repeat(2)
    scales = torch.tensor(args.branch_scales).repeat(count // 4)
    initial_task = initialize(z_base, scales)
    initial_decoder = torch.empty(count, 52)
    for w, b in unpack_decoder(initial_decoder):
        bound = 1 / math.sqrt(w.shape[-1])
        w.uniform_(-bound, bound)
        b.uniform_(-bound, bound)
    # One matched half starts from the original decoder, one from random weights.
    for i in range(count):
        if i % 2 or args.decoder_init == "d0":
            initial_decoder[i] = pack_decoder(codes[i // args.runs_per_code][1])
    task = initial_task.repeat(len(args.weights), 1).requires_grad_()
    decoder = initial_decoder.repeat(len(args.weights), 1).requires_grad_()
    z0 = z_base.repeat(len(args.weights), 1, 1)
    lambdas = torch.tensor(args.weights).repeat_interleave(count)
    rates = rates_base.repeat(len(args.weights))
    targets = x[None].expand(len(task), -1, -1)
    xor_targets = y[None].expand(len(task), -1, -1)
    manifest = {
        "encoder_seeds": [4, 9], "runs_per_encoder_per_weight": args.runs_per_code,
        "reconstruction_weights": args.weights, "total_training_runs": len(task),
        "steps": args.steps, "seed": args.seed, "dtype": "float64", "optimizer": "Adam",
        "learning_rates": [.001, .003, .01, .03], "branch_scales": args.branch_scales,
        "schedule": "initial learning rate for first 75% of steps; one tenth thereafter",
        "decoder_initialization": ("all copied from D0" if args.decoder_init == "d0" else
                                   "even within-arm model indices random; odd copied from D0"),
        "decoder_standardization": ("Recompute coordinate means/stds over all 16 codes each step; detach these statistics for gradients; fold the affine map into saved decoder weights"
                                    if args.standardize_decoder else "none"),
        "paired_arms": "Same initial task and decoder parameters for each reconstruction weight",
        "positive_weight_objective": "mean XOR BCE + lambda * mean four-bit reconstruction BCE",
        "zero_weight_control": "XOR-only circuit; decoder learns reconstruction with z4 detached",
        "checkpoint_selection": ("Maximize inputs with all six outputs correct, then total correct task-patterns, then minimize joint loss; zero-weight control maximizes XOR pattern accuracy then minimizes XOR loss; checked every 250 steps including endpoints; no readability criterion"
                                 if args.checkpoint_selection == "accuracy" else
                                 "Minimum circuit objective at step 0 and every 250 steps, including last; no readability criterion"),
        "decoder_rate_multiplier": args.decoder_rate_multiplier,
        "initial_encoder": "Previously trained reconstruction-only encoders, seeds 4 and 9, frozen throughout",
        "encoder_architecture": [4, 4, 4, 2],
        "final_decoder_architecture": [2, 4, 4, 4],
        "residual_blocks": 4, "state_width": 2, "branch_width": 2,
        "checkpoint_rule": "All reported accuracies and readability refer to the same saved checkpoint per run",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "training-source.py").write_bytes(Path(__file__).read_bytes())
    best_loss = torch.full((len(task),), float("inf"))
    best_task = task.detach().clone()
    best_decoder = decoder.detach().clone()
    best_step = torch.zeros(len(task), dtype=torch.int64)
    best_accuracy = torch.full((len(task),), -1, dtype=torch.int64)
    moments = [(torch.zeros_like(p), torch.zeros_like(p)) for p in [task, decoder]]
    history = []
    started = time.monotonic()

    def decoder_coordinates(z):
        if not args.standardize_decoder:
            return z
        mean = z.detach().mean(1, keepdim=True)
        scale = z.detach().std(1, keepdim=True).clamp_min(1e-6)
        return (z - mean) / scale

    def export_decoder(parameters, z):
        exported = parameters.detach().clone()
        if args.standardize_decoder:
            mean = z.detach().mean(1, keepdim=True)
            scale = z.detach().std(1, keepdim=True).clamp_min(1e-6)
            first_w, first_b = unpack_decoder(exported)[0]
            first_w.div_(scale)
            first_b.sub_(mean @ first_w.transpose(1, 2))
        return exported

    def losses():
        xor, z4 = forward_task(task, z0)
        decoder_input = torch.where((lambdas == 0)[:, None, None], z4.detach(), z4)
        reconstructed = forward_decoder(decoder, decoder_coordinates(decoder_input))
        task_loss = F.binary_cross_entropy_with_logits(xor, xor_targets, reduction="none").mean((1, 2))
        reconstruction_loss = F.binary_cross_entropy_with_logits(reconstructed, targets, reduction="none").mean((1, 2))
        objective = task_loss + lambdas * reconstruction_loss
        # This extra term updates only the decoder in the zero-weight control.
        optimization_loss = objective + (lambdas == 0) * reconstruction_loss
        return optimization_loss, objective, xor, reconstructed

    for step in range(args.steps + 1):
        if step:
            loss, _, _, _ = losses()
            loss.sum().backward()
            with torch.no_grad():
                lr = rates * (1 if step < args.steps * .75 else .1)
                for parameter, (m, v), multiplier in zip([task, decoder], moments, [1., args.decoder_rate_multiplier]):
                    m.mul_(.9).add_(parameter.grad, alpha=.1)
                    v.mul_(.999).addcmul_(parameter.grad, parameter.grad, value=.001)
                    parameter.add_(-multiplier * lr[:, None] * (m / (1 - .9 ** step)) /
                                   ((v / (1 - .999 ** step)).sqrt() + 1e-8))
                    parameter.grad = None
        if step % 250 == 0 or step == args.steps:
            with torch.no_grad():
                _, objective, xor, reconstructed = losses()
                assert torch.isfinite(objective).all(), f"Non-finite objective at step {step}"
                improved = objective < best_loss
                if args.checkpoint_selection == "accuracy":
                    xor_ok = ((xor > 0) == xor_targets.bool()).all(-1)
                    rec_ok = ((reconstructed > 0) == targets.bool()).all(-1)
                    # 33 exceeds the maximum possible secondary score (32).
                    score = torch.where(lambdas == 0, xor_ok.sum(-1),
                                        33 * (xor_ok & rec_ok).sum(-1) + xor_ok.sum(-1) + rec_ok.sum(-1))
                    improved = (score > best_accuracy) | ((score == best_accuracy) & improved)
                    best_accuracy[improved] = score[improved]
                best_loss[improved] = objective[improved]
                best_task[improved] = task[improved]
                # Saved models are ordinary pointwise MLPs, with no dependency
                # on the other inputs in a batch at evaluation time.
                _, current_z4 = forward_task(task, z0)
                best_decoder[improved] = export_decoder(decoder, current_z4)[improved]
                best_step[improved] = step
            if step % 2000 == 0 or step == args.steps:
                with torch.no_grad():
                    xor, z4 = forward_task(best_task, z0)
                    reconstructed = forward_decoder(best_decoder, z4)
                    good_xor = ((xor > 0) == xor_targets.bool()).all((1, 2))
                    good_rec = ((reconstructed > 0) == targets.bool()).all((1, 2))
                record = {"step": step, "elapsed_seconds": round(time.monotonic() - started, 2),
                          "by_weight": [{"lambda": weight,
                              "xor_correct": int(good_xor[k*count:(k+1)*count].sum()),
                              "reconstruction_correct": int(good_rec[k*count:(k+1)*count].sum()),
                              "both_correct": int((good_xor & good_rec)[k*count:(k+1)*count].sum())}
                              for k, weight in enumerate(args.weights)]}
                history.append(record)
                print(json.dumps(record), flush=True)
    rows = []
    for i in range(len(task)):
        local = i % count
        transition, head = task_state(best_task[i])
        rows.append({"id": i, "matched_run": local, "encoder_seed": [4, 9][local // args.runs_per_code],
                     "lambda": args.weights[i // count], "learning_rate": float(rates[i]),
                     "branch_scale": float(scales[local]),
                     "decoder_initialization": "D0" if local % 2 or args.decoder_init == "d0" else "random",
                     "checkpoint_step": int(best_step[i]), "selection_loss": float(best_loss[i]),
                     "transition": transition, "head": head, "d4": decoder_state(best_decoder[i])})
    output = {"protocol": manifest,
              "encoders": {str(seed): {"encoder": tensor_state(codes[k][0]), "d0": tensor_state(codes[k][1])}
                           for k, seed in enumerate([4, 9])}, "runs": rows}
    (args.output / "trained-weights.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.output / "training-history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Saved {len(rows)} independent runs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
